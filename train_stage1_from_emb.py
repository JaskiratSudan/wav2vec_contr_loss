import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Dataset
from tqdm import tqdm

from compression_module import CompressionModule
from loss import SupConBinaryLoss

# =======================
#        CONFIG
# =======================
# Paths to the .npy files you just created
DATA_ROOT = "/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings"

TRAIN_EMB_PATH = os.path.join(DATA_ROOT, "train_embeddings_mean.npy")
TRAIN_LAB_PATH = os.path.join(DATA_ROOT, "train_labels.npy")
DEV_EMB_PATH = os.path.join(DATA_ROOT, "dev_embeddings_mean.npy")
DEV_LAB_PATH = os.path.join(DATA_ROOT, "dev_labels.npy")

INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

EPOCHS = 100
BATCH_SIZE = 256
LR = 5e-3
WEIGHT_DECAY = 3e-3
TEMPERATURE = 0.2
NUM_WORKERS = 0
SEED = 1337
SAVE_DIR = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/precomputed"

# ---- Uniformity regularizer ----
UNIFORMITY_WEIGHT = 0.2
UNIFORMITY_T = 2.0
SUPCON_SIMILARITY = "geodesic"  # "cosine" or "geodesic"

# ---- Hard-neg difficulty knobs ----
TOPK_NEG = 15
WARMUP_EPOCHS = 8
ALPHA_END = 1
ALPHA_RAMP_EPOCHS = 80

# =======================
#      HELPERS
# =======================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class NumpyMemmapDataset(Dataset):
    """
    Reads directly from the disk-based .npy files without loading them into RAM.
    """
    def __init__(self, emb_path, lab_path):
        # mmap_mode='r' keeps the array on disk
        self.embeddings = np.load(emb_path, mmap_mode='r')
        # Labels are small enough to load into RAM for faster sampling logic
        self.labels = np.load(lab_path) 
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Read specific slice from disk
        x = torch.from_numpy(self.embeddings[index].copy()) # Shape: (1024, 250)
        y = torch.tensor(self.labels[index], dtype=torch.long)
        return x, y

class BalancedBatchSampler(Sampler):
    """
    Balanced sampler updated to work with the loaded numpy labels.
    """
    def __init__(self, labels_array, batch_size: int):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.labels = labels_array
        
        # Identify indices
        self.real = np.where(self.labels == 1)[0]
        self.fake = np.where(self.labels == 0)[0]
        
        self.per_class = batch_size // 2
        self.num_batches = min(len(self.real)//self.per_class, len(self.fake)//self.per_class)

    def __len__(self): 
        return self.num_batches

    def __iter__(self):
        # Shuffle indices every epoch
        np.random.shuffle(self.real)
        np.random.shuffle(self.fake)
        
        r = self.real[: self.num_batches * self.per_class]
        f = self.fake[: self.num_batches * self.per_class]
        
        for b in range(self.num_batches):
            batch_r = r[b*self.per_class : (b+1)*self.per_class]
            batch_f = f[b*self.per_class : (b+1)*self.per_class]
            idx = np.concatenate([batch_r, batch_f])
            np.random.shuffle(idx)
            yield idx.tolist()

def alpha_for_epoch(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS: return 0.0
    t = min(1.0, (epoch - WARMUP_EPOCHS) / max(1, ALPHA_RAMP_EPOCHS))
    return t * ALPHA_END

# =======================
#       TRAIN / DEV
# =======================
def train_one_epoch(head, loss_fn, loader, optimizer, device, epoch):
    head.train()
    total, steps = 0.0, 0
    alpha = alpha_for_epoch(epoch)
    
    # Wrap loader with tqdm for the progress bar
    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch")
    
    for embeddings, labels in pbar:
        embeddings = embeddings.to(device) # (B, 1024, 250)
        labels = labels.to(device)

        hs = embeddings.unsqueeze(1) 

        seq = head(hs)                                     
        z = F.normalize(seq.mean(dim=-1), p=2, dim=1)      

        loss = loss_fn(z, labels, topk_neg=TOPK_NEG, alpha=alpha)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()

        total += loss.item(); steps += 1
        
        # Update the progress bar description with current loss
        pbar.set_postfix(loss=total/steps, alpha=alpha)
        
    return total / max(1, steps), alpha

@torch.no_grad()
def evaluate(head, loss_fn, loader, device):
    head.eval()
    total, steps = 0.0, 0
    for embeddings, labels in loader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)
        
        hs = embeddings.unsqueeze(1) # (B, 1, 1024, 250)
        
        seq = head(hs)
        z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        
        loss = loss_fn(z, labels, topk_neg=TOPK_NEG, alpha=0.0)
        total += loss.item(); steps += 1
    return total / max(1, steps)

def main():
    MODEL_NAME = "facebook/wav2vec2-xls-r-300m" # Only for naming logs now
    RUN_TAG = "precomputed_mean_" + MODEL_NAME.replace("/", "__")

    set_seed(SEED)
    global SAVE_DIR
    SAVE_DIR = os.path.join(SAVE_DIR, RUN_TAG)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Training on precomputed features: {DATA_ROOT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Setup Datasets ---
    train_ds = NumpyMemmapDataset(TRAIN_EMB_PATH, TRAIN_LAB_PATH)
    dev_ds = NumpyMemmapDataset(DEV_EMB_PATH, DEV_LAB_PATH)

    # --- Setup Loaders ---
    # Note: No 'collate_fn' needed because data is already fixed size (1024, 250)
    train_loader = DataLoader(
        train_ds, 
        batch_sampler=BalancedBatchSampler(train_ds.labels, BATCH_SIZE),
        num_workers=NUM_WORKERS, 
        pin_memory=False
    )
    
    dev_loader = DataLoader(
        dev_ds, 
        batch_sampler=BalancedBatchSampler(dev_ds.labels, BATCH_SIZE),
        num_workers=NUM_WORKERS, 
        pin_memory=False
    )

    # --- Model ---
    # CompressionModule expects (InputDim, HiddenDim). 
    # It must handle the (B, 1, 1024, T) input we provide.
    head = CompressionModule(INPUT_DIM, HIDDEN_DIM, DROPOUT).to(device)

    loss_fn = SupConBinaryLoss(
        temperature=TEMPERATURE,
        similarity=SUPCON_SIMILARITY,
        uniformity_weight=UNIFORMITY_WEIGHT,
        uniformity_t=UNIFORMITY_T,
    )

    optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best, best_path = float("inf"), None
    print("Start Training...")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, alpha = train_one_epoch(head, loss_fn, train_loader, optim, device, epoch)
        dev_loss = evaluate(head, loss_fn, dev_loader, device)
        
        print(f"[epoch {epoch:03d}] alpha={alpha:.2f} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}")

        if dev_loss < best:
            best = dev_loss
            best_path = os.path.join(SAVE_DIR, f"best_model.pt")
            torch.save({
                "epoch": epoch,
                "compression_state_dict": head.state_dict(),
                "train_loss": train_loss,
                "dev_loss": dev_loss,
            }, best_path)
            print(f"✓ Saved best -> {best_path} (dev={best:.4f})")

if __name__ == "__main__":
    main()
