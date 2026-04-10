"""
train_stage1_cached.py

Trains the CompressionModule (projection head) with SupCon loss on
pre-extracted XLSR feature caches.

Uses an IterableDataset that streams one shard (~5000 samples, ~10MB)
at a time — only one shard is in memory during training, not the full
86K sample dataset.

Env vars
--------
CACHE_DIR         : directory containing xlsr_train.pt and xlsr_dev.pt (required)
SAVE_DIR          : where to save the best head checkpoint (required)
RESUME_HEAD_CKPT  : path to a CompressionModule checkpoint to warm-start from
BATCH_SIZE        : (default: 512)
EPOCHS            : (default: 50)
HEAD_LR           : (default: 5e-4)
WEIGHT_DECAY      : (default: 3e-3)
PATIENCE          : early stopping patience (default: 10)
TEMPERATURE       : SupCon temperature (default: 0.07)
SIMILARITY        : cosine | geodesic (default: geodesic)
UNIFORMITY_WEIGHT : (default: 0.0)
UNIFORMITY_T      : (default: 2.0)
FF_SEED           : random seed (default: 42)
"""

import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from compression_module import CompressionModule
from loss import SupConBinaryLoss
from stage1_utils import set_seed


# ---------------------------------------------------------------------------
# IterableDataset — streams one shard at a time
# ---------------------------------------------------------------------------

class ShardedXLSRDataset(torch.utils.data.IterableDataset):
    """
    Reads shard manifest (xlsr_{split}.pt) and yields samples
    one shard at a time. Peak RAM = one shard (~10MB), not the full dataset.

    For training: shuffles shard order and samples within each shard each epoch.
    For dev/eval: fixed order for reproducibility.
    """
    def __init__(self, cache_path: str, shuffle: bool = False, seed: int = 42):
        manifest          = torch.load(cache_path, map_location="cpu")
        self.shard_paths  = manifest["shard_paths"]
        self.n_samples    = manifest["n_samples"]
        self.shuffle      = shuffle
        self.seed         = seed
        self.epoch        = 0

        # Build a lightweight label list (just ints) for reporting
        print(f"Reading labels from {len(self.shard_paths)} shards...")
        labels = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            labels.extend(shard["labels"].tolist())
        bon = sum(1 for l in labels if l == 1)
        spo = len(labels) - bon
        print(f"  {len(labels)} samples  bonafide={bon}  spoof={spo}")
        self.labels = labels   # kept for loss_floor diagnostics only

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng          = random.Random(self.seed + self.epoch)
        shard_order  = list(range(len(self.shard_paths)))
        if self.shuffle:
            rng.shuffle(shard_order)

        for si in shard_order:
            shard   = torch.load(self.shard_paths[si], map_location="cpu")
            feats   = shard["features"]           # List[Tensor(1024, T_i)] fp16
            labels  = shard["labels"].tolist()
            sources = shard["sources"]

            idx_order = list(range(len(labels)))
            if self.shuffle:
                rng.shuffle(idx_order)

            for i in idx_order:
                feat = feats[i].float()
                # Ensure (1024, T) — shards may be saved as (T, 1024)
                if feat.shape[0] != 1024:
                    feat = feat.T
                yield feat, torch.tensor(labels[i], dtype=torch.long), sources[i]

    def __len__(self):
        return self.n_samples


# ---------------------------------------------------------------------------
# Collate — pad variable-length (1024, T_i) tensors in a batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    features, labels, sources = zip(*batch)
    max_t  = max(f.shape[1] for f in features)
    padded = torch.zeros(len(features), 1024, max_t)
    for i, f in enumerate(features):
        padded[i, :, :f.shape[1]] = f
    return padded, torch.stack(list(labels)), sources


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------

def run_one_epoch(head, loss_fn, loader, optimizer, device, train: bool):
    head.train() if train else head.eval()
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for features, labels, *_ in loader:
            features = features.to(device)          # (B, 1024, T)
            labels   = labels.to(device)

            # CompressionModule expects (B, K, F, T) — unsqueeze K=1 since we already averaged layers
            out = head(features.unsqueeze(1))   # (B, 1, 1024, T) → forward → (B, 256, T)
            embeddings = out.mean(dim=-1)        # mean-pool over T → (B, 256)
            embeddings = F.normalize(embeddings, dim=-1)

            loss = loss_fn(embeddings, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cache_dir   = Path(os.environ.get("CACHE_DIR",  ""))
    save_dir    = Path(os.environ.get("SAVE_DIR",   ""))
    resume_ckpt = os.environ.get("RESUME_HEAD_CKPT", "")

    if not cache_dir:
        raise ValueError("CACHE_DIR must be set")
    if not save_dir:
        raise ValueError("SAVE_DIR must be set")

    save_dir.mkdir(parents=True, exist_ok=True)

    batch_size   = int(os.environ.get("BATCH_SIZE",          "512"))
    epochs       = int(os.environ.get("EPOCHS",              "50"))
    head_lr      = float(os.environ.get("HEAD_LR",           "5e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY",      "3e-3"))
    patience     = int(os.environ.get("PATIENCE",            "10"))
    temperature  = float(os.environ.get("TEMPERATURE",       "0.07"))
    similarity   = os.environ.get("SIMILARITY",              "geodesic")
    unif_weight  = float(os.environ.get("UNIFORMITY_WEIGHT", "0.0"))
    unif_t       = float(os.environ.get("UNIFORMITY_T",      "2.0"))
    seed         = int(os.environ.get("FF_SEED",             "42"))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice  : {device}")
    print(f"Cache   : {cache_dir}")
    print(f"Config  : batch={batch_size}  lr={head_lr}  tau={temperature}  "
          f"sim={similarity}  epochs={epochs}\n")

    # ------------------------------------------------------------------
    # Datasets — stream shards, not full data in RAM
    # ------------------------------------------------------------------
    print("--- Train dataset ---")
    train_ds = ShardedXLSRDataset(
        str(cache_dir / "xlsr_train.pt"), shuffle=True, seed=seed
    )
    print("\n--- Dev dataset ---")
    dev_ds = ShardedXLSRDataset(
        str(cache_dir / "xlsr_dev.pt"), shuffle=False, seed=seed
    )

    # num_workers=0 required for IterableDataset with shard files
    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size,
        num_workers=0, pin_memory=True, collate_fn=collate_fn,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    head = CompressionModule(input_dim=1024, hidden_dim=256, dropout_rate=0.0).to(device)

    if resume_ckpt and Path(resume_ckpt).is_file():
        ckpt = torch.load(resume_ckpt, map_location=device)
        head.load_state_dict(ckpt["compression_state_dict"])
        print(f"\nLoaded head weights from: {resume_ckpt}  "
              f"(epoch {ckpt.get('epoch','?')})")

    # ------------------------------------------------------------------
    # Loss + optimizer
    # ------------------------------------------------------------------
    loss_fn = SupConBinaryLoss(
        temperature=temperature,
        similarity=similarity,
        uniformity_weight=unif_weight,
        uniformity_t=unif_t,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=head_lr, weight_decay=weight_decay
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_dev_loss     = float("inf")
    epochs_no_improve = 0
    best_path = Path(os.environ.get("SAVE_PATH", str(save_dir / "stage1_cached_head_best.pt")))

    print(f"\nTraining for up to {epochs} epochs  (patience={patience})\n")

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)   # re-shuffles shard order each epoch
        t0 = time.time()

        train_loss = run_one_epoch(head, loss_fn, train_loader, optimizer, device, train=True)
        dev_loss   = run_one_epoch(head, loss_fn, dev_loader,   optimizer, device, train=False)

        elapsed = time.time() - t0
        print(f"[epoch {epoch:03d}]  train={train_loss:.4f}  dev={dev_loss:.4f}  "
              f"time={elapsed:.1f}s", end="")

        if dev_loss < best_dev_loss:
            best_dev_loss     = dev_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":                  epoch,
                    "compression_state_dict": head.state_dict(),
                    "train_loss":             train_loss,
                    "dev_loss":               dev_loss,
                    "config": {
                        "temperature":  temperature,
                        "similarity":   similarity,
                        "batch_size":   batch_size,
                        "hidden_dim":   256,
                    },
                },
                best_path,
            )
            print(f"  ✓ saved (dev={best_dev_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"  (no improve {epochs_no_improve}/{patience})")

        if patience and epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest dev loss : {best_dev_loss:.4f}")
    print(f"Checkpoint    : {best_path}")


if __name__ == "__main__":
    main()