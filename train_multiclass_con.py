# train_stage1.py
# Stage-1 training: Frozen Wav2Vec2-Large -> CompressionModule -> clip embeddings
# Loss: Supervised Contrastive (binary), with warmup then hard-mining.
#
# Uses:
#   - encoder.py (Wav2Vec2Encoder)
#   - compression_module.py (CompressionModule you provided)
#   - loss.py (SupConBinaryLoss)
#   - data_loader.py (ASVspoof2019Dataset, pad_collate_fn_speaker_source)

import os
import random
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from loss import SupConBinaryLoss, SupConMultiClassLoss

from data_loader import (
    ASVspoof2019Dataset,
    pad_collate_fn_speaker_source,
    pad_collate_fn_speaker_source_multiclass,
)

# ============================================================
#                       CONFIG
# ============================================================

# ----- Train paths -----
TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"

# ----- Dev paths (edit to your dev set) -----
DEV_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac"
DEV_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt"

# Audio
MAX_DURATION_SECONDS = 5       # pad/trim to 5s at 16kHz
TARGET_SAMPLE_RATE = 16000

# Model
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
# TAKE_LAST_K_LAYERS = 8         # how many top layers to stack
INPUT_DIM = 1024               # wav2vec2-large hidden size
HIDDEN_DIM = 256               # final embedding dim from CompressionModule
DROPOUT = 0.1

# Training
EPOCHS = 50
WARMUP_EPOCHS = 6              # epochs with full SupCon before hard-mining
BATCH_SIZE = 256                # must be even (balanced batches)
LR = 1e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.1              # SupCon temperature
HARD_TOPK_NEG = 5              # top-k hardest negatives per anchor (after warmup)
USE_HARD_POS = True            # use hardest positive in hard-mining mode
SEED = 1337
NUM_WORKERS = 4
SAVE_DIR = "checkpoints_stage1"

LOG_EVERY = 50                 # steps

# ============================================================
#                    UTILS / SAMPLER
# ============================================================

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class BalancedBatchSampler(Sampler[List[int]]):
    """
    Yields balanced batches: half bonafide, half spoof.
    Assumes dataset.data is a list where item[1] is label in {0,1}.
    """
    def __init__(self, dataset, batch_size: int):
        assert batch_size % 2 == 0, "batch_size must be even for balanced sampling."
        self.dataset = dataset
        self.batch_size = batch_size

        # dataset.data: (path, label, speaker_id, audio_name)
        self.real_idx = [i for i, item in enumerate(dataset.data) if item[1] == 1]
        self.fake_idx = [i for i, item in enumerate(dataset.data) if item[1] == 0]

        if len(self.real_idx) == 0 or len(self.fake_idx) == 0:
            raise RuntimeError("Dataset must contain both bonafide and spoof samples.")

        self.per_class = batch_size // 2
        # Limit by the smaller class
        self.num_batches = min(
            len(self.real_idx) // self.per_class,
            len(self.fake_idx) // self.per_class,
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        random.shuffle(self.real_idx)
        random.shuffle(self.fake_idx)

        real_trunc = self.real_idx[: self.num_batches * self.per_class]
        fake_trunc = self.fake_idx[: self.num_batches * self.per_class]

        for b in range(self.num_batches):
            r_start = b * self.per_class
            f_start = b * self.per_class

            batch_indices = (
                real_trunc[r_start:r_start + self.per_class]
                + fake_trunc[f_start:f_start + self.per_class]
            )
            random.shuffle(batch_indices)
            yield batch_indices

# ============================================================
#                    TRAINING LOOP
# ============================================================

def train_one_epoch(
    encoder,
    head,
    loss_fn,
    loader,
    optimizer,
    device,
    epoch,
    warmup_epochs,
    hard_topk_neg=5,
    use_hard_pos=True,
    log_every=50,
):
    encoder.eval()      # frozen
    head.train()

    running_loss = 0.0
    step = 0

    hard_mining = (epoch >= warmup_epochs)

    for batch in loader:
        # batch: (waveforms, bin_labels, multi_labels, speakers, sources)
        waveforms, _, multi_labels, _, _ = batch

        waveforms = waveforms.to(device)
        multi_labels = multi_labels.to(device).long()

        attn_mask = (waveforms != 0.0).long()

        with torch.no_grad():
            hs_4d = encoder(waveforms, attention_mask=attn_mask)

        seq = head(hs_4d)
        z = seq.mean(dim=-1)
        z = F.normalize(z, p=2, dim=1)

        loss = loss_fn(z, multi_labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % log_every == 0:
            avg = running_loss / step
            print(
                f"[epoch {epoch:03d} | step {step:05d}] "
                f"loss={loss.item():.4f} (avg={avg:.4f}) | "
                f"mode={'HARD' if hard_mining else 'WARMUP'}"
            )

    return running_loss / max(1, step)

# ============================================================
#                  DEV EVAL (VALIDATION)
# ============================================================

def evaluate_on_dev(
    encoder,
    head,
    loss_fn,
    loader,
    device,
):
    encoder.eval()
    head.eval()
    total_loss, steps = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            waveforms, _, multi_labels, _, _ = batch
            waveforms = waveforms.to(device)
            multi_labels = multi_labels.to(device).long()

            attn_mask = (waveforms != 0.0).long()
            hs_4d = encoder(waveforms, attention_mask=attn_mask)
            seq = head(hs_4d)

            z = seq.mean(dim=-1)
            z = F.normalize(z, p=2, dim=1)

            loss = loss_fn(z, multi_labels)
            total_loss += loss.item()
            steps += 1

    return total_loss / max(1, steps)

# ============================================================
#                        MAIN
# ============================================================

def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------- Train Dataset & Loader --------
    train_ds = ASVspoof2019Dataset(
        root_dir=TRAIN_ROOT,
        protocol_file=TRAIN_PROTOCOL,
        subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )
    train_sampler = BalancedBatchSampler(train_ds, batch_size=BATCH_SIZE)

    train_loader = DataLoader(
    train_ds,
    batch_sampler=train_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=pad_collate_fn_speaker_source_multiclass,
)
    # -------- Dev Dataset & Loader --------
    dev_ds = ASVspoof2019Dataset(
        root_dir=DEV_ROOT,
        protocol_file=DEV_PROTOCOL,
        subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )

    # Use balanced batches for a clean SupCon dev signal
    dev_sampler = BalancedBatchSampler(dev_ds, batch_size=BATCH_SIZE)

    dev_loader = DataLoader(
    dev_ds,
    batch_sampler=dev_sampler,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    collate_fn=pad_collate_fn_speaker_source_multiclass,
)

    # -------- Models --------
    encoder = Wav2Vec2Encoder(
        model_name=MODEL_NAME,
        # take_last_k_layers=TAKE_LAST_K_LAYERS,
        freeze_encoder=True,
    ).to(device)

    head = CompressionModule(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT,
    ).to(device)

    # -------- Loss & Optimizer --------
    loss_fn = SupConMultiClassLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    print("==> Starting Stage-1 training")
    print(
        f"Encoder frozen: True | "
        f"SupCon-Binary (τ={TEMPERATURE}) | "
        f"Warmup epochs: {WARMUP_EPOCHS} | "
        f"Hard mining after warmup"
    )

    best_dev_loss = float("inf")
    best_path = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            encoder=encoder,
            head=head,
            loss_fn=loss_fn,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            warmup_epochs=WARMUP_EPOCHS,
            hard_topk_neg=HARD_TOPK_NEG,
            use_hard_pos=USE_HARD_POS,
            log_every=LOG_EVERY,
        )

        dev_loss = evaluate_on_dev(
            encoder=encoder,
            head=head,
            loss_fn=loss_fn,
            loader=dev_loader,
            device=device,
        )

        print(f"[epoch {epoch:03d}] train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}")

        # Save checkpoint (head only; encoder is fixed)
        ckpt = {
            "epoch": epoch,
            "compression_state_dict": head.state_dict(),
            "train_loss": train_loss,
            "dev_loss": dev_loss,
            "config": {
                "MODEL_NAME": MODEL_NAME,
                # "TAKE_LAST_K_LAYERS": TAKE_LAST_K_LAYERS,
                "INPUT_DIM": INPUT_DIM,
                "HIDDEN_DIM": HIDDEN_DIM,
                "DROPOUT": DROPOUT,
                "EPOCHS": EPOCHS,
                "WARMUP_EPOCHS": WARMUP_EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "LR": LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "TEMPERATURE": TEMPERATURE,
                "HARD_TOPK_NEG": HARD_TOPK_NEG,
            },
        }

        # path_epoch = os.path.join(SAVE_DIR, f"stage1_head_epoch{epoch:03d}.pt")
        # torch.save(ckpt, path_epoch)

        # Track best on dev
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            best_path = os.path.join(SAVE_DIR, "stage1_head_best.pt")
            torch.save(ckpt, best_path)
            print(f"[epoch {epoch:03d}] ✓ New best dev_loss={best_dev_loss:.4f} -> {best_path}")

    print("==> Training complete.")
    if best_path is not None:
        print(f"Best head (by dev_loss): {best_path} (dev_loss={best_dev_loss:.4f})")

if __name__ == "__main__":
    main()
