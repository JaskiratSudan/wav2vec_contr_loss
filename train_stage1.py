# train_stage1.py
import os, random
from typing import List
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from data_loader import ASVspoof2019Dataset, pad_collate_fn_speaker_source, pad_collate_fn_speaker_source_multiclass
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from loss import SupConBinaryLoss

# ---- RawBoost (train-time only) ----
from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise

# =======================
#        CONFIG
# =======================
TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"
DEV_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac"
DEV_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt"

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 5

INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

EPOCHS = 100
BATCH_SIZE = 256
NUM_SAMPLES = None      # e.g. 5000 to subsample; None for all
LR = 1e-3
WEIGHT_DECAY = 1e-3
TEMPERATURE = 0.2
NUM_WORKERS = 4
SEED = 1337
SAVE_DIR = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost"

# ---- Hard-neg difficulty knobs ----
TOPK_NEG = 8            # hardest negatives per anchor
WARMUP_EPOCHS = 8        # pure full SupCon for stability
ALPHA_END = 1          # final alpha weight on mined loss
ALPHA_RAMP_EPOCHS = 80   # epochs to go 0 -> ALPHA_END after warmup

# ---- RawBoost ----
USE_RAWBOOST = True
RAWBOOST_PROB = 0.9

# =======================
#      HELPERS
# =======================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int):
        assert batch_size % 2 == 0
        self.batch_size = batch_size
        self.data = dataset.data
        self.real = [i for i, it in enumerate(self.data) if it[1] == 1]
        self.fake = [i for i, it in enumerate(self.data) if it[1] == 0]
        self.per_class = batch_size // 2
        self.num_batches = min(len(self.real)//self.per_class, len(self.fake)//self.per_class)
    def __len__(self): return self.num_batches
    def __iter__(self):
        random.shuffle(self.real); random.shuffle(self.fake)
        r = self.real[: self.num_batches*self.per_class]
        f = self.fake[: self.num_batches*self.per_class]
        for b in range(self.num_batches):
            idx = r[b*self.per_class:(b+1)*self.per_class] + f[b*self.per_class:(b+1)*self.per_class]
            random.shuffle(idx); yield idx

def apply_rawboost_batch(x: torch.Tensor) -> torch.Tensor:
    """
    RawBoost augmentation per utterance (train only). Keeps padding at 0.
    """
    if not USE_RAWBOOST: return x
    device = x.device
    pad_mask = (x != 0.0)                 # True where real samples exist
    a = x.detach().cpu().numpy()
    for i in range(a.shape[0]):
        if random.random() < RAWBOOST_PROB:
            y = LnL_convolutive_noise(
                a[i], N_f=5, nBands=5,
                minF=20.0,  maxF=8000.0,
                minBW=100.0, maxBW=1000.0,
                minCoeff=10, maxCoeff=100,
                minG=0.0, maxG=0.0,
                minBiasLinNonLin=5.0, maxBiasLinNonLin=20.0,
                fs=TARGET_SAMPLE_RATE
            )
            if random.random() < 0.5:
                y = SSI_additive_noise(
                    y, SNRmin=10.0, SNRmax=40.0, nBands=5,
                    minF=20.0, maxF=8000.0, minBW=100.0, maxBW=1000.0,
                    minCoeff=10, maxCoeff=100, minG=0.0, maxG=0.0,
                    fs=TARGET_SAMPLE_RATE
                )
            if random.random() < 0.5:
                y = ISD_additive_noise(y, P=10.0, g_sd=2.0)
            a[i] = y
    y = torch.from_numpy(a).to(device=device, dtype=x.dtype)
    return y * pad_mask.to(device=y.device, dtype=y.dtype)  # zero-out padding again

def alpha_for_epoch(epoch: int) -> float:
    if epoch <= WARMUP_EPOCHS:
        return 0.0
    t = min(1.0, (epoch - WARMUP_EPOCHS) / max(1, ALPHA_RAMP_EPOCHS))
    return t * ALPHA_END

# =======================
#       TRAIN / DEV
# =======================
def train_one_epoch(encoder, head, loss_fn, loader, optimizer, device, epoch):
    encoder.eval(); head.train()
    total, steps = 0.0, 0
    alpha = alpha_for_epoch(epoch)
    for waveforms, labels, *_ in loader:
        waveforms = waveforms.to(device); labels = labels.to(device).long()
        if USE_RAWBOOST:
            waveforms = apply_rawboost_batch(waveforms)
        attn = (waveforms != 0.0).long()

        with torch.no_grad():
            hs = encoder(waveforms, attention_mask=attn)   # (B,K,F,T)
        seq = head(hs)                                     # (B,H,T)
        z = F.normalize(seq.mean(dim=-1), p=2, dim=1)      # (B,H)

        loss = loss_fn(z, labels, topk_neg=TOPK_NEG, alpha=alpha)

        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
        optimizer.step()

        total += loss.item(); steps += 1
    return total / max(1, steps), alpha

@torch.no_grad()
def evaluate(encoder, head, loss_fn, loader, device):
    encoder.eval(); head.eval()
    total, steps = 0.0, 0
    for waveforms, labels, *_ in loader:
        waveforms = waveforms.to(device); labels = labels.to(device).long()
        attn = (waveforms != 0.0).long()
        hs = encoder(waveforms, attention_mask=attn)
        seq = head(hs)
        z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        # dev uses alpha=0 for a stable metric
        loss = loss_fn(z, labels, topk_neg=TOPK_NEG, alpha=0.0)
        total += loss.item(); steps += 1
    return total / max(1, steps)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/wav2vec2-large-960h",
        help="HF ID for Wav2Vec2, e.g. facebook/wav2vec2-large-960h or microsoft/wavlm-base-plus"
    )
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    RUN_TAG = MODEL_NAME.replace("/", "__")

    set_seed(SEED)
    # Put checkpoints under a subfolder named after the model
    global SAVE_DIR
    SAVE_DIR = os.path.join(SAVE_DIR, RUN_TAG)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Model: {MODEL_NAME}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | RawBoost={USE_RAWBOOST} (p={RAWBOOST_PROB})")

    train_ds = ASVspoof2019Dataset(
        root_dir=TRAIN_ROOT, protocol_file=TRAIN_PROTOCOL, subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS, target_sample_rate=TARGET_SAMPLE_RATE,
        num_samples=NUM_SAMPLES
    )
    dev_ds = ASVspoof2019Dataset(
        root_dir=DEV_ROOT, protocol_file=DEV_PROTOCOL, subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS, target_sample_rate=TARGET_SAMPLE_RATE,
        num_samples=NUM_SAMPLES
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=BalancedBatchSampler(train_ds, BATCH_SIZE),
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=pad_collate_fn_speaker_source_multiclass
    )
    dev_loader = DataLoader(
        dev_ds, batch_sampler=BalancedBatchSampler(dev_ds, BATCH_SIZE),
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=pad_collate_fn_speaker_source_multiclass
    )

    encoder = Wav2Vec2Encoder(model_name=MODEL_NAME, freeze_encoder=True).to(device)
    head = CompressionModule(INPUT_DIM, HIDDEN_DIM, DROPOUT).to(device)

    loss_fn = SupConBinaryLoss(temperature=TEMPERATURE)
    optim = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best, best_path = float("inf"), None
    for epoch in range(1, EPOCHS + 1):
        train_loss, alpha = train_one_epoch(encoder, head, loss_fn, train_loader, optim, device, epoch)
        dev_loss = evaluate(encoder, head, loss_fn, dev_loader, device)
        print(f"[epoch {epoch:03d}] alpha={alpha:.2f} | train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f}")

        if dev_loss < best:
            best = dev_loss
            best_path = os.path.join(SAVE_DIR, f"{RUN_TAG}_stage1_head_best.pt")
            torch.save({
                "epoch": epoch,
                "compression_state_dict": head.state_dict(),
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "config": {
                    "MODEL_NAME": MODEL_NAME,
                    "RUN_TAG": RUN_TAG,
                    "INPUT_DIM": INPUT_DIM, "HIDDEN_DIM": HIDDEN_DIM, "DROPOUT": DROPOUT,
                    "BATCH_SIZE": BATCH_SIZE, "LR": LR, "WEIGHT_DECAY": WEIGHT_DECAY,
                    "TEMPERATURE": TEMPERATURE, "TOPK_NEG": TOPK_NEG,
                    "WARMUP_EPOCHS": WARMUP_EPOCHS, "ALPHA_END": ALPHA_END, "ALPHA_RAMP_EPOCHS": ALPHA_RAMP_EPOCHS,
                    "USE_RAWBOOST": USE_RAWBOOST, "RAWBOOST_PROB": RAWBOOST_PROB
                },
            }, best_path)
            print(f"✓ Saved best -> {best_path} (dev={best:.4f})")

    if best_path:
        print(f"Best checkpoint: {best_path} (dev={best:.4f})")

if __name__ == "__main__":
    main()
