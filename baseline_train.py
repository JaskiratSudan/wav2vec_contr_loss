# train_end2end_bce.py
import os, random
from typing import List
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

from data_loader import ASVspoof2019Dataset, pad_collate_fn_speaker_source_multiclass
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from loss import BCEBinaryLoss, compute_pos_weight_from_dataset

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

# End-to-end finetuning XLSR-300m with batch_size=256 will almost certainly OOM.
# Use smaller batch and/or grad accumulation.
EPOCHS = 100
train_batch_size = 128
dev_batch_size = 128
NUM_SAMPLES = None
NUM_WORKERS = 4
SEED = 1337

# LRs: encoder should be much smaller than head/classifier
enc_lr = 1e-5
head_lr = 5e-4
WEIGHT_DECAY = 1e-4

GRAD_CLIP = 5.0
USE_AMP = True

SAVE_DIR = "/home/jsudan/wav2vec_contr_loss/checkpoints_baseline/bce"
PATIENCE = 10  # EER patience

# ---- RawBoost ----
USE_RAWBOOST = True
RAWBOOST_PROB = 0.7


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
    if not USE_RAWBOOST: return x
    device = x.device
    pad_mask = (x != 0.0)
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
    return y * pad_mask.to(device=y.device, dtype=y.dtype)

def compute_eer_and_thresh(labels01, scores):
    labels01 = labels01.astype(np.int32)
    scores = scores.astype(np.float64)

    P = (labels01 == 1).sum()
    N = (labels01 == 0).sum()

    order = np.argsort(-scores)
    y = labels01[order]
    s = scores[order]

    tp = fp = 0
    best = 1e9
    best_eer = 1.0
    best_thresh = s[0]

    i = 0
    while i < len(s):
        thresh = s[i]
        while i < len(s) and s[i] == thresh:
            tp += int(y[i] == 1)
            fp += int(y[i] == 0)
            i += 1

        fn = P - tp
        fpr = fp / float(N)
        fnr = fn / float(P)

        d = abs(fpr - fnr)
        if d < best:
            best = d
            best_eer = (fpr + fnr) / 2.0
            best_thresh = thresh

    return float(best_eer), float(best_thresh)



# =======================
#        MODEL
# =======================
class End2EndBCEModel(nn.Module):
    """
    encoder: returns (B,K,F,T)
    compression: (B,K,F,T) -> (B,H,T)
    time-pool: (B,H,T) -> (B,H)
    classifier: (B,H) -> (B,) logits
    """
    def __init__(self, encoder: nn.Module, compression: nn.Module, hidden_dim: int, finetune_encoder: bool):
        super().__init__()
        self.encoder = encoder
        self.compression = compression
        self.classifier = nn.Linear(hidden_dim, 1)
        self.finetune_encoder = finetune_encoder

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor):
        if self.finetune_encoder:
            hs = self.encoder(waveforms, attention_mask=attention_mask)   # (B,K,F,T)
        else:
            with torch.no_grad():
                hs = self.encoder(waveforms, attention_mask=attention_mask)   # (B,K,F,T)
        seq = self.compression(hs)                                        # (B,H,T)
        emb = seq.mean(dim=-1)                                            # (B,H)
        logits = self.classifier(emb).squeeze(-1)                         # (B,)
        return logits, emb

# =======================
#      TRAIN / DEV
# =======================
def train_one_epoch(model, loss_fn, loader, optimizer, device, scaler):
    model.train()
    total, steps = 0.0, 0

    for waveforms, labels, *_ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device).float()

        if USE_RAWBOOST:
            waveforms = apply_rawboost_batch(waveforms)

        attn = (waveforms != 0.0).long()

        optimizer.zero_grad(set_to_none=True)

        use_amp = (scaler is not None)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits, _ = model(waveforms, attn)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(waveforms, attn)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        total += float(loss.item())
        steps += 1

    return total / max(1, steps)


@torch.no_grad()
def evaluate_dev(model, loss_fn, loader, device):
    model.eval()
    total, steps = 0.0, 0
    all_logits = []
    all_labels = []

    for waveforms, labels, *_ in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device).float()
        attn = (waveforms != 0.0).long()

        logits, _ = model(waveforms, attn)
        loss = loss_fn(logits, labels)

        total += float(loss.item())
        steps += 1

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    probs = torch.sigmoid(all_logits).numpy()
    y = all_labels.numpy().astype(np.int32)

    eer, thr = compute_eer_and_thresh(y, probs)
    acc05 = float(((probs > 0.5).astype(np.float32) == y).mean())
    acc_thr = float(((probs > thr).astype(np.float32) == y).mean())

    print(f"[DEV] acc@0.5={acc05*100:.2f}% | acc@eer_thr={acc_thr*100:.2f}% | eer_thr={thr:.4f}")

    return total / max(1, steps), eer, acc05

def main():
    MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument(
        "--finetune_encoder",
        type=int,
        default=0,
        choices=[0, 1],
        help="Enable encoder finetuning (1) or keep frozen (0).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=train_batch_size,
        help="Training batch size.",
    )
    parser.add_argument(
        "--dev_batch_size",
        type=int,
        default=dev_batch_size,
        help="Dev batch size.",
    )
    parser.add_argument(
        "--enc_lr",
        type=float,
        default=enc_lr,
        help="Encoder learning rate (used when finetuning).",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=head_lr,
        help="Head/classifier learning rate.",
    )
    args = parser.parse_args()
    MODEL_NAME = args.model_name
    RUN_TAG = MODEL_NAME.replace("/", "__")
    FINETUNE_ENCODER = bool(args.finetune_encoder)
    train_bs = args.train_batch_size
    dev_bs = args.dev_batch_size
    enc_lr_val = args.enc_lr
    head_lr_val = args.head_lr

    set_seed(SEED)

    save_dir = os.path.join(SAVE_DIR, RUN_TAG)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {MODEL_NAME}")
    print(f"Device: {device} | AMP={USE_AMP} | RawBoost={USE_RAWBOOST} (p={RAWBOOST_PROB})")
    print(f"Finetune encoder: {FINETUNE_ENCODER}")

    # ---- datasets ----
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

    # Train: balanced sampler (like your stage1)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=BalancedBatchSampler(train_ds, train_bs),
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass
    )

    # Dev: IMPORTANT for EER — evaluate on natural dev distribution (no balancing).
    dev_loader = DataLoader(
        dev_ds,
        batch_size=dev_bs,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass
    )

    # ---- encoder + compression ----
    # End-to-end finetuning if enabled
    encoder = Wav2Vec2Encoder(model_name=MODEL_NAME, freeze_encoder=not FINETUNE_ENCODER).to(device)
    head = CompressionModule(INPUT_DIM, HIDDEN_DIM, DROPOUT).to(device)

    model = End2EndBCEModel(
        encoder=encoder,
        compression=head,
        hidden_dim=HIDDEN_DIM,
        finetune_encoder=FINETUNE_ENCODER,
    ).to(device)

    # ---- BCE loss (with pos_weight) ----
    # If you keep balanced training, pos_weight matters less, but it’s still fine.
    pos_weight = compute_pos_weight_from_dataset(train_ds)
    print(f"pos_weight (neg/pos) = {pos_weight:.4f}")
    loss_fn = BCEBinaryLoss(pos_weight=pos_weight).to(device)

    # ---- optimizer with param groups (safer for finetuning) ----
    params = [
        {"params": model.compression.parameters(), "lr": head_lr_val},
        {"params": model.classifier.parameters(), "lr": head_lr_val},
    ]
    if FINETUNE_ENCODER:
        params.insert(0, {"params": model.encoder.parameters(), "lr": enc_lr_val})
    optimizer = torch.optim.AdamW(params, weight_decay=WEIGHT_DECAY)

    scaler = None
    if (USE_AMP and device.type == "cuda"):
        scaler = torch.cuda.amp.GradScaler()

    # ---- early stopping on dev EER ----
    best_eer = float("inf")
    best_path = None
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, loss_fn, train_loader, optimizer, device, scaler)
        dev_loss, dev_eer, dev_acc = evaluate_dev(model, loss_fn, dev_loader, device)

        print(
            f"[epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | dev_loss={dev_loss:.4f} | "
            f"dev_eer={dev_eer*100:.2f}% | dev_acc={dev_acc*100:.2f}%"
        )

        improved = dev_eer < best_eer
        if improved:
            best_eer = dev_eer
            no_improve = 0
            best_path = os.path.join(save_dir, f"{RUN_TAG}_baseline_bce_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_eer": best_eer,
                    "train_loss": train_loss,
                    "dev_loss": dev_loss,
                    "config": {
                        "MODEL_NAME": MODEL_NAME,
                        "INPUT_DIM": INPUT_DIM,
                        "HIDDEN_DIM": HIDDEN_DIM,
                        "DROPOUT": DROPOUT,
                        "enc_lr": enc_lr_val,
                        "head_lr": head_lr_val,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "train_batch_size": train_bs,
                        "dev_batch_size": dev_bs,
                        "USE_RAWBOOST": USE_RAWBOOST,
                        "RAWBOOST_PROB": RAWBOOST_PROB,
                        "PATIENCE": PATIENCE,
                        "FINETUNE_ENCODER": FINETUNE_ENCODER,
                    },
                },
                best_path,
            )
            print(f"✓ Saved best (EER={best_eer*100:.2f}%) -> {best_path}")
        else:
            no_improve += 1
            print(f"No EER improvement ({no_improve}/{PATIENCE}) | best={best_eer*100:.2f}%")

        if no_improve >= PATIENCE:
            print(f"[EARLY STOP] Patience reached. Best dev EER = {best_eer*100:.2f}%")
            break

    if best_path:
        print(f"Best checkpoint: {best_path} (best EER={best_eer*100:.2f}%)")


if __name__ == "__main__":
    main()
