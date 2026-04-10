"""
train_stage2_cached.py

Trains a linear BCE classifier (Stage 2) on pre-cached 256-dim projection
head embeddings. Embedding extraction is handled upstream by
extract_head_embeddings.py — this script only does training and model selection.

Cache format expected (produced by extract_head_embeddings.py)
--------------------------------------------------------------
    {
        "embeddings":  FloatTensor [N, 256],
        "labels":      LongTensor  [N],          # 1=bonafide, 0=spoof
        "sources":     List[str]   length N,
        "audio_names": List[str]   length N,
    }

Usage
-----
    python train_stage2_cached.py \
        --emb_cache_train  /path/to/head_emb_train.pt \
        --emb_cache_dev    /path/to/head_emb_dev.pt \
        --save_path        /path/to/run_id_stage2_classifier_best.pt \
        --hidden_dim       256 \
        --batch_size       512 \
        --epochs           10 \
        --patience         5 \
        --lr               5e-4 \
        --weight_decay     3e-3 \
        --seed             42
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset — loads the entire cache into memory (only 256-dim vecs, ~20MB)
# ---------------------------------------------------------------------------

class EmbeddingDataset(Dataset):
    """
    Loads a single .pt file produced by extract_head_embeddings.py.
    Expected keys: embeddings [N, D], labels [N], sources [N], audio_names [N]
    """
    def __init__(self, cache_path: str):
        cache = torch.load(cache_path, map_location="cpu")
        self.embeddings  = cache["embeddings"].float()   # [N, 256]
        self.labels      = cache["labels"].float()       # [N]  BCE needs float
        self.sources     = cache["sources"]
        self.audio_names = cache.get("audio_names", [""] * len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Classifier head — single linear layer to match the paper
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)   # [B]


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()
    criterion  = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct    = 0
    n          = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels     = labels.to(device)          # (B,) float32

            logits = model(embeddings)              # (B,)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            preds   = (logits > 0.0).float()
            correct += (preds == labels).sum().item()
            n       += len(labels)

    return total_loss / max(len(loader), 1), correct / max(n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_cache_train", required=True,
                        help="Path to head_emb_train.pt")
    parser.add_argument("--emb_cache_dev",   required=True,
                        help="Path to head_emb_dev.pt")
    parser.add_argument("--save_path",       required=True,
                        help="Full path to write the best classifier checkpoint .pt")
    parser.add_argument("--hidden_dim",      type=int,   default=256)
    parser.add_argument("--batch_size",      type=int,   default=512)
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--patience",        type=int,   default=5)
    parser.add_argument("--lr",              type=float, default=5e-4)
    parser.add_argument("--weight_decay",    type=float, default=3e-3)
    parser.add_argument("--seed",            type=int,   default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Train cache : {args.emb_cache_train}")
    print(f"Dev cache   : {args.emb_cache_dev}")
    print(f"Save path   : {args.save_path}\n")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds = EmbeddingDataset(args.emb_cache_train)
    dev_ds   = EmbeddingDataset(args.emb_cache_dev)

    bon_tr = int((train_ds.labels == 1).sum())
    spo_tr = int((train_ds.labels == 0).sum())
    bon_dv = int((dev_ds.labels == 1).sum())
    spo_dv = int((dev_ds.labels == 0).sum())
    print(f"Train: bonafide={bon_tr}  spoof={spo_tr}  total={len(train_ds)}")
    print(f"Dev  : bonafide={bon_dv}  spoof={spo_dv}  total={len(dev_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------
    # Model + optimizer
    # ------------------------------------------------------------------
    model     = ClassifierHead(args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    save_path         = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    best_dev_loss     = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        dev_loss,   dev_acc   = run_epoch(model, dev_loader,   optimizer, device, train=False)
        elapsed = time.time() - t0

        print(
            f"[epoch {epoch:03d}]  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc*100:.1f}%  "
            f"dev_loss={dev_loss:.4f}  dev_acc={dev_acc*100:.1f}%  "
            f"time={elapsed:.1f}s",
            end="",
        )

        if dev_loss < best_dev_loss:
            best_dev_loss     = dev_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch":             epoch,
                    "classifier_state_dict": model.state_dict(),  # for evaluate_stage2.py
                    "model_state_dict":  model.state_dict(),      # for existing eval script
                    "dev_loss":          dev_loss,
                    "dev_acc":           dev_acc,
                    "config": {
                        "HEAD_TYPE":  "linear",
                        "IN_DIM":     args.hidden_dim,
                        "HIDDEN_DIM": 128,
                        "DROPOUT":    0.2,
                    },
                },
                save_path,
            )
            print(f"  ✓ saved (dev={best_dev_loss:.4f}  acc={dev_acc*100:.1f}%)")
        else:
            epochs_no_improve += 1
            print(f"  (no improve {epochs_no_improve}/{args.patience})")

        if args.patience and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest dev loss : {best_dev_loss:.4f}")
    print(f"Checkpoint    : {save_path}")


if __name__ == "__main__":
    main()