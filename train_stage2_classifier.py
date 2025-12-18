# train_stage2_classifier.py
# --------------------------------------------------------
# Stage-2: Train a lightweight binary classifier on top of
# fixed Stage-1 embeddings saved by extract_stage1_embeddings.py.
#
# Expects:
#   stage1_embeddings/train_embeddings.npy  (N_train, D)
#   stage1_embeddings/train_labels.npy      (N_train,)
#   stage1_embeddings/dev_embeddings.npy    (N_dev, D)
#   stage1_embeddings/dev_labels.npy        (N_dev,)
#
# Output:
#   checkpoints_stage2/stage2_binary_head_best.pt
#   Early stopping is based on dev EER (lower is better).
# --------------------------------------------------------

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Optional: for AUC / EER
try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


# ============================================================
#                       CONFIG
# ============================================================

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

EMB_DIR = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/{MODEL_NAME}"
TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"

BATCH_SIZE = 64
EPOCHS = 200          # upper bound; early stopping will cut this
LR = 1e-4
WEIGHT_DECAY = 1e-4

# "linear" -> single FC layer
# "mlp"    -> small MLP with one hidden layer
HEAD_TYPE = "mlp"
HIDDEN_DIM = 128
DROPOUT = 0.2

SEED = 1337
SAVE_DIR = f"checkpoints_stage2/with_rawboost/no_hbm/{MODEL_NAME}"
LOG_EVERY = 10

# Early stopping based on EER (lower is better)
PATIENCE = 15         # epochs without EER improvement before stopping

# ============================================================
#                       UTILS
# ============================================================

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LinearBinaryHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits, shape (B,)
        return self.fc(x).squeeze(-1)


class SmallMLPBinaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits, shape (B,)
        return self.net(x).squeeze(-1)


def compute_pos_weight(labels: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    labels: numpy array of 0/1 with 1 = bonafide (positive class).
    """
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        return 1.0
    return float(neg) / float(pos)


def compute_metrics(y_true: torch.Tensor, logits: torch.Tensor):
    """
    Compute accuracy (always), and AUC/EER if sklearn is available.
    y_true, logits: 1D CPU tensors.
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        correct = (preds == y_true).sum().item()
        acc = correct / y_true.numel()

        auc = None
        eer = None

        if HAVE_SKLEARN:
            y_np = y_true.cpu().numpy()
            score_np = probs.cpu().numpy()
            try:
                auc = roc_auc_score(y_np, score_np)
                fpr, tpr, _ = roc_curve(y_np, score_np)
                fnr = 1.0 - tpr
                idx = np.nanargmin(np.abs(fnr - fpr))
                eer = (fpr[idx] + fnr[idx]) / 2.0
            except Exception:
                # If something goes wrong, keep auc/eer as None
                pass

    return acc, auc, eer


# ============================================================
#                       MAIN
# ============================================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not HAVE_SKLEARN:
        print("[WARN] sklearn not found. EER/AUC will be unavailable; "
              "early stopping will fall back to dev loss.")

    # ----- Load embeddings -----
    train_emb_path = os.path.join(EMB_DIR, f"{TRAIN_SPLIT}_embeddings.npy")
    train_lab_path = os.path.join(EMB_DIR, f"{TRAIN_SPLIT}_labels.npy")
    dev_emb_path   = os.path.join(EMB_DIR, f"{DEV_SPLIT}_embeddings.npy")
    dev_lab_path   = os.path.join(EMB_DIR, f"{DEV_SPLIT}_labels.npy")

    X_train = np.load(train_emb_path).astype("float32")
    y_train = np.load(train_lab_path).astype("float32")
    X_dev   = np.load(dev_emb_path).astype("float32")
    y_dev   = np.load(dev_lab_path).astype("float32")

    print(f"Train embeddings: {X_train.shape}, Dev embeddings: {X_dev.shape}")

    in_dim = X_train.shape[1]

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    dev_ds = TensorDataset(
        torch.from_numpy(X_dev),
        torch.from_numpy(y_dev),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # ----- Build classifier head -----
    if HEAD_TYPE == "linear":
        clf = LinearBinaryHead(in_dim=in_dim)
    elif HEAD_TYPE == "mlp":
        clf = SmallMLPBinaryHead(in_dim=in_dim, hidden=HIDDEN_DIM, dropout=DROPOUT)
    else:
        raise ValueError(f"Unknown HEAD_TYPE: {HEAD_TYPE}")

    clf.to(device)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Class imbalance handling
    pos_weight_val = compute_pos_weight(y_train)
    print(f"Class balance: pos_weight={pos_weight_val:.3f}")
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        clf.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # ----- Early stopping state (based on EER if available) -----
    best_eer = float("inf")       # lower is better
    best_dev_loss = float("inf")  # fallback if EER unavailable
    epochs_no_improve = 0
    best_path = None

    use_eer_for_patience = HAVE_SKLEARN
    if not use_eer_for_patience:
        print("[INFO] Using dev loss for early stopping (EER unavailable).")

    # ----- Training loop -----
    for epoch in range(1, EPOCHS + 1):
        # --------------------- TRAIN ---------------------
        clf.train()
        total_train_loss = 0.0
        n_train = 0

        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device)
            yb = yb.to(device)

            logits = clf(xb)
            loss = criterion(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = xb.size(0)
            total_train_loss += loss.item() * batch_size
            n_train += batch_size

            if step % LOG_EVERY == 0:
                print(f"[epoch {epoch:03d} | step {step:04d}] train_loss={loss.item():.4f}")

        avg_train_loss = total_train_loss / max(1, n_train)

        # ---------------------- DEV ----------------------
        clf.eval()
        total_dev_loss = 0.0
        n_dev = 0
        all_dev_logits = []
        all_dev_labels = []

        with torch.no_grad():
            for xb, yb in dev_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = clf(xb)
                loss = criterion(logits, yb)

                batch_size = xb.size(0)
                total_dev_loss += loss.item() * batch_size
                n_dev += batch_size

                all_dev_logits.append(logits.cpu())
                all_dev_labels.append(yb.cpu())

        avg_dev_loss = total_dev_loss / max(1, n_dev)

        all_dev_logits = torch.cat(all_dev_logits, dim=0)
        all_dev_labels = torch.cat(all_dev_labels, dim=0)

        dev_acc, dev_auc, dev_eer = compute_metrics(all_dev_labels, all_dev_logits)

        # Log metrics
        msg = (
            f"[epoch {epoch:03d}] "
            f"train_loss={avg_train_loss:.4f} | dev_loss={avg_dev_loss:.4f} | "
            f"dev_acc={dev_acc * 100:.2f}%"
        )
        if dev_auc is not None:
            msg += f" | dev_auc={dev_auc:.4f}"
        else:
            msg += " | dev_auc=N/A"
        if dev_eer is not None:
            msg += f" | dev_eer={dev_eer * 100:.2f}%"
        else:
            msg += " | dev_eer=N/A"
        print(msg)

        # ----------------- EARLY STOPPING -----------------
        improved = False

        if use_eer_for_patience and (dev_eer is not None):
            # Monitor EER (lower is better)
            if dev_eer < best_eer:
                best_eer = dev_eer
                improved = True
        elif not use_eer_for_patience:
            # Fallback: monitor dev loss (lower is better)
            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                improved = True

        if improved:
            epochs_no_improve = 0
            # Save best checkpoint
            best_path = os.path.join(SAVE_DIR, "stage2_binary_head_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": clf.state_dict(),
                    "train_loss": avg_train_loss,
                    "dev_loss": avg_dev_loss,
                    "dev_acc": dev_acc,
                    "dev_auc": dev_auc,
                    "dev_eer": dev_eer,
                    "monitor": "EER" if use_eer_for_patience and dev_eer is not None else "dev_loss",
                    "best_eer": best_eer,
                    "best_dev_loss": best_dev_loss,
                    "config": {
                        "EMB_DIR": EMB_DIR,
                        "TRAIN_SPLIT": TRAIN_SPLIT,
                        "DEV_SPLIT": DEV_SPLIT,
                        "HEAD_TYPE": HEAD_TYPE,
                        "IN_DIM": in_dim,
                        "HIDDEN_DIM": HIDDEN_DIM,
                        "DROPOUT": DROPOUT,
                        "LR": LR,
                        "WEIGHT_DECAY": WEIGHT_DECAY,
                        "BATCH_SIZE": BATCH_SIZE,
                        "EPOCHS": EPOCHS,
                        "PATIENCE": PATIENCE,
                        "pos_weight": pos_weight_val,
                    },
                },
                best_path,
            )
            if use_eer_for_patience and dev_eer is not None:
                print(f"[epoch {epoch:03d}] ✓ New best EER={best_eer * 100:.2f}% -> {best_path}")
            elif not use_eer_for_patience:
                print(f"[epoch {epoch:03d}] ✓ New best dev_loss={best_dev_loss:.4f} -> {best_path}")
        else:
            epochs_no_improve += 1
            if use_eer_for_patience and dev_eer is not None:
                print(
                    f"[epoch {epoch:03d}] No EER improvement for {epochs_no_improve} "
                    f"epoch(s) (best={best_eer * 100:.2f}%)"
                )
            elif not use_eer_for_patience:
                print(
                    f"[epoch {epoch:03d}] No dev_loss improvement for {epochs_no_improve} "
                    f"epoch(s) (best={best_dev_loss:.4f})"
                )

        # Check patience
        if epochs_no_improve >= PATIENCE:
            if use_eer_for_patience and dev_eer is not None:
                print(
                    f"[EARLY STOP] Patience reached ({PATIENCE}) with no EER improvement. "
                    f"Best EER = {best_eer * 100:.2f}%"
                )
            else:
                print(
                    f"[EARLY STOP] Patience reached ({PATIENCE}) with no dev_loss improvement. "
                    f"Best dev_loss = {best_dev_loss:.4f}"
                )
            break

    print("==> Stage-2 training complete.")
    if best_path is not None:
        print(f"Best classifier checkpoint: {best_path}")


if __name__ == "__main__":
    main()
