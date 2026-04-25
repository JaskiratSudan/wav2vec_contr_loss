import os
import random

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


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
        return self.net(x).squeeze(-1)


def build_head(cfg, in_dim: int) -> nn.Module:
    if cfg.head_type == "linear":
        return LinearBinaryHead(in_dim=in_dim)
    if cfg.head_type == "mlp":
        return SmallMLPBinaryHead(in_dim=in_dim, hidden=cfg.hidden_dim, dropout=cfg.dropout)
    raise ValueError(f"Unknown HEAD_TYPE: {cfg.head_type}")


def compute_pos_weight(labels: np.ndarray) -> float:
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        return 1.0
    return float(neg) / float(pos)


def compute_metrics(y_true: torch.Tensor, logits: torch.Tensor):
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
                pass

    return acc, auc, eer


def train_classifier(cfg, train_loader, dev_loader, device, in_dim, pos_weight_val, ckpt_cfg):
    clf = build_head(cfg, in_dim=in_dim).to(device)
    os.makedirs(cfg.save_dir, exist_ok=True)

    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(
        clf.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_eer = float("inf")
    best_dev_loss = float("inf")
    epochs_no_improve = 0
    best_path = None

    use_eer_for_patience = HAVE_SKLEARN
    if not use_eer_for_patience:
        print("[INFO] Using dev loss for early stopping (EER unavailable).")

    for epoch in range(1, cfg.epochs + 1):
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

            if step % cfg.log_every == 0:
                print(f"[epoch {epoch:03d} | step {step:04d}] train_loss={loss.item():.4f}")

        avg_train_loss = total_train_loss / max(1, n_train)

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

        improved = False
        if use_eer_for_patience and (dev_eer is not None):
            if dev_eer < best_eer:
                best_eer = dev_eer
                improved = True
        elif not use_eer_for_patience:
            if avg_dev_loss < best_dev_loss:
                best_dev_loss = avg_dev_loss
                improved = True

        if improved:
            epochs_no_improve = 0
            best_path = os.path.join(cfg.save_dir, "stage2_binary_head_best.pt")
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
                    "config": ckpt_cfg,
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

        if epochs_no_improve >= cfg.patience:
            if use_eer_for_patience and dev_eer is not None:
                print(
                    f"[EARLY STOP] Patience reached ({cfg.patience}) with no EER improvement. "
                    f"Best EER = {best_eer * 100:.2f}%"
                )
            else:
                print(
                    f"[EARLY STOP] Patience reached ({cfg.patience}) with no dev_loss improvement. "
                    f"Best dev_loss = {best_dev_loss:.4f}"
                )
            break

    print("==> Stage-2 training complete.")
    if best_path is not None:
        print(f"Best classifier checkpoint: {best_path}")
    return best_path
