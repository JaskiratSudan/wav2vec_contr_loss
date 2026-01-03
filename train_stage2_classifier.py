# train_stage2_classifier.py
# --------------------------------------------------------
# Stage-2: Train a lightweight binary classifier on top of
# fixed Stage-1 embeddings saved by extract_stage1_embeddings.py.
# --------------------------------------------------------

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from stage2_config import build_config, ckpt_config
from stage2_utils import HAVE_SKLEARN, set_seed, compute_pos_weight, train_classifier


def main():
    cfg = build_config()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not HAVE_SKLEARN:
        print(
            "[WARN] sklearn not found. EER/AUC will be unavailable; "
            "early stopping will fall back to dev loss."
        )

    train_emb_path = os.path.join(cfg.emb_dir, f"{cfg.train_split}_embeddings.npy")
    train_lab_path = os.path.join(cfg.emb_dir, f"{cfg.train_split}_labels.npy")
    dev_emb_path = os.path.join(cfg.emb_dir, f"{cfg.dev_split}_embeddings.npy")
    dev_lab_path = os.path.join(cfg.emb_dir, f"{cfg.dev_split}_labels.npy")

    X_train = np.load(train_emb_path).astype("float32")
    y_train = np.load(train_lab_path).astype("float32")
    X_dev = np.load(dev_emb_path).astype("float32")
    y_dev = np.load(dev_lab_path).astype("float32")

    print(f"Train embeddings: {X_train.shape}, Dev embeddings: {X_dev.shape}")

    in_dim = X_train.shape[1]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dev_ds = TensorDataset(torch.from_numpy(X_dev), torch.from_numpy(y_dev))

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    pos_weight_val = compute_pos_weight(y_train)
    print(f"Class balance: pos_weight={pos_weight_val:.3f}")

    ckpt_cfg = ckpt_config(cfg, in_dim, pos_weight_val)
    train_classifier(cfg, train_loader, dev_loader, device, in_dim, pos_weight_val, ckpt_cfg)


if __name__ == "__main__":
    main()
