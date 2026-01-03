import argparse
import os
from types import SimpleNamespace

# =======================
#        DEFAULTS
# =======================
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

EMB_DIR = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/{MODEL_NAME}"
TRAIN_SPLIT = "train"
DEV_SPLIT = "dev"

BATCH_SIZE = 64
EPOCHS = 200
LR = 1e-4
WEIGHT_DECAY = 1e-4

HEAD_TYPE = "linear"
HIDDEN_DIM = 128
DROPOUT = 0.2

SEED = 1337
SAVE_DIR = f"checkpoints_stage2/supcon_geodesic_dist/{MODEL_NAME}"
LOG_EVERY = 10

PATIENCE = 15


def build_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=str, default=EMB_DIR, help="Directory with stage-1 embeddings.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Directory to save stage-2 checkpoints.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Max epochs (early stopping may stop earlier).")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY, help="Weight decay.")
    parser.add_argument(
        "--head_type",
        type=str,
        default=HEAD_TYPE,
        choices=["linear", "mlp"],
        help="Classifier head type.",
    )
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM, help="Hidden dimension for MLP head.")
    parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout for MLP head.")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience.")
    args = parser.parse_args()

    return SimpleNamespace(
        emb_dir=args.emb_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        head_type=args.head_type,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        model_name=MODEL_NAME,
        train_split=TRAIN_SPLIT,
        dev_split=DEV_SPLIT,
        seed=SEED,
        log_every=LOG_EVERY,
    )


def ckpt_config(cfg, in_dim, pos_weight_val):
    return {
        "EMB_DIR": cfg.emb_dir,
        "TRAIN_SPLIT": cfg.train_split,
        "DEV_SPLIT": cfg.dev_split,
        "HEAD_TYPE": cfg.head_type,
        "IN_DIM": in_dim,
        "HIDDEN_DIM": cfg.hidden_dim,
        "DROPOUT": cfg.dropout,
        "LR": cfg.lr,
        "WEIGHT_DECAY": cfg.weight_decay,
        "BATCH_SIZE": cfg.batch_size,
        "EPOCHS": cfg.epochs,
        "PATIENCE": cfg.patience,
        "pos_weight": pos_weight_val,
    }
