# generate_eval_score_file.py
# --------------------------------------------------------
# Generate ASVspoof-style CM score files using *saved embeddings*:
#   - ASVspoof2019 EVAL split  -> scores/with_rawboost/score_cm_eval.txt
#   - In-The-Wild (ITW)        -> scores/with_rawboost/score_cm_itw.txt
#
# Each line:
#   <utt_id> <source> <key> <score>
# --------------------------------------------------------

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ============================================================
#                       CONFIG
# ============================================================

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

# Paths to saved embeddings & labels (edit these to match your setup)
EVAL_EMB_PATH   = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/{MODEL_NAME}/eval_embeddings.npy"
EVAL_LABEL_PATH = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/{MODEL_NAME}/eval_labels.npy"

ITW_EMB_PATH    = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/{MODEL_NAME}/itw_embeddings.npy"
ITW_LABEL_PATH  = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/{MODEL_NAME}/itw_labels.npy"

# EVAL_EMB_PATH   = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/supcon_uniformity/eval_embeddings.npy"
# EVAL_LABEL_PATH = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ASV/supcon_uniformity/eval_labels.npy"

# ITW_EMB_PATH    = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/supcon_uniformity/itw_embeddings.npy"
# ITW_LABEL_PATH  = f"/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings/stage1_embeddings/ITW/supcon_uniformity/itw_labels.npy"

# Stage-2 checkpoint (trained on those embeddings)
STAGE2_CKPT = f"checkpoints_stage2/supcon_geodesic_dist/{MODEL_NAME}/stage2_binary_head_best.pt"
# STAGE2_CKPT = "checkpoints_stage2/supcon_uniformity/facebook/wav2vec2-xls-r-300m/stage2_binary_head_best.pt"

BATCH_SIZE = 256
SEED = 1337

# ----- Score files -----
SCORE_FILE_ASV_EVAL = f"scores/supcon_geodesic_dist/{MODEL_NAME}/score_cm_eval.txt"
SCORE_FILE_ITW      = f"scores/supcon_geodesic_dist/{MODEL_NAME}/score_cm_itw.txt"

# SCORE_FILE_ASV_EVAL = f"/home/jsudan/wav2vec_contr_loss/scores/supcon_uniformity/facebook/wav2vec2-xls-r-300m/score_cm_eval.txt"
# SCORE_FILE_ITW      = f"/home/jsudan/wav2vec_contr_loss/scores/supcon_uniformity/facebook/wav2vec2-xls-r-300m/score_cm_itw.txt"

# Toggle which datasets to score
RUN_ASV_EVAL = True
RUN_ITW      = True

def safe_load(ckpt_path, map_location=None):
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(ckpt_path, map_location=map_location)

# ============================================================
#                       UTILS
# ============================================================

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Stage-2 heads (same as in train_stage2_classifier.py)
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


def load_stage2_head(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = safe_load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    head_type  = cfg.get("HEAD_TYPE", "linear")
    in_dim     = cfg.get("IN_DIM", 256)
    hidden_dim = cfg.get("HIDDEN_DIM", 128)
    dropout    = cfg.get("DROPOUT", 0.2)

    if head_type == "linear":
        clf = LinearBinaryHead(in_dim=in_dim).to(device)
    elif head_type == "mlp":
        clf = SmallMLPBinaryHead(in_dim=in_dim, hidden=hidden_dim, dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown HEAD_TYPE in Stage-2 ckpt: {head_type}")

    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    print(f"Loaded Stage-2 head: type={head_type}, in_dim={in_dim}, "
          f"hidden_dim={hidden_dim}, dropout={dropout}")
    return clf

# ============================================================
#                SCORE FILE GENERATION USING EMBS
# ============================================================

def write_cm_scores_from_embeddings(
    emb_path: str,
    label_path: str,
    clf: nn.Module,
    device: torch.device,
    score_path: str,
    utt_prefix: str,
):
    """
    emb_path : .npy, shape (N, D)
    label_path : .npy, shape (N,) with 0/1 labels (0=spoof, 1=bonafide)
    """
    print(f"Loading embeddings from: {emb_path}")
    embs   = np.load(emb_path)      # (N, D)
    labels = np.load(label_path)    # (N,)

    assert embs.shape[0] == labels.shape[0], "Embeddings and labels size mismatch"

    N = embs.shape[0]
    print(f"Total samples: {N}, emb_dim={embs.shape[1]}")
    print(f"Writing scores to: {score_path}")

    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    with open(score_path, "w") as f:
        for start in tqdm(range(0, N, BATCH_SIZE), desc=f"Scoring {utt_prefix}"):
            end = min(start + BATCH_SIZE, N)
            batch_embs   = torch.from_numpy(embs[start:end]).to(device=device, dtype=torch.float32)
            batch_labels = labels[start:end]

            with torch.no_grad():
                logits = clf(batch_embs)                # (B,)
                scores = logits.cpu().numpy()           # higher = more bonafide-like

            for i in range(end - start):
                idx   = start + i
                utt_id = f"{utt_prefix}_{idx:06d}"
                source = "NA"
                key    = "bonafide" if int(batch_labels[i]) == 1 else "spoof"
                score  = scores[i]
                f.write(f"{utt_id} {source} {key} {score:.6f}\n")

    print(f"Done writing scores: {score_path}")

# ============================================================
#                       MAIN
# ============================================================

def main():
    global STAGE2_CKPT, EVAL_EMB_PATH, EVAL_LABEL_PATH, ITW_EMB_PATH, ITW_LABEL_PATH
    global SCORE_FILE_ASV_EVAL, SCORE_FILE_ITW
    global BATCH_SIZE, RUN_ASV_EVAL, RUN_ITW, SEED
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage2_ckpt",
        type=str,
        default=STAGE2_CKPT,
        help="Path to stage-2 checkpoint."
    )
    parser.add_argument(
        "--eval_emb_path",
        type=str,
        default=EVAL_EMB_PATH,
        help="Path to ASV eval embeddings (.npy)."
    )
    parser.add_argument(
        "--eval_label_path",
        type=str,
        default=EVAL_LABEL_PATH,
        help="Path to ASV eval labels (.npy)."
    )
    parser.add_argument(
        "--itw_emb_path",
        type=str,
        default=ITW_EMB_PATH,
        help="Path to ITW embeddings (.npy)."
    )
    parser.add_argument(
        "--itw_label_path",
        type=str,
        default=ITW_LABEL_PATH,
        help="Path to ITW labels (.npy)."
    )
    parser.add_argument(
        "--score_file_asv_eval",
        type=str,
        default=SCORE_FILE_ASV_EVAL,
        help="Output score file for ASV eval."
    )
    parser.add_argument(
        "--score_file_itw",
        type=str,
        default=SCORE_FILE_ITW,
        help="Output score file for ITW."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size for scoring."
    )
    parser.add_argument(
        "--run_asv_eval",
        type=int,
        default=int(RUN_ASV_EVAL),
        choices=[0, 1],
        help="Score ASV eval set (1) or skip (0)."
    )
    parser.add_argument(
        "--run_itw",
        type=int,
        default=int(RUN_ITW),
        choices=[0, 1],
        help="Score ITW set (1) or skip (0)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed."
    )
    args = parser.parse_args()

    STAGE2_CKPT = args.stage2_ckpt
    EVAL_EMB_PATH = args.eval_emb_path
    EVAL_LABEL_PATH = args.eval_label_path
    ITW_EMB_PATH = args.itw_emb_path
    ITW_LABEL_PATH = args.itw_label_path
    SCORE_FILE_ASV_EVAL = args.score_file_asv_eval
    SCORE_FILE_ITW = args.score_file_itw
    BATCH_SIZE = args.batch_size
    RUN_ASV_EVAL = bool(args.run_asv_eval)
    RUN_ITW = bool(args.run_itw)
    SEED = args.seed

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading Stage-2 checkpoint from: {STAGE2_CKPT}")
    clf = load_stage2_head(STAGE2_CKPT, device=device)

    # ---- ASVspoof2019 eval ----
    if RUN_ASV_EVAL:
        if os.path.exists(SCORE_FILE_ASV_EVAL):
            print(f"[SKIP] Found existing ASV eval score file: {SCORE_FILE_ASV_EVAL}")
        else:
            write_cm_scores_from_embeddings(
                emb_path=EVAL_EMB_PATH,
                label_path=EVAL_LABEL_PATH,
                clf=clf,
                device=device,
                score_path=SCORE_FILE_ASV_EVAL,
                utt_prefix="asv_eval",
            )

    # ---- ITW ----
    if RUN_ITW:
        if os.path.exists(SCORE_FILE_ITW):
            print(f"[SKIP] Found existing ITW score file: {SCORE_FILE_ITW}")
        else:
            write_cm_scores_from_embeddings(
                emb_path=ITW_EMB_PATH,
                label_path=ITW_LABEL_PATH,
                clf=clf,
                device=device,
                score_path=SCORE_FILE_ITW,
                utt_prefix="itw",
            )

    print("All requested score files handled (generated or skipped).")


if __name__ == "__main__":
    main()
