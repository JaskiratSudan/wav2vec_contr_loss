"""
evaluate_stage2.py

Loads the best Stage 2 classifier checkpoint and the cached projection-head
embeddings for the eval split, then reports:
  - Overall EER
  - EER broken down per generator (Source)
  - Bonafide score distribution summary

Results are printed to stdout AND written to --out_file.

Usage
-----
    python evaluate_stage2.py \
        --emb_cache_eval  /path/to/head_emb_eval.pt \
        --stage2_ckpt     /path/to/run_id_stage2_classifier_best.pt \
        --hidden_dim      256 \
        --out_file        /path/to/eval_results.txt \
        --speaker         Barack_Obama

Cache format expected (produced by extract_head_embeddings.py)
--------------------------------------------------------------
    {
        "embeddings": FloatTensor [N, 256],   # l2-normalized
        "labels":     LongTensor  [N],        # 1=bonafide, 0=spoof
        "sources":    List[str]   length N,   # generator name or '-' for real
        "audio_names": List[str]  length N,   # filename, for debugging
    }
"""

import argparse
import sys
from pathlib import Path
from io import StringIO

import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# EER computation
# ---------------------------------------------------------------------------

def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """
    Compute EER given binary labels (1=bonafide, 0=spoof) and scores
    (higher = more bonafide-like, i.e. classifier logit).

    Returns (eer_percent, threshold).
    """
    # Sort by descending score
    sorted_idx = np.argsort(scores)[::-1]
    labels_sorted = labels[sorted_idx]

    n_pos = labels.sum()          # bonafide
    n_neg = len(labels) - n_pos   # spoof

    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")

    # Cumulative FA and FR at each threshold
    fa = np.cumsum(1 - labels_sorted) / n_neg   # false acceptance rate
    fr = 1 - np.cumsum(labels_sorted) / n_pos   # false rejection rate

    # Find crossover point
    diff = fa - fr
    idx  = np.argmin(np.abs(diff))
    eer  = (fa[idx] + fr[idx]) / 2.0 * 100.0
    threshold = scores[sorted_idx[idx]]
    return float(eer), float(threshold)


# ---------------------------------------------------------------------------
# Minimal classifier head (must match train_stage2_cached.py definition)
# ---------------------------------------------------------------------------

class ClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _line(width=62):
    return "=" * width

def _div(width=62):
    return "-" * width


def format_results(
    speaker: str,
    overall_eer: float,
    overall_thresh: float,
    per_source: dict,
    score_stats: dict,
) -> str:
    buf = StringIO()

    def p(*args, **kwargs):
        print(*args, **kwargs, file=buf)

    p(_line())
    p(f"  Evaluation Results — {speaker}")
    p(_line())

    p(f"\n  Overall EER : {overall_eer:.2f}%  (threshold={overall_thresh:.4f})")
    p(f"  Total samples: {score_stats['n_total']}  "
      f"(bonafide={score_stats['n_bon']}  spoof={score_stats['n_spo']})")

    p(f"\n  Score distribution:")
    p(f"    Bonafide  mean={score_stats['bon_mean']:.4f}  "
      f"std={score_stats['bon_std']:.4f}  "
      f"min={score_stats['bon_min']:.4f}  "
      f"max={score_stats['bon_max']:.4f}")
    p(f"    Spoof     mean={score_stats['spo_mean']:.4f}  "
      f"std={score_stats['spo_std']:.4f}  "
      f"min={score_stats['spo_min']:.4f}  "
      f"max={score_stats['spo_max']:.4f}")

    p(f"\n  Per-generator EER:")
    p(f"  {_div(58)}")
    p(f"  {'Source':<30} {'n_bon':>6} {'n_spo':>6} {'EER (%)':>9}")
    p(f"  {_div(58)}")

    # Sort: bonafide row first ('-'), then generators by EER descending
    bon_rows  = [(s, v) for s, v in per_source.items() if v["n_bon"] > 0 and v["n_spo"] == 0]
    spo_rows  = [(s, v) for s, v in per_source.items() if v["n_spo"] > 0]
    spo_rows.sort(key=lambda x: x[1]["eer"] if not np.isnan(x[1]["eer"]) else -1, reverse=True)

    for src, v in bon_rows + spo_rows:
        eer_str = f"{v['eer']:.2f}" if not np.isnan(v["eer"]) else "N/A"
        p(f"  {src:<30} {v['n_bon']:>6} {v['n_spo']:>6} {eer_str:>9}")

    p(f"  {_div(58)}")
    p(f"  {'OVERALL':<30} {score_stats['n_bon']:>6} {score_stats['n_spo']:>6} {overall_eer:>8.2f}%")
    p(_line())

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_cache_eval", required=True,
                        help="Path to head_emb_eval.pt (produced by extract_head_embeddings.py)")
    parser.add_argument("--stage2_ckpt",    required=True,
                        help="Path to the best Stage 2 classifier .pt checkpoint")
    parser.add_argument("--hidden_dim",     type=int, default=256)
    parser.add_argument("--out_file",       required=True,
                        help="Path to write the text results file")
    parser.add_argument("--speaker",        default="",
                        help="Speaker name for display only")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Load cached embeddings
    # ------------------------------------------------------------------
    print(f"Loading eval embeddings: {args.emb_cache_eval}")
    cache = torch.load(args.emb_cache_eval, map_location="cpu")

    embeddings  = cache["embeddings"].float()          # [N, 256]
    labels      = cache["labels"].long()               # [N]
    sources     = cache["sources"]                     # List[str], length N
    audio_names = cache.get("audio_names", [""] * len(labels))

    N = len(labels)
    print(f"Loaded {N} eval samples.")

    # ------------------------------------------------------------------
    # Load Stage 2 classifier
    # ------------------------------------------------------------------
    print(f"Loading Stage 2 checkpoint: {args.stage2_ckpt}")
    ckpt = torch.load(args.stage2_ckpt, map_location=device)

    model = ClassifierHead(args.hidden_dim).to(device)
    model.load_state_dict(ckpt["classifier_state_dict"])
    model.eval()

    # ------------------------------------------------------------------
    # Run inference (batched to avoid OOM on large eval sets)
    # ------------------------------------------------------------------
    all_scores = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch_emb = embeddings[start : start + batch_size].to(device)
            logits = model(batch_emb)           # [B]
            all_scores.append(logits.cpu())

    scores_tensor = torch.cat(all_scores)       # [N]
    scores_np = scores_tensor.numpy()
    labels_np = labels.numpy()

    # ------------------------------------------------------------------
    # Overall EER
    # ------------------------------------------------------------------
    overall_eer, overall_thresh = compute_eer(labels_np, scores_np)

    # ------------------------------------------------------------------
    # Score distribution stats
    # ------------------------------------------------------------------
    bon_mask = labels_np == 1
    spo_mask = labels_np == 0

    def _stats(arr):
        if len(arr) == 0:
            return dict(mean=float("nan"), std=float("nan"),
                        min=float("nan"), max=float("nan"))
        return dict(mean=arr.mean(), std=arr.std(),
                    min=arr.min(),  max=arr.max())

    score_stats = {
        "n_total": N,
        "n_bon":   int(bon_mask.sum()),
        "n_spo":   int(spo_mask.sum()),
        **{f"bon_{k}": v for k, v in _stats(scores_np[bon_mask]).items()},
        **{f"spo_{k}": v for k, v in _stats(scores_np[spo_mask]).items()},
    }

    # ------------------------------------------------------------------
    # Per-generator EER
    # ------------------------------------------------------------------
    unique_sources = sorted(set(sources))
    per_source = {}

    for src in unique_sources:
        mask     = np.array([s == src for s in sources])
        src_lbl  = labels_np[mask]
        src_scr  = scores_np[mask]
        eer, _   = compute_eer(src_lbl, src_scr)
        per_source[src] = {
            "eer":   eer,
            "n_bon": int((src_lbl == 1).sum()),
            "n_spo": int((src_lbl == 0).sum()),
        }

    # ------------------------------------------------------------------
    # Format and output
    # ------------------------------------------------------------------
    result_str = format_results(
        speaker=args.speaker or "unknown",
        overall_eer=overall_eer,
        overall_thresh=overall_thresh,
        per_source=per_source,
        score_stats=score_stats,
    )

    print("\n" + result_str)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result_str)
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()