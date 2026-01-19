#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np

from evaluation import compute_eer


def _load_scores(path: Path):
    data = np.genfromtxt(path, dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"Score file must have 4 columns: {path}")

    keys = data[:, 2]
    scores = data[:, 3].astype(float)
    labels = np.where(keys == "bonafide", 1, 0).astype(int)
    return scores, labels


def _eer_and_acc(scores: np.ndarray, labels: np.ndarray):
    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]
    eer, threshold = compute_eer(bona_scores, spoof_scores)
    preds = (scores >= threshold).astype(int)
    acc = (preds == labels).mean()
    return eer * 100.0, acc * 100.0, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Compute EER and accuracy from score files."
    )
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        help="Name=path pair, e.g. ASV5=/path/score_cm_eval_asv5.txt (repeatable).",
    )
    args = parser.parse_args()

    if not args.score:
        raise SystemExit("Provide at least one --score Name=path entry.")

    results = []
    for entry in args.score:
        if "=" in entry:
            name, path_str = entry.split("=", 1)
        else:
            path_str = entry
            name = Path(path_str).stem
        path = Path(path_str)
        scores, labels = _load_scores(path)
        eer, acc, threshold = _eer_and_acc(scores, labels)
        results.append((name, path, eer, acc, threshold, scores.size))

    name_w = max(len(r[0]) for r in results) if results else 4
    eer_w = 7
    acc_w = 7
    n_w = max(len(str(r[5])) for r in results) if results else 1
    print("EER/Accuracy from score files")
    header = f"{'NAME':<{name_w}}  {'EER%':>{eer_w}}  {'ACC%':>{acc_w}}  {'N':>{n_w}}  SCORE_FILE"
    print(header)
    print("-" * len(header))
    for name, path, eer, acc, threshold, n in results:
        print(
            f"{name:<{name_w}}  {eer:>{eer_w}.3f}  {acc:>{acc_w}.2f}  {n:>{n_w}d}  {path}"
        )


if __name__ == "__main__":
    main()
