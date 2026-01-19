#!/usr/bin/env python3
"""Compute EERs from score files under the scores directory."""

import argparse
import os

import numpy as np

from evaluation import calculate_EER, compute_eer


TARGET_FILES = {
    "score_cm_eval.txt": "asv",
    "score_cm_itw.txt": "itw",
}


def iter_experiments(scores_dir):
    for entry in os.scandir(scores_dir):
        if entry.is_dir():
            yield entry.name, entry.path


def find_score_files(exp_path):
    matches = []
    for dirpath, _, filenames in os.walk(exp_path):
        for filename in filenames:
            tag = TARGET_FILES.get(filename)
            if tag:
                path = os.path.join(dirpath, filename)
                matches.append((tag, path))
    return sorted(matches, key=lambda item: (item[0], item[1]))

def compute_accuracy_at_eer(score_path: str) -> float:
    cm_data = np.genfromtxt(score_path, dtype=str)
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(float)
    labels = (cm_keys == "bonafide").astype(int)
    bonafide_scores = cm_scores[labels == 1]
    spoof_scores = cm_scores[labels == 0]
    _, eer_thresh = compute_eer(bonafide_scores, spoof_scores)
    preds = (cm_scores >= eer_thresh).astype(int)
    return float((preds == labels).mean() * 100.0)


def main():
    parser = argparse.ArgumentParser(
        description="Print EERs for ASV and ITW score files in scores/"
    )
    parser.add_argument(
        "--scores-dir",
        default="scores",
        help="Path to scores directory (default: scores)",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Include score file paths in output",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("scores", "results.txt"),
        help="Path to write results table (default: scores/results.txt)",
    )
    args = parser.parse_args()

    scores_dir = args.scores_dir
    if not os.path.isdir(scores_dir):
        raise SystemExit(f"Scores directory not found: {scores_dir}")

    results = {"asv": [], "itw": []}
    for exp_name, exp_path in sorted(iter_experiments(scores_dir)):
        score_files = find_score_files(exp_path)
        if not score_files:
            continue
        for tag, path in score_files:
            eer = calculate_EER(path)
            acc = compute_accuracy_at_eer(path)
            rel_path = os.path.relpath(path, scores_dir)
            results.setdefault(tag, []).append((exp_name, eer, acc, rel_path))

    output_lines = []
    for tag in ("asv", "itw"):
        rows = results.get(tag, [])
        if not rows:
            continue
        output_lines.append(tag)
        output_lines.append(
            "experiment\teer\tacc" + ("\tpath" if args.show_paths else "")
        )
        for exp_name, eer, acc, rel_path in rows:
            if args.show_paths:
                output_lines.append(f"{exp_name}\t{eer:.4f}\t{acc:.2f}\t{rel_path}")
            else:
                output_lines.append(f"{exp_name}\t{eer:.4f}\t{acc:.2f}")
        output_lines.append("")

    output_text = "\n".join(output_lines).rstrip() + "\n"
    print(output_text, end="")
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(output_text)


if __name__ == "__main__":
    main()
