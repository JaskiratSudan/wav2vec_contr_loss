#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from generate_eval_score_file import load_stage2_head, write_cm_scores_from_embeddings


def _run_if_exists(emb_path: Path, lab_path: Path, score_path: Path,
                   clf: torch.nn.Module, device: torch.device, utt_prefix: str):
    if not emb_path.exists() or not lab_path.exists():
        print(f"[SKIP] Missing embeddings or labels: {emb_path} / {lab_path}")
        return
    write_cm_scores_from_embeddings(
        emb_path=str(emb_path),
        label_path=str(lab_path),
        clf=clf,
        device=device,
        score_path=str(score_path),
        utt_prefix=utt_prefix,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--asv19_dir", type=str)
    parser.add_argument("--asv5_dir", type=str)
    parser.add_argument("--itw_dir", type=str)
    parser.add_argument("--ff_dir", type=str)
    parser.add_argument("--fakexpose_dir", type=str)
    parser.add_argument("--mlaad_dir", type=str)

    parser.add_argument("--score_asv19", type=str)
    parser.add_argument("--score_asv5", type=str)
    parser.add_argument("--score_itw", type=str)
    parser.add_argument("--score_ff", type=str)
    parser.add_argument("--score_fakexpose", type=str)
    parser.add_argument("--score_mlaad", type=str)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    clf = load_stage2_head(args.stage2_ckpt, device=device)

    if args.asv19_dir and args.score_asv19:
        base = Path(args.asv19_dir)
        _run_if_exists(base / "eval_embeddings.npy", base / "eval_labels.npy",
                       Path(args.score_asv19), clf, device, "asv19_eval")

    if args.asv5_dir and args.score_asv5:
        base = Path(args.asv5_dir)
        _run_if_exists(base / "eval_embeddings.npy", base / "eval_labels.npy",
                       Path(args.score_asv5), clf, device, "asv5_eval")

    if args.itw_dir and args.score_itw:
        base = Path(args.itw_dir)
        _run_if_exists(base / "itw_embeddings.npy", base / "itw_labels.npy",
                       Path(args.score_itw), clf, device, "itw")

    if args.ff_dir and args.score_ff:
        base = Path(args.ff_dir)
        _run_if_exists(base / "ff_embeddings.npy", base / "ff_labels.npy",
                       Path(args.score_ff), clf, device, "ff")

    if args.fakexpose_dir and args.score_fakexpose:
        base = Path(args.fakexpose_dir)
        _run_if_exists(base / "fakexpose_embeddings.npy", base / "fakexpose_labels.npy",
                       Path(args.score_fakexpose), clf, device, "fakexpose")

    if args.mlaad_dir and args.score_mlaad:
        base = Path(args.mlaad_dir)
        _run_if_exists(base / "dev_embeddings.npy", base / "dev_labels.npy",
                       Path(args.score_mlaad), clf, device, "mlaad_dev")


if __name__ == "__main__":
    main()
