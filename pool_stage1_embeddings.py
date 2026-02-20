#!/usr/bin/env python3
import argparse
import os
import json
from typing import Dict, List, Tuple

import numpy as np


def _load_split(d: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    emb_path = os.path.join(d, f"{split}_embeddings.npy")
    lab_path = os.path.join(d, f"{split}_labels.npy")
    X = np.load(emb_path)
    y = np.load(lab_path)
    return X, y


def _exists_split(d: str, split: str) -> bool:
    return (
        os.path.exists(os.path.join(d, f"{split}_embeddings.npy"))
        and os.path.exists(os.path.join(d, f"{split}_labels.npy"))
    )


def pool_split(
    split: str,
    sources: List[Tuple[str, str]],  # (name, dir)
    out_dir: str,
    require_any: bool = True,
) -> Dict:
    Xs, ys = [], []
    per_source = {}

    d0 = None
    for name, d in sources:
        if not _exists_split(d, split):
            per_source[name] = {"used": False, "reason": "missing_split_files"}
            continue

        X, y = _load_split(d, split)
        if X.ndim != 2:
            raise ValueError(f"[{name}] {split}_embeddings.npy must be 2D, got {X.shape}")
        if y.ndim != 1:
            y = y.reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"[{name}] {split}: X rows {X.shape[0]} != y rows {y.shape[0]}")

        if d0 is None:
            d0 = X.shape[1]
        else:
            if X.shape[1] != d0:
                raise ValueError(
                    f"[{name}] {split}: embedding dim mismatch {X.shape[1]} vs expected {d0}"
                )

        Xs.append(X)
        ys.append(y.astype(np.int64, copy=False))

        uniq, cnt = np.unique(y, return_counts=True)
        per_source[name] = {
            "used": True,
            "n": int(len(y)),
            "label_hist": {str(int(u)): int(c) for u, c in zip(uniq, cnt)},
            "dir": d,
        }

    if require_any and len(Xs) == 0:
        raise RuntimeError(
            f"No inputs had split='{split}'. Nothing to pool. "
            f"Check your input dirs or pass --allow_missing_splits."
        )

    if len(Xs) == 0:
        return {"split": split, "pooled": False, "reason": "no_inputs_with_split", "per_source": per_source}

    Xp = np.concatenate(Xs, axis=0)
    yp = np.concatenate(ys, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_embeddings.npy"), Xp)
    np.save(os.path.join(out_dir, f"{split}_labels.npy"), yp)

    uniq, cnt = np.unique(yp, return_counts=True)
    pooled_hist = {str(int(u)): int(c) for u, c in zip(uniq, cnt)}

    return {
        "split": split,
        "pooled": True,
        "n": int(yp.shape[0]),
        "dim": int(Xp.shape[1]),
        "label_hist": pooled_hist,
        "per_source": per_source,
    }


def parse_sources(in_dirs: str, names: str) -> List[Tuple[str, str]]:
    dirs = [x.strip() for x in in_dirs.split(",") if x.strip()]
    if not dirs:
        raise ValueError("Empty --in_dirs")

    if names:
        ns = [x.strip() for x in names.split(",") if x.strip()]
        if len(ns) != len(dirs):
            raise ValueError("--names must have same length as --in_dirs")
    else:
        ns = [os.path.basename(d.rstrip("/")) or f"src{i}" for i, d in enumerate(dirs)]

    sources = []
    for n, d in zip(ns, dirs):
        if not os.path.isdir(d):
            raise ValueError(f"Input dir does not exist: {d}")
        sources.append((n, d))
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dirs", required=True,
                    help="Comma-separated dirs that contain {train,dev,eval}_embeddings.npy and *_labels.npy")
    ap.add_argument("--names", default="",
                    help="Optional comma-separated names for the dirs (same count as in_dirs)")
    ap.add_argument("--out_dir", required=True, help="Output emb_dir for Stage-2 (pooled)")
    ap.add_argument("--splits", default="train,dev",
                    help="Comma-separated splits to pool (default: train,dev). You can add eval.")
    ap.add_argument("--allow_missing_splits", action="store_true",
                    help="If set: don't error when a split is missing in all inputs; just skip it.")
    args = ap.parse_args()

    sources = parse_sources(args.in_dirs, args.names)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    summary = {"inputs": [{"name": n, "dir": d} for n, d in sources], "out_dir": args.out_dir, "splits": {}}

    for split in splits:
        info = pool_split(
            split=split,
            sources=sources,
            out_dir=args.out_dir,
            require_any=not args.allow_missing_splits,
        )
        summary["splits"][split] = info
        if info.get("pooled"):
            print(f"[OK] pooled split='{split}' -> n={info['n']} dim={info['dim']} labels={info['label_hist']}")
        else:
            print(f"[SKIP] split='{split}': {info.get('reason')}")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pooling_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[OK] wrote summary -> {os.path.join(args.out_dir, 'pooling_summary.json')}")


if __name__ == "__main__":
    main()
