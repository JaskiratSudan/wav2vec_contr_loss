"""
split_famousfigures.py

1. Load protocol → filter to one speaker
2. Print per-generator sample counts
3. Split 70 / 10 / 20 stratified by (Label × Source) — every class and
   every generator is split independently so each split is representative
4. Write three tab-separated protocol files in the same format as the original

Usage
-----
    python split_famousfigures.py \
        --protocol  /nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt \
        --speaker   Anthony_Blinken \
        --out_dir   /nfs/turbo/umd-hafiz/issf_server_data/famousfigures/splits

Output files (in --out_dir)
---------------------------
    Anthony_Blinken_train.txt
    Anthony_Blinken_dev.txt
    Anthony_Blinken_eval.txt
"""

import argparse
import random
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Step 1 — load & filter
# ---------------------------------------------------------------------------

def load_protocol(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")
    df["Label"] = df["Label"].astype(str).str.lower().str.replace("bona-fide", "bonafide")
    return df


def filter_speaker(df: pd.DataFrame, speaker: str) -> pd.DataFrame:
    out = df[df["Speaker"].astype(str) == speaker].copy()
    if len(out) == 0:
        available = sorted(df["Speaker"].unique())
        raise ValueError(
            f"Speaker '{speaker}' not found in protocol.\n"
            f"Available speakers:\n  " + "\n  ".join(available)
        )
    return out


# ---------------------------------------------------------------------------
# Step 2 — print per-generator counts
# ---------------------------------------------------------------------------

def print_generator_counts(df: pd.DataFrame, speaker: str):
    print(f"\n{'='*60}")
    print(f"  Speaker: {speaker}  ({len(df):,} total samples)")
    print(f"{'='*60}")
    print(f"  {'Source':<30} {'bonafide':>10} {'spoof':>10} {'total':>10}")
    print(f"  {'-'*60}")

    sources = sorted(df["Source"].unique())
    for src in sources:
        sub = df[df["Source"] == src]
        bon = int((sub["Label"] == "bonafide").sum())
        spo = int((sub["Label"] != "bonafide").sum())
        print(f"  {src:<30} {bon:>10} {spo:>10} {bon+spo:>10}")

    print(f"  {'-'*60}")
    bon = int((df["Label"] == "bonafide").sum())
    spo = int((df["Label"] != "bonafide").sum())
    print(f"  {'TOTAL':<30} {bon:>10} {spo:>10} {len(df):>10}\n")


# ---------------------------------------------------------------------------
# Step 3 — stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    df: pd.DataFrame,
    train_frac: float,
    dev_frac: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each (Label × Source) group, shuffle and split into train / dev / eval.
    Guarantees at least 1 row in dev and eval for every group that has >= 3 rows.
    Leftover rows (from rounding) go to train.
    """
    rng = random.Random(seed)
    train_idx, dev_idx, eval_idx = [], [], []

    print(f"  {'Source × Label':<40} {'total':>7} {'train':>7} {'dev':>7} {'eval':>7}")
    print(f"  {'-'*70}")

    for (label, source), group in df.groupby(["Label", "Source"]):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n = len(indices)

        if n < 3:
            # Too small to split — put everything in train
            train_idx.extend(indices)
            tag = f"{source} / {label}"
            print(f"  {tag:<40} {n:>7} {n:>7} {'0':>7} {'0':>7}  <- too small, all->train")
            continue

        n_dev  = max(1, round(n * dev_frac))
        n_eval = max(1, round(n * (1.0 - train_frac - dev_frac)))
        # Guard: never let dev+eval consume everything
        if n_dev + n_eval >= n:
            n_dev  = max(1, n_dev - 1)
            n_eval = max(1, n - n_dev - 1)
        n_train = n - n_dev - n_eval

        train_idx.extend(indices[:n_train])
        dev_idx.extend(  indices[n_train : n_train + n_dev])
        eval_idx.extend( indices[n_train + n_dev :])

        tag = f"{source} / {label}"
        print(f"  {tag:<40} {n:>7} {n_train:>7} {n_dev:>7} {n_eval:>7}")

    print(f"  {'-'*70}")
    t, d, e = len(train_idx), len(dev_idx), len(eval_idx)
    grand = t + d + e
    print(f"  {'TOTAL':<40} {grand:>7} {t:>7} {d:>7} {e:>7}")
    print(f"  {'%':<40} {'':>7} {t/grand*100:>6.1f}% {d/grand*100:>6.1f}% {e/grand*100:>6.1f}%\n")

    return df.loc[train_idx].copy(), df.loc[dev_idx].copy(), df.loc[eval_idx].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol",   required=True,
                        help="Path to the FamousFigures protocol file")
    parser.add_argument("--speaker",    required=True,
                        help="Speaker name, e.g. Anthony_Blinken")
    parser.add_argument("--out_dir",    required=True,
                        help="Directory to write the three output protocol files")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--dev_frac",   type=float, default=0.10)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    assert args.train_frac + args.dev_frac < 1.0, \
        "train_frac + dev_frac must be < 1.0 (remainder is eval)"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & filter
    print(f"\nLoading protocol: {args.protocol}")
    df = load_protocol(args.protocol)
    print(f"Total rows in protocol: {len(df):,}")

    df_spk = filter_speaker(df, args.speaker)

    # 2. Show per-generator counts
    print_generator_counts(df_spk, args.speaker)

    # 3. Split
    print(f"Splitting {len(df_spk):,} rows -> "
          f"train={args.train_frac*100:.0f}% / "
          f"dev={args.dev_frac*100:.0f}% / "
          f"eval={(1-args.train_frac-args.dev_frac)*100:.0f}%\n")

    train_df, dev_df, eval_df = stratified_split(
        df_spk,
        train_frac=args.train_frac,
        dev_frac=args.dev_frac,
        seed=args.seed,
    )

    # 4. Write
    for split_name, split_df in [("train", train_df), ("dev", dev_df), ("eval", eval_df)]:
        out_path = out_dir / f"{args.speaker}_{split_name}.txt"
        split_df.to_csv(out_path, sep="\t", index=False)
        bon = int((split_df["Label"] == "bonafide").sum())
        spo = int((split_df["Label"] != "bonafide").sum())
        print(f"  Written: {out_path}")
        print(f"           bonafide={bon}  spoof={spo}  total={len(split_df)}")

    print("\nDone.")


if __name__ == "__main__":
    main()