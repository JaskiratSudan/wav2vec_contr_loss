#!/usr/bin/env python3
import argparse
import csv
import os
import multiprocessing as mp
from functools import partial
from pathlib import Path

import librosa
from tqdm import tqdm

from data_loader import ASVspoof2021DFDataset, ASVspoof2021LADataset
from eval_datasets import (
    DEFAULT_ASV21_DF_PROTOCOL,
    DEFAULT_ASV21_DF_ROOT,
    DEFAULT_ASV21_LA_PROTOCOL,
    DEFAULT_ASV21_LA_ROOT,
)


def _check_corrupted(path: str, target_sample_rate: int):
    try:
        librosa.load(path, sr=target_sample_rate, mono=True)
        return None
    except Exception:
        return path


def _collect_missing_and_corrupted(dataset, label: str, workers: int):
    missing = []
    corrupted = []

    existing_paths = []
    for audio_path, *_ in dataset.data:
        path = Path(audio_path)
        if not path.exists():
            missing.append(str(path))
        else:
            existing_paths.append(str(path))

    if workers <= 1:
        for path in tqdm(existing_paths, desc=f"Checking {label}", unit="file"):
            if _check_corrupted(path, dataset.target_sample_rate):
                corrupted.append(path)
        return missing, corrupted

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers) as pool:
        check_fn = partial(_check_corrupted, target_sample_rate=dataset.target_sample_rate)
        iterator = pool.imap_unordered(check_fn, existing_paths, chunksize=32)
        for result in tqdm(iterator, total=len(existing_paths), desc=f"Checking {label}", unit="file"):
            if result is not None:
                corrupted.append(result)

    return missing, corrupted


def _write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["dataset", "missing_path", "corrupted_path"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Check ASVspoof2021 DF/LA datasets for missing or corrupted audio files."
    )
    parser.add_argument(
        "--df-protocol",
        default=DEFAULT_ASV21_DF_PROTOCOL,
        help="Path to DF protocol file",
    )
    parser.add_argument(
        "--df-root",
        default=DEFAULT_ASV21_DF_ROOT,
        help="Root dir for DF audio files",
    )
    parser.add_argument(
        "--la-protocol",
        default=DEFAULT_ASV21_LA_PROTOCOL,
        help="Path to LA protocol file",
    )
    parser.add_argument(
        "--la-root",
        default=DEFAULT_ASV21_LA_ROOT,
        help="Root dir for LA audio files",
    )
    parser.add_argument(
        "--output-csv",
        default="asv2021_missing_corrupt.csv",
        help="Output CSV path",
    )
    parser.add_argument("--subset", default="all", choices=["all", "bonafide", "spoof"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=1337)
    parser.add_argument("--df-ext", default=".flac", help="File extension for DF files")
    parser.add_argument("--la-ext", default=".flac", help="File extension for LA files")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes to use",
    )
    args = parser.parse_args()

    df_dataset = ASVspoof2021DFDataset(
        protocol_file=args.df_protocol,
        root_dir=args.df_root,
        num_samples=args.num_samples,
        subset=args.subset,
        sample_seed=args.sample_seed,
        file_ext=args.df_ext,
        skip_missing=False,
    )
    la_dataset = ASVspoof2021LADataset(
        protocol_file=args.la_protocol,
        root_dir=args.la_root,
        num_samples=args.num_samples,
        subset=args.subset,
        sample_seed=args.sample_seed,
        file_ext=args.la_ext,
        skip_missing=False,
    )

    rows = []
    df_missing, df_corrupted = _collect_missing_and_corrupted(df_dataset, "DF", args.workers)
    la_missing, la_corrupted = _collect_missing_and_corrupted(la_dataset, "LA", args.workers)

    for path in df_missing:
        rows.append({"dataset": "DF", "missing_path": path, "corrupted_path": ""})
    for path in df_corrupted:
        rows.append({"dataset": "DF", "missing_path": "", "corrupted_path": path})
    for path in la_missing:
        rows.append({"dataset": "LA", "missing_path": path, "corrupted_path": ""})
    for path in la_corrupted:
        rows.append({"dataset": "LA", "missing_path": "", "corrupted_path": path})

    _write_csv(rows, Path(args.output_csv))

    print(
        f"DF missing: {len(df_missing)} | DF corrupted: {len(df_corrupted)} | "
        f"LA missing: {len(la_missing)} | LA corrupted: {len(la_corrupted)}"
    )
    print(f"Wrote CSV to: {args.output_csv}")


if __name__ == "__main__":
    main()
