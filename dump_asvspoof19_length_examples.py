#!/usr/bin/env python3
"""
Dump short/exact/long ASVspoof2019 examples to disk to inspect preprocessing.
"""

import argparse
import os
from pathlib import Path

import soundfile as sf
import torch
import pandas as pd

from data_loader import ASVspoof2019Dataset, InTheWildDataset


def get_duration_seconds(path: Path) -> float:
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return info.frames / float(info.samplerate)
    except Exception:
        return 0.0
    return 0.0


def pick_examples(dataset, max_duration, exact_tolerance, limit):
    short = []
    exact = []
    long = []

    for idx, item in enumerate(dataset.data):
        path = item[0]
        duration = get_duration_seconds(path)
        if duration <= 0:
            continue
        if abs(duration - max_duration) <= exact_tolerance:
            if len(exact) < limit:
                exact.append((idx, path, duration))
        elif duration < max_duration:
            if len(short) < limit:
                short.append((idx, path, duration))
        else:
            if len(long) < limit:
                long.append((idx, path, duration))
        if len(short) >= limit and len(exact) >= limit and len(long) >= limit:
            break

    return short, exact, long


def write_wave(path: Path, audio, sample_rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def save_example(ds, idx, raw_path, duration, out_dir: Path):
    raw_audio, raw_sr = sf.read(str(raw_path))
    if raw_audio.ndim > 1:
        raw_audio = raw_audio.mean(axis=1)
    write_wave(out_dir / "raw.wav", raw_audio, raw_sr)

    item = ds[idx]
    waveform = item[0]
    label = int(item[1].item()) if torch.is_tensor(item[1]) else int(item[1])

    if waveform.ndim == 1:
        write_wave(out_dir / "processed.wav", waveform.cpu().numpy(), ds.target_sample_rate)
    else:
        for i in range(waveform.shape[0]):
            name = f"processed_chunk_{i:02d}.wav"
            write_wave(out_dir / name, waveform[i].cpu().numpy(), ds.target_sample_rate)

    meta_path = out_dir / "meta.txt"
    with open(meta_path, "w", encoding="utf-8") as handle:
        handle.write(f"path={raw_path}\n")
        handle.write(f"duration_seconds={duration:.4f}\n")
        handle.write(f"label={label}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    parser.add_argument("--protocol_file", required=True)
    parser.add_argument("--itw_root_dir")
    parser.add_argument("--itw_protocol_file")
    parser.add_argument("--out_dir", default="debug_audio_samples")
    parser.add_argument("--max_duration_seconds", type=float, default=5.0)
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--exact_tolerance", type=float, default=0.01)
    args = parser.parse_args()

    ds = ASVspoof2019Dataset(
        root_dir=args.root_dir,
        protocol_file=args.protocol_file,
        subset="all",
        max_duration_seconds=int(args.max_duration_seconds),
        target_sample_rate=args.target_sample_rate,
    )

    short, exact, long = pick_examples(
        ds, args.max_duration_seconds, args.exact_tolerance, args.limit
    )

    base = Path(args.out_dir)
    for group_name, group in (("short", short), ("exact", exact), ("long", long)):
        for i, (idx, path, duration) in enumerate(group):
            sample_dir = base / group_name / f"{i:02d}_{path.stem}"
            save_example(ds, idx, path, duration, sample_dir)

    print(
        f"Wrote samples to {base} "
        f"(short={len(short)}, exact={len(exact)}, long={len(long)})"
    )

    if args.itw_root_dir and args.itw_protocol_file:
        itw_ds = InTheWildDataset(
            root_dir=args.itw_root_dir,
            protocol_file=args.itw_protocol_file,
            subset="all",
            max_duration_seconds=int(args.max_duration_seconds),
            target_sample_rate=args.target_sample_rate,
        )

        proto = pd.read_csv(args.itw_protocol_file)
        proto["label"] = proto["label"].replace("bona-fide", "bonafide")
        proto["path"] = proto["file"].apply(lambda name: str(Path(args.itw_root_dir) / name))

        itw_items = []
        for idx, row in proto.iterrows():
            path = Path(row["path"])
            duration = get_duration_seconds(path)
            if duration <= 0:
                continue
            itw_items.append((idx, path, duration))

        short = []
        exact = []
        long = []
        for idx, path, duration in itw_items:
            if abs(duration - args.max_duration_seconds) <= args.exact_tolerance:
                if len(exact) < args.limit:
                    exact.append((idx, path, duration))
            elif duration < args.max_duration_seconds:
                if len(short) < args.limit:
                    short.append((idx, path, duration))
            else:
                if len(long) < args.limit:
                    long.append((idx, path, duration))
            if len(short) >= args.limit and len(exact) >= args.limit and len(long) >= args.limit:
                break

        base = Path(args.out_dir) / "itw"
        for group_name, group in (("short", short), ("exact", exact), ("long", long)):
            for i, (idx, path, duration) in enumerate(group):
                sample_dir = base / group_name / f"{i:02d}_{path.stem}"
                save_example(itw_ds, idx, path, duration, sample_dir)

        print(
            f"Wrote ITW samples to {base} "
            f"(short={len(short)}, exact={len(exact)}, long={len(long)})"
        )


if __name__ == "__main__":
    main()
