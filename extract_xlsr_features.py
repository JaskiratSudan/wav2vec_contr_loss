"""
extract_xlsr_features.py

Runs train / dev / eval splits through the frozen XLSR encoder and saves
ONE consolidated .pt file per split (not per-sample files).

Output format per split:
    {
        "features":    fp16 Tensor [N, 1024, T_max],  # zero-padded along T
        "lengths":     LongTensor  [N],               # actual T per sample
        "labels":      LongTensor  [N],               # 1=bonafide, 0=spoof
        "sources":     List[str]   length N,
        "audio_names": List[str]   length N,
    }

Fake sampling (70% per generator) is applied to the TRAIN split only.
Dev and eval use all rows in their protocol files.

Env vars
--------
RESUME_CKPT       : Stage 1 checkpoint to load encoder weights from (required)
CACHE_DIR         : root directory to write xlsr_train.pt / dev / eval (required)
FF_TRAIN_PROTOCOL : path to Barack_Obama_train.txt
FF_DEV_PROTOCOL   : path to Barack_Obama_dev.txt
FF_EVAL_PROTOCOL  : path to Barack_Obama_eval.txt
FF_ROOT           : optional root prefix for relative audio paths
FF_FAKE_FRAC      : fraction of spoof rows to keep per generator, train only (default: 0.70)
FF_SEED           : random seed (default: 42)
BATCH_SIZE        : inference batch size (default: 16)
NUM_WORKERS       : dataloader workers (default: 4)
"""

import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import FamousFiguresDataset
from encoder import Wav2Vec2Encoder
from stage1_config import build_config


# ---------------------------------------------------------------------------
# Per-generator fake sampling
# ---------------------------------------------------------------------------

def sample_per_class(data: list, frac: float, seed: int) -> list:
    """
    data : list of rows from FamousFiguresDataset.rows
           each row is (audio_path, label_int, speaker, source)
    Samples `frac` of bonafide rows (randomly) AND `frac` of spoof rows
    per generator independently.
    """
    rng = random.Random(seed)

    # Sample bonafide
    bonafide = [r for r in data if r[1] == 1]
    rng.shuffle(bonafide)
    n_bon_keep = max(1, round(len(bonafide) * frac))
    sampled_bonafide = bonafide[:n_bon_keep]
    print(f"  [sampling] {'bonafide (real)':<30} total={len(bonafide):>6}  kept={n_bon_keep:>6}")

    # Sample spoof per generator
    by_source = defaultdict(list)
    for r in data:
        if r[1] == 0:
            by_source[r[3]].append(r)

    sampled_spoof = []
    for source, group in sorted(by_source.items()):
        rng.shuffle(group)
        n_keep = max(1, round(len(group) * frac))
        sampled_spoof.extend(group[:n_keep])
        print(f"  [sampling] {source:<30} total={len(group):>6}  kept={n_keep:>6}")

    result = sampled_bonafide + sampled_spoof
    rng.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_with_name(batch):
    waveforms, labels, speakers, sources, names = zip(*batch)
    padded = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    attn = (padded != 0.0).long()
    return padded, attn, torch.stack(list(labels)), speakers, sources, names


# ---------------------------------------------------------------------------
# Extraction — returns consolidated tensors for one split
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_split(
    encoder: Wav2Vec2Encoder,
    protocol_file: str,
    split_name: str,
    cache_dir: Path,
    cfg,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    fake_frac: float = None,
    seed: int = 42,
) -> dict:

    ds = FamousFiguresDataset(
        protocol_file=protocol_file,
        root_dir=os.environ.get("FF_ROOT", ""),
        subset="all",
        target_sample_rate=cfg.target_sample_rate,
        max_duration_seconds=cfg.max_duration_seconds,
        return_audio_name=True,
    )

    # Apply fake sampling to train split only
    if fake_frac is not None and fake_frac < 1.0 and split_name == "train":
        print(f"  Applying {fake_frac*100:.0f}% per-generator fake sampling...")
        bon = sum(1 for r in ds.rows if r[1] == 1)
        spo = sum(1 for r in ds.rows if r[1] == 0)
        print(f"  Before: bonafide={bon}  spoof={spo}  total={len(ds.rows)}")
        ds.rows = sample_per_class(ds.rows, frac=fake_frac, seed=seed)
        bon = sum(1 for r in ds.rows if r[1] == 1)
        spo = sum(1 for r in ds.rows if r[1] == 0)
        print(f"  After:  bonafide={bon}  spoof={spo}  total={len(ds.rows)}")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_with_name,
    )

    # Shard setup — flush to disk every SHARD_SIZE samples, never accumulate all in RAM
    SHARD_SIZE    = 1000
    shard_dir     = cache_dir / f"{split_name}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    buf_features  = []
    buf_lengths   = []
    buf_labels    = []
    buf_sources   = []
    buf_names     = []
    shard_idx     = 0
    shard_paths   = []

    def flush_shard():
        nonlocal shard_idx
        path = shard_dir / f"shard_{shard_idx:04d}.pt"
        torch.save({
            "features":    buf_features[:],
            "lengths":     torch.tensor(buf_lengths, dtype=torch.long),
            "labels":      torch.tensor(buf_labels,  dtype=torch.long),
            "sources":     buf_sources[:],
            "audio_names": buf_names[:],
        }, path)
        shard_paths.append(str(path))
        shard_idx += 1
        buf_features.clear(); buf_lengths.clear()
        buf_labels.clear();   buf_sources.clear(); buf_names.clear()

    total = len(ds)
    n_done = 0
    t0 = time.time()

    encoder.eval()
    for batch_idx, (waveforms, attn, labels, speakers, sources, names) in enumerate(loader):
        waveforms = waveforms.to(device)
        attn      = attn.to(device)

        hidden = encoder(waveforms, attn)
        if hidden.dim() == 4:
            hidden = hidden.mean(dim=1)
        hidden = hidden.permute(0, 2, 1).cpu().half()  # (B, 1024, T) fp16

        for i in range(hidden.shape[0]):
            feat = hidden[i]
            buf_features.append(feat)
            buf_lengths.append(feat.shape[1])
            buf_labels.append(int(labels[i].item()))
            buf_sources.append(str(sources[i]))
            buf_names.append(str(Path(names[i]).stem))

        n_done += len(names)

        if len(buf_features) >= SHARD_SIZE:
            flush_shard()

        if (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / n_done * (total - n_done) if n_done < total else 0
            print(f"  [{split_name}] {n_done}/{total}  shards={shard_idx}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    if buf_features:
        flush_shard()

    elapsed = time.time() - t0
    print(f"  [{split_name}] Done — {n_done} samples  "
          f"{len(shard_paths)} shards  time={elapsed:.1f}s")

    return {
        "shard_dir":   str(shard_dir),
        "shard_paths": shard_paths,
        "n_samples":   n_done,
    }



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = build_config()

    ckpt_path = os.environ.get("RESUME_CKPT", "")
    cache_dir = Path(os.environ.get("CACHE_DIR", ""))
    if not ckpt_path or not Path(ckpt_path).is_file():
        raise ValueError(f"RESUME_CKPT must point to a valid checkpoint. Got: '{ckpt_path}'")
    if not cache_dir:
        raise ValueError("CACHE_DIR must be set")

    cache_dir.mkdir(parents=True, exist_ok=True)

    batch_size  = int(os.environ.get("BATCH_SIZE",   "16"))
    num_workers = int(os.environ.get("NUM_WORKERS",  "4"))
    fake_frac   = float(os.environ.get("FF_FAKE_FRAC", "0.70"))
    seed        = int(os.environ.get("FF_SEED",       "42"))

    splits = {
        "train": os.environ.get("FF_TRAIN_PROTOCOL", ""),
        "dev":   os.environ.get("FF_DEV_PROTOCOL",   ""),
        "eval":  os.environ.get("FF_EVAL_PROTOCOL",  ""),
    }
    missing = [k for k, v in splits.items() if not v]
    if missing:
        raise ValueError(f"Missing protocol env vars for splits: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice     : {device}")
    print(f"Cache dir  : {cache_dir}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Fake frac  : {fake_frac} (train only)\n")

    # Build encoder and load weights
    encoder = Wav2Vec2Encoder(
        model_name=cfg.model_name,
        freeze_encoder=True,
    ).to(device)
    if hasattr(encoder, "model") and hasattr(encoder.model, "config"):
        if hasattr(encoder.model.config, "layerdrop"):
            encoder.model.config.layerdrop = 0.0

    ckpt = torch.load(ckpt_path, map_location=device)
    if "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
        print(f"Loaded encoder weights from checkpoint (epoch {ckpt.get('epoch', '?')})")
    else:
        print("No encoder_state_dict in checkpoint — using pretrained XLS-R weights")

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    # Extract and save each split
    for split_name, protocol_file in splits.items():
        out_path = cache_dir / f"xlsr_{split_name}.pt"

        if out_path.exists():
            print(f"\n--- Skipping {split_name} (already exists: {out_path}) ---")
            continue

        print(f"\n--- Extracting: {split_name} ---")
        cache = extract_split(
            encoder=encoder,
            protocol_file=protocol_file,
            split_name=split_name,
            cache_dir=cache_dir,
            cfg=cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
            fake_frac=fake_frac if split_name == "train" else None,
            seed=seed,
        )

        # Save manifest (shard paths) — actual data is already on disk in shards
        torch.save(cache, out_path)
        print(f"  Manifest saved: {out_path}  ({cache['n_samples']} samples, {len(cache['shard_paths'])} shards)")

    print("\nAll splits extracted.")
    for split_name in splits:
        p = cache_dir / f"xlsr_{split_name}.pt"
        if p.exists():
            m = torch.load(p, map_location="cpu")
            print(f"  {split_name}: {m['n_samples']} samples  {len(m['shard_paths'])} shards  -> {m['shard_dir']}")


if __name__ == "__main__":
    main()