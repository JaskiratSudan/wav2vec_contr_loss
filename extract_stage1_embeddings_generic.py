#!/usr/bin/env python3
import os
import argparse
import random
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from data_loader import (
    ASVspoof2019Dataset,
    ASVspoof5Dataset,
    MLAADMailabsDataset,
    pad_collate_fn_speaker_source_multiclass,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=True)


def _cfg_get(cfg: dict, *keys, default=None):
    for k in keys:
        if k in cfg:
            return cfg[k]
    return default


class Stage1Backbone(torch.nn.Module):
    """
    Frozen Stage-1 encoder + compression head that outputs a single L2-normalized clip embedding per utterance.
    """
    def __init__(self, model_name: str, ckpt_path: str, device: torch.device):
        super().__init__()
        self.device = device

        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {}) or {}

        input_dim = int(_cfg_get(cfg, "INPUT_DIM", "input_dim", default=1024))
        hidden_dim = int(_cfg_get(cfg, "HIDDEN_DIM", "hidden_dim", default=256))
        dropout = float(_cfg_get(cfg, "DROPOUT", "dropout", default=0.1))

        self.encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)

        if "encoder_state_dict" in ckpt:
            load_state_dict_flexible(self.encoder, ckpt["encoder_state_dict"])
            print("[OK] Loaded finetuned encoder weights from checkpoint.")
        else:
            print("[INFO] No encoder_state_dict in ckpt (encoder likely frozen during Stage-1).")

        self.head = CompressionModule(input_dim, hidden_dim, dropout).to(device)
        load_state_dict_flexible(self.head, ckpt["compression_state_dict"])

        self.encoder.eval()
        self.head.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs_4d = self.encoder(waveforms, attention_mask=attention_mask)  # (B, K, F, T) in your codebase
        seq = self.head(hs_4d)                                         # (B, D, T)
        z = seq.mean(dim=-1)                                           # (B, D)
        z = F.normalize(z, p=2, dim=1)
        return z


def build_dataset(dataset: str, root_dir: str, protocol_file: str,
                  max_duration_seconds: int, target_sample_rate: int,
                  num_samples: str, seed: int):
    dataset = dataset.lower()
    ns = None if (num_samples is None or str(num_samples).lower() == "none") else int(num_samples)

    if dataset in {"asv19", "asvspoof2019", "2019"}:
        return ASVspoof2019Dataset(
            root_dir=root_dir,
            protocol_file=protocol_file,
            subset="all",
            num_samples=ns,
            target_sample_rate=target_sample_rate,
            max_duration_seconds=max_duration_seconds,
        )

    if dataset in {"asv5", "asvspoof5", "5"}:
        return ASVspoof5Dataset(
            root_dir=root_dir,
            protocol_file=protocol_file,
            subset="all",
            num_samples=ns,
            sample_seed=seed,
            target_sample_rate=target_sample_rate,
            max_duration_seconds=max_duration_seconds,
        )

    if dataset in {"mlaad"}:
        return MLAADMailabsDataset(
            root_dir=root_dir,
            protocol_file=protocol_file,
            subset="all",
            num_samples=ns,
            sample_seed=seed,
            target_sample_rate=target_sample_rate,
            max_duration_seconds=max_duration_seconds,
        )

    raise ValueError(f"Unknown --dataset={dataset}")


def extract_split(backbone: Stage1Backbone,
                  dataset_name: str,
                  split: str,
                  root_dir: str,
                  protocol_file: str,
                  out_dir: str,
                  device: torch.device,
                  batch_size: int,
                  num_workers: int,
                  max_duration_seconds: int,
                  target_sample_rate: int,
                  num_samples: str,
                  seed: int,
                  overwrite: bool):
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, f"{split}_embeddings.npy")
    lab_path = os.path.join(out_dir, f"{split}_labels.npy")

    if (not overwrite) and os.path.exists(emb_path) and os.path.exists(lab_path):
        print(f"[SKIP] {dataset_name}:{split} exists -> {emb_path}")
        return

    ds = build_dataset(
        dataset=dataset_name,
        root_dir=root_dir,
        protocol_file=protocol_file,
        max_duration_seconds=max_duration_seconds,
        target_sample_rate=target_sample_rate,
        num_samples=num_samples,
        seed=seed,
    )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
    )

    Xs, ys = [], []
    backbone.eval()

    for waveforms, labels, *_ in tqdm(loader, desc=f"[{dataset_name}] {split}"):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        attn = (waveforms != 0.0).long()
        z = backbone(waveforms, attn)  # (B, D)
        Xs.append(z.cpu().numpy())
        ys.append(labels.cpu().numpy())

    if len(Xs) == 0:
        raise RuntimeError(f"No samples extracted for {dataset_name}:{split}. Check root/protocol.")

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).reshape(-1).astype(np.int64, copy=False)

    np.save(emb_path, X)
    np.save(lab_path, y)

    uniq, cnt = np.unique(y, return_counts=True)
    hist = {int(u): int(c) for u, c in zip(uniq, cnt)}
    print(f"[OK] Saved {dataset_name}:{split} -> X={X.shape}, y={y.shape}, hist={hist}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["asv5", "asv19", "mlaad"])
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--splits", default="train,dev",
                    help="Comma-separated: train,dev,eval (only extracts splits you provide roots+protocols for).")
    ap.add_argument("--allow_missing_splits", action="store_true",
                    help="If a split root/protocol is missing, skip instead of error.")

    # Per-split IO
    ap.add_argument("--train_root", default="")
    ap.add_argument("--train_protocol", default="")
    ap.add_argument("--dev_root", default="")
    ap.add_argument("--dev_protocol", default="")
    ap.add_argument("--eval_root", default="")
    ap.add_argument("--eval_protocol", default="")

    # Extraction params
    ap.add_argument("--max_duration_seconds", type=int, default=10)
    ap.add_argument("--target_sample_rate", type=int, default=16000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_samples", default="none")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    backbone = Stage1Backbone(
        model_name=args.model_name,
        ckpt_path=args.stage1_ckpt,
        device=device,
    )

    split_specs: Dict[str, Tuple[str, str]] = {
        "train": (args.train_root, args.train_protocol),
        "dev":  (args.dev_root, args.dev_protocol),
        "eval": (args.eval_root, args.eval_protocol),
    }

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    for split in splits:
        if split not in split_specs:
            raise ValueError(f"Unknown split '{split}' in --splits")

        root, prot = split_specs[split]
        if (not root) or (not prot):
            msg = f"[WARN] Missing {split} root/protocol for dataset={args.dataset}"
            if args.allow_missing_splits:
                print(msg + " -> skipping")
                continue
            raise RuntimeError(msg + " (set --allow_missing_splits to skip)")

        extract_split(
            backbone=backbone,
            dataset_name=args.dataset,
            split=split,
            root_dir=root,
            protocol_file=prot,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
            num_samples=args.num_samples,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
