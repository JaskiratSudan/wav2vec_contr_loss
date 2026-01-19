#!/usr/bin/env python3
import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import (
    ASVspoof2019Dataset,
    ASVspoof5Dataset,
    InTheWildDataset,
    FamousFiguresDataset,
    FakeXposeDataset,
    MLAADMailabsDataset,
    pad_collate_fn,
)
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule


def set_seed(seed: int = 1337):
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


class Stage1Backbone(torch.nn.Module):
    def __init__(self, model_name: str, ckpt_path: str, device: torch.device):
        super().__init__()
        self.encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        input_dim = cfg.get("INPUT_DIM", 1024)
        hidden_dim = cfg.get("HIDDEN_DIM", 256)
        dropout = cfg.get("DROPOUT", 0.1)
        if "encoder_state_dict" in ckpt:
            load_state_dict_flexible(self.encoder, ckpt["encoder_state_dict"])
            print("[OK] Loaded finetuned encoder weights from checkpoint.")

        self.head = CompressionModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
        ).to(device)
        load_state_dict_flexible(self.head, ckpt["compression_state_dict"])

        self.encoder.eval()
        self.head.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs_4d = self.encoder(waveforms, attention_mask=attention_mask)
        seq = self.head(hs_4d)
        z = seq.mean(dim=-1)
        z = F.normalize(z, p=2, dim=1)
        return z


def _save_embeddings(ds, out_dir: str, split_name: str, backbone: Stage1Backbone,
                     device: torch.device, batch_size: int, num_workers: int):
    os.makedirs(out_dir, exist_ok=True)
    emb_path = os.path.join(out_dir, f"{split_name}_embeddings.npy")
    lab_path = os.path.join(out_dir, f"{split_name}_labels.npy")

    if os.path.exists(emb_path) and os.path.exists(lab_path):
        print(f"[SKIP] {split_name} embeddings already exist: {emb_path}")
        return

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn,
    )

    all_embs = []
    all_labels = []
    for waveforms, labels in loader:
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        attn_mask = (waveforms != 0.0).long()
        z = backbone(waveforms, attn_mask)
        all_embs.append(z.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if not all_embs:
        print(f"[WARN] No samples for {split_name}.")
        return

    embs = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    np.save(emb_path, embs)
    np.save(lab_path, labels)
    print(f"[OK] Saved {split_name}: embeddings {embs.shape}, labels {labels.shape}")
    print(f"     -> {emb_path}")
    print(f"     -> {lab_path}")


def _split_dataset(ds, seed: int, train_frac: float = 0.8):
    rng = random.Random(seed)
    idx = list(range(len(ds.data)))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_frac)
    train_idx = idx[:n_train]
    dev_idx = idx[n_train:]

    train_ds = ds
    dev_ds = MLAADMailabsDataset.__new__(MLAADMailabsDataset)
    dev_ds.__dict__ = train_ds.__dict__.copy()
    train_ds.data = [ds.data[i] for i in train_idx]
    dev_ds.data = [ds.data[i] for i in dev_idx]
    return train_ds, dev_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--max_duration_seconds", type=int, default=5)

    parser.add_argument("--asv19_train_root", type=str)
    parser.add_argument("--asv19_train_protocol", type=str)
    parser.add_argument("--asv19_dev_root", type=str)
    parser.add_argument("--asv19_dev_protocol", type=str)
    parser.add_argument("--asv19_eval_root", type=str)
    parser.add_argument("--asv19_eval_protocol", type=str)
    parser.add_argument("--asv19_out_dir", type=str)

    parser.add_argument("--asv5_train_root", type=str)
    parser.add_argument("--asv5_train_protocol", type=str)
    parser.add_argument("--asv5_dev_root", type=str)
    parser.add_argument("--asv5_dev_protocol", type=str)
    parser.add_argument("--asv5_eval_root", type=str)
    parser.add_argument("--asv5_eval_protocol", type=str)
    parser.add_argument("--asv5_out_dir", type=str)

    parser.add_argument("--itw_root", type=str)
    parser.add_argument("--itw_protocol", type=str)
    parser.add_argument("--itw_out_dir", type=str)

    parser.add_argument("--ff_protocol", type=str)
    parser.add_argument("--ff_root", type=str, default="")
    parser.add_argument("--ff_out_dir", type=str)

    parser.add_argument("--fakexpose_root", type=str)
    parser.add_argument("--fakexpose_out_dir", type=str)

    parser.add_argument("--mlaad_root", type=str)
    parser.add_argument("--mlaad_protocol", type=str)
    parser.add_argument("--mlaad_out_dir", type=str)
    parser.add_argument("--mlaad_split_seed", type=int, default=2027)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    backbone = Stage1Backbone(args.model_name, args.stage1_ckpt, device=device)

    # ASV19
    if args.asv19_out_dir:
        ds = ASVspoof2019Dataset(
            root_dir=args.asv19_train_root,
            protocol_file=args.asv19_train_protocol,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.asv19_out_dir, "train", backbone, device, args.batch_size, args.num_workers)

        ds = ASVspoof2019Dataset(
            root_dir=args.asv19_dev_root,
            protocol_file=args.asv19_dev_protocol,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.asv19_out_dir, "dev", backbone, device, args.batch_size, args.num_workers)

        ds = ASVspoof2019Dataset(
            root_dir=args.asv19_eval_root,
            protocol_file=args.asv19_eval_protocol,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.asv19_out_dir, "eval", backbone, device, args.batch_size, args.num_workers)

    # ASV5
    if args.asv5_out_dir:
        ds = ASVspoof5Dataset(
            root_dir=args.asv5_train_root,
            protocol_file=args.asv5_train_protocol,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.asv5_out_dir, "train", backbone, device, args.batch_size, args.num_workers)

        ds = ASVspoof5Dataset(
            root_dir=args.asv5_dev_root,
            protocol_file=args.asv5_dev_protocol,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.asv5_out_dir, "dev", backbone, device, args.batch_size, args.num_workers)

        ds = ASVspoof5Dataset(
            root_dir=args.asv5_eval_root,
            protocol_file=args.asv5_eval_protocol,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.asv5_out_dir, "eval", backbone, device, args.batch_size, args.num_workers)

    # ITW
    if args.itw_out_dir:
        ds = InTheWildDataset(
            root_dir=args.itw_root,
            protocol_file=args.itw_protocol,
            subset=None,
            max_duration_seconds=args.max_duration_seconds,
        )
        _save_embeddings(ds, args.itw_out_dir, "itw", backbone, device, args.batch_size, args.num_workers)

    # Famous Figures
    if args.ff_out_dir:
        ds = FamousFiguresDataset(
            protocol_file=args.ff_protocol,
            root_dir=args.ff_root,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.ff_out_dir, "ff", backbone, device, args.batch_size, args.num_workers)

    # FakeXpose
    if args.fakexpose_out_dir:
        ds = FakeXposeDataset(
            root_dir=args.fakexpose_root,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        _save_embeddings(ds, args.fakexpose_out_dir, "fakexpose", backbone, device, args.batch_size, args.num_workers)

    # MLAAD/MAILabs
    if args.mlaad_out_dir:
        full_ds = MLAADMailabsDataset(
            protocol_file=args.mlaad_protocol,
            root_dir=args.mlaad_root,
            subset="all",
            max_duration_seconds=args.max_duration_seconds,
            target_sample_rate=args.target_sample_rate,
        )
        train_ds, dev_ds = _split_dataset(full_ds, seed=args.mlaad_split_seed, train_frac=0.8)
        _save_embeddings(train_ds, args.mlaad_out_dir, "train", backbone, device, args.batch_size, args.num_workers)
        _save_embeddings(dev_ds, args.mlaad_out_dir, "dev", backbone, device, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
