# plot_stage1_umap_asv5_simple.py
import os
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import umap

from data_loader import ASVspoof5Dataset, pad_collate_fn_speaker_source_multiclass
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule


def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_ckpt_path(ckpt_path: str, run_tag: str) -> str:
    if os.path.isfile(ckpt_path):
        return ckpt_path
    base_dir = os.path.dirname(ckpt_path)
    if base_dir:
        alt = os.path.join(base_dir, run_tag, f"{run_tag}_stage1_head_best.pt")
        if os.path.isfile(alt):
            return alt
    if os.path.isdir(ckpt_path):
        alt = os.path.join(ckpt_path, run_tag, f"{run_tag}_stage1_head_best.pt")
        if os.path.isfile(alt):
            return alt
    tried = [ckpt_path]
    if base_dir:
        tried.append(os.path.join(base_dir, run_tag, f"{run_tag}_stage1_head_best.pt"))
    if os.path.isdir(ckpt_path):
        tried.append(os.path.join(ckpt_path, run_tag, f"{run_tag}_stage1_head_best.pt"))
    raise FileNotFoundError(f"Checkpoint not found. Tried: {tried}")


def load_encoder_from_ckpt(encoder: torch.nn.Module, ckpt: dict) -> bool:
    if "encoder_state_dict" not in ckpt:
        return False
    state_dict = ckpt["encoder_state_dict"]
    try:
        encoder.load_state_dict(state_dict, strict=True)
        return True
    except RuntimeError:
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        encoder.load_state_dict(cleaned, strict=True)
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--plots_dir", type=str, required=True)
    parser.add_argument("--eval_root", type=str, required=True)
    parser.add_argument("--eval_protocol", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_duration_seconds", type=int, default=5)
    parser.add_argument("--target_sample_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    run_tag = args.model_name.replace("/", "__")
    ckpt_path = resolve_ckpt_path(args.ckpt_path, run_tag)
    plots_dir = os.path.join(args.plots_dir, run_tag)
    os.makedirs(plots_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Saving to: {plots_dir}")

    eval_ds = ASVspoof5Dataset(
        root_dir=args.eval_root,
        protocol_file=args.eval_protocol,
        subset="all",
        max_duration_seconds=args.max_duration_seconds,
        target_sample_rate=args.target_sample_rate,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
    )

    encoder = Wav2Vec2Encoder(model_name=args.model_name, freeze_encoder=True).to(device)
    encoder.eval()
    head = CompressionModule(input_dim=1024, hidden_dim=256, dropout_rate=0.1).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    if load_encoder_from_ckpt(encoder, ckpt):
        print("Loaded finetuned encoder weights from checkpoint.")
    state_dict = ckpt.get("compression_state_dict", ckpt)
    head.load_state_dict(state_dict, strict=True)
    head.eval()

    all_embs = []
    all_labels = []
    for waveforms, labels, *_ in eval_loader:
        waveforms = waveforms.to(device)
        attn_mask = (waveforms != 0.0).long()
        hs_4d = encoder(waveforms, attention_mask=attn_mask)
        seq = head(hs_4d)
        z = F.normalize(seq.mean(dim=-1), p=2, dim=1)
        all_embs.append(z.detach().cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    embs = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed)
    emb2d = reducer.fit_transform(embs)

    plt.figure(figsize=(8, 6))
    colors = np.where(labels == 1, "tab:blue", "tab:red")
    plt.scatter(emb2d[:, 0], emb2d[:, 1], c=colors, s=6, alpha=0.7)
    plt.title("ASVspoof5 Eval UMAP (bonafide=blue, spoof=red)")
    out_path = os.path.join(plots_dir, "umap_asv5_eval.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved plot -> {out_path}")


if __name__ == "__main__":
    main()
