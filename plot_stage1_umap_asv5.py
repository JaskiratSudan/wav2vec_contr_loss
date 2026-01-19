# plot_stage1_umap_asv5.py

import os
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import umap
import plotly.express as px

from data_loader import (
    ASVspoof5EvalTrack1Dataset,
    ASVspoof2019Dataset,
    InTheWildDataset,
    pad_collate_fn_speaker_source_multiclass,
    pad_collate_fn_speaker_source,
)
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule

# =========================
#         CONFIG
# =========================

# ASVspoof5 eval set paths (track-1)
EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac"
EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv"

# ASVspoof19 eval set paths
ASV19_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
ASV19_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

# In-the-wild eval paths
ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

# Stage-1 checkpoint (from train_stage1_asv5.py)
MODEL_NAME = "facebook/wav2vec2-large-960h"
CKPT_PATH = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/asv5_supcon/facebook__wav2vec2-large-960h/facebook__wav2vec2-large-960h_stage1_head_best.pt"

# Model config (must match Stage-1 training)
INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

# Audio / loader
MAX_DURATION_SECONDS = 5
TARGET_SAMPLE_RATE = 16000
BATCH_SIZE = 64
NUM_WORKERS = 4

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 1337

# Output
PLOTS_DIR = "/home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ASV5"

# Misc
SEED = 1337
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
#       HELPERS
# =========================

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_ckpt_path(ckpt_path: str, run_tag: str) -> str:
    """
    Resolve a checkpoint path:
      1) If ckpt_path exists as a file, use it.
      2) Else, try: <dirname(ckpt_path)>/<run_tag>/<run_tag>_stage1_head_best.pt
      3) Else, if ckpt_path is a directory, try: <ckpt_path>/<run_tag>/<run_tag>_stage1_head_best.pt
    """
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

def collect_embeddings(
    loader: DataLoader,
    encoder: torch.nn.Module,
    head: torch.nn.Module,
    device: torch.device,
):
    all_embs = []
    all_bin_labels = []
    all_attack_ids = []
    all_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 5:
                waveforms, bin_labels, attack_ids, _, sources = batch
            elif len(batch) == 4:
                waveforms, bin_labels, _, sources = batch
                attack_ids = None
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            waveforms = waveforms.to(device)
            bin_labels = bin_labels.to(device).long()
            if attack_ids is not None:
                attack_ids = attack_ids.to(device).long()

            attn_mask = (waveforms != 0.0).long()

            hs_4d = encoder(waveforms, attention_mask=attn_mask)
            seq = head(hs_4d)

            z = seq.mean(dim=-1)
            z = F.normalize(z, p=2, dim=1)

            all_embs.append(z.cpu().numpy())
            all_bin_labels.append(bin_labels.cpu().numpy())
            if attack_ids is not None:
                all_attack_ids.append(attack_ids.cpu().numpy())
            all_names.extend(list(sources))

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} samples...")

    all_embs = np.concatenate(all_embs, axis=0)
    all_bin_labels = np.concatenate(all_bin_labels, axis=0)
    all_attack_ids = np.concatenate(all_attack_ids, axis=0) if all_attack_ids else None
    return all_embs, all_bin_labels, all_attack_ids, all_names

def build_label_names(bin_labels, attack_ids, idx_to_attack, label_mode):
    labels = []
    if label_mode == "attack":
        for b, a in zip(bin_labels, attack_ids):
            if b == 1:
                labels.append("Real")
            else:
                name = idx_to_attack.get(int(a), f"Attack{int(a)}")
                labels.append(name)
    elif label_mode == "binary":
        for b in bin_labels:
            labels.append("Real" if b == 1 else "Fake")
    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")
    return np.array(labels)

def plot_umap(embs, labels, names, plots_dir, title, basename):
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    embs_2d = reducer.fit_transform(embs)

    print("Saving PNG plot...")
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))

    if "Real" in unique_labels:
        mask_real = (labels == "Real")
        plt.scatter(
            embs_2d[mask_real, 0],
            embs_2d[mask_real, 1],
            s=8,
            alpha=0.6,
            c="blue",
            label="Real",
        )

    for lab in unique_labels:
        if lab == "Real":
            continue
        mask = (labels == lab)
        if np.any(mask):
            plt.scatter(
                embs_2d[mask, 0],
                embs_2d[mask, 1],
                s=8,
                alpha=0.6,
                label=lab,
            )

    plt.legend(markerscale=2, fontsize=8)
    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    png_path = os.path.join(plots_dir, f"{basename}.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved PNG: {png_path}")

    print("Saving interactive HTML plot...")
    color_map = {"Real": "blue"}
    fig = px.scatter(
        x=embs_2d[:, 0],
        y=embs_2d[:, 1],
        color=labels,
        hover_name=names,
        title=title,
        labels={"x": "UMAP-1", "y": "UMAP-2", "color": "Class"},
        color_discrete_map=color_map,
    )

    html_path = os.path.join(plots_dir, f"{basename}.html")
    fig.write_html(html_path)
    print(f"Saved HTML: {html_path}")


# =========================
#          MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="HF model id, e.g. facebook/wav2vec2-large-960h")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH,
                        help="Checkpoint file OR base directory containing per-model subfolders.")
    parser.add_argument("--plots_dir", type=str, default=PLOTS_DIR,
                        help="Base directory to save plots; a subfolder per model tag will be created.")
    parser.add_argument("--eval_root", type=str, default=EVAL_ROOT,
                        help="Root dir containing ASVspoof5 eval .flac files.")
    parser.add_argument("--eval_protocol", type=str, default=EVAL_PROTOCOL,
                        help="ASVspoof5 eval track-1 protocol file.")
    parser.add_argument("--asv19_eval_root", type=str, default=ASV19_EVAL_ROOT,
                        help="Root dir containing ASVspoof2019 eval .flac files.")
    parser.add_argument("--asv19_eval_protocol", type=str, default=ASV19_EVAL_PROTOCOL,
                        help="ASVspoof2019 eval protocol file.")
    parser.add_argument("--itw_root", type=str, default=ITW_ROOT,
                        help="Root dir containing in-the-wild audio files.")
    parser.add_argument("--itw_protocol", type=str, default=ITW_PROTOCOL,
                        help="In-the-wild protocol CSV.")
    args = parser.parse_args()

    model_name = args.model_name
    run_tag = model_name.replace("/", "__")
    ckpt_path = resolve_ckpt_path(args.ckpt_path, run_tag)
    plots_dir = os.path.join(args.plots_dir, run_tag)

    set_seed(SEED)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Saving to: {plots_dir}")

    encoder = Wav2Vec2Encoder(
        model_name=model_name,
        freeze_encoder=True,
    ).to(DEVICE)
    encoder.eval()

    head = CompressionModule(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if load_encoder_from_ckpt(encoder, ckpt):
        print("Loaded finetuned encoder weights from checkpoint.")
    state_dict = ckpt.get("compression_state_dict", ckpt)
    head.load_state_dict(state_dict, strict=True)
    head.eval()

    datasets = [
        {
            "name": "asv5_eval",
            "title": f"Stage-1 {run_tag} + Compression UMAP (ASVspoof5 Eval) by Attack Type",
            "basename": "stage1_umap_asv5_eval_by_attack",
            "dataset": ASVspoof5EvalTrack1Dataset(
                root_dir=args.eval_root,
                protocol_file=args.eval_protocol,
                subset="all",
                max_duration_seconds=MAX_DURATION_SECONDS,
                target_sample_rate=TARGET_SAMPLE_RATE,
            ),
            "collate_fn": pad_collate_fn_speaker_source_multiclass,
            "label_mode": "attack",
        },
        {
            "name": "asv19_eval",
            "title": f"Stage-1 {run_tag} + Compression UMAP (ASVspoof19 Eval) by Attack Type",
            "basename": "stage1_umap_asv19_eval_by_attack",
            "dataset": ASVspoof2019Dataset(
                root_dir=args.asv19_eval_root,
                protocol_file=args.asv19_eval_protocol,
                subset="all",
                max_duration_seconds=MAX_DURATION_SECONDS,
                target_sample_rate=TARGET_SAMPLE_RATE,
            ),
            "collate_fn": pad_collate_fn_speaker_source_multiclass,
            "label_mode": "attack",
        },
        {
            "name": "itw_eval",
            "title": f"Stage-1 {run_tag} + Compression UMAP (ITW Eval) by Label",
            "basename": "stage1_umap_itw_eval_by_label",
            "dataset": InTheWildDataset(
                root_dir=args.itw_root,
                protocol_file=args.itw_protocol,
                subset="all",
                max_duration_seconds=MAX_DURATION_SECONDS,
                target_sample_rate=TARGET_SAMPLE_RATE,
            ),
            "collate_fn": pad_collate_fn_speaker_source,
            "label_mode": "binary",
        },
    ]

    for cfg in datasets:
        print(f"Collecting embeddings on {cfg['name']}...")
        loader = DataLoader(
            cfg["dataset"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=cfg["collate_fn"],
        )

        idx_to_attack = {v: k for k, v in cfg["dataset"].attack_to_idx.items()} if cfg["label_mode"] == "attack" else {}
        embs, bin_labels, attack_ids, names = collect_embeddings(loader, encoder, head, DEVICE)
        print(f"Total eval embeddings: {embs.shape[0]} (dim={embs.shape[1]})")
        label_names = build_label_names(bin_labels, attack_ids, idx_to_attack, cfg["label_mode"])
        plot_umap(embs, label_names, names, plots_dir, cfg["title"], cfg["basename"])

    print("Done.")

if __name__ == "__main__":
    main()
