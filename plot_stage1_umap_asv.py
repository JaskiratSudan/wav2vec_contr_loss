# plot_stage1_umap_asv.py

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

from data_loader import ASVspoof2019Dataset, pad_collate_fn_speaker_source_multiclass
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule

# =========================
#         CONFIG
# =========================

# Eval set paths
EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

# Stage-1 checkpoint (from train_stage1.py)
# If you used the new training script that nests by model tag, this can be the *base*
# directory (e.g., ".../checkpoints_stage1/supcon/..."), and we'll resolve below.

MODEL_NAME = "facebook/wav2vec2-large-960h"
# MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

CKPT_PATH = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost/facebook__wav2vec2-large-960h/stage1_head_best.pt"

if "wav2vec2-xls-r-300m" in MODEL_NAME.lower():
    CKPT_PATH = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost/facebook__wav2vec2-xls-r-300m/facebook__wav2vec2-xls-r-300m_stage1_head_best.pt"

# Model config (must match Stage-1 training)
INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

# Audio / loader
MAX_DURATION_SECONDS = 5
TARGET_SAMPLE_RATE = 16000
BATCH_SIZE = 256
NUM_WORKERS = 4

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 1337

# Output
PLOTS_DIR = "/home/jsudan/wav2vec_contr_loss/plots/dep_embeddings/ASV"

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

    # Try sibling run_tag folder next to specified file path
    base_dir = os.path.dirname(ckpt_path)
    if base_dir:
        alt = os.path.join(base_dir, run_tag, f"{run_tag}_stage1_head_best.pt")
        if os.path.isfile(alt):
            return alt

    # If provided path is actually a directory, try inside it
    if os.path.isdir(ckpt_path):
        alt = os.path.join(ckpt_path, run_tag, f"{run_tag}_stage1_head_best.pt")
        if os.path.isfile(alt):
            return alt

    # Give a helpful error with the attempted alternatives
    tried = [ckpt_path]
    if base_dir:
        tried.append(os.path.join(base_dir, run_tag, f"{run_tag}_stage1_head_best.pt"))
    if os.path.isdir(ckpt_path):
        tried.append(os.path.join(ckpt_path, run_tag, f"{run_tag}_stage1_head_best.pt"))
    raise FileNotFoundError(f"Checkpoint not found. Tried: {tried}")


# =========================
#          MAIN
# =========================

def main():
    # Parse optional CLI, defaulting to the constants above
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="HF model id, e.g. facebook/wav2vec2-large-960h")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH,
                        help="Path to checkpoint file OR base directory containing per-model subfolders.")
    parser.add_argument("--plots_dir", type=str, default=PLOTS_DIR,
                        help="Base directory to save plots; a subfolder per model tag will be created.")
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

    # -------- Dataset & Loader (eval) --------
    eval_ds = ASVspoof2019Dataset(
        root_dir=EVAL_ROOT,
        protocol_file=EVAL_PROTOCOL,
        subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )

    # Reverse map from index -> attack_id key
    idx_to_attack = {v: k for k, v in eval_ds.attack_to_idx.items()}

    eval_loader = DataLoader(
        eval_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
    )

    # -------- Encoder (frozen) --------
    encoder = Wav2Vec2Encoder(
        model_name=model_name,
        freeze_encoder=True,
    ).to(DEVICE)
    encoder.eval()

    # -------- Compression head --------
    head = CompressionModule(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout_rate=DROPOUT,
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    state_dict = ckpt.get("compression_state_dict", ckpt)
    head.load_state_dict(state_dict, strict=True)
    head.eval()

    print("Collecting embeddings on eval set...")

    all_embs = []
    all_bin_labels = []
    all_attack_ids = []
    all_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            # batch: (waveforms, bin_labels, multi_labels, speakers, sources)
            waveforms, bin_labels, attack_ids, _, sources = batch

            waveforms = waveforms.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE).long()
            attack_ids = attack_ids.to(DEVICE).long()

            attn_mask = (waveforms != 0.0).long()

            # Encoder -> (B, K, F, T)
            hs_4d = encoder(waveforms, attention_mask=attn_mask)

            # Head -> (B, H, T)
            seq = head(hs_4d)

            # Mean over time -> (B, H); L2 normalize
            z = seq.mean(dim=-1)
            z = F.normalize(z, p=2, dim=1)

            all_embs.append(z.cpu().numpy())
            all_bin_labels.append(bin_labels.cpu().numpy())
            all_attack_ids.append(attack_ids.cpu().numpy())
            all_names.extend(list(sources))

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} samples...")

    # Concatenate
    all_embs = np.concatenate(all_embs, axis=0)
    all_bin_labels = np.concatenate(all_bin_labels, axis=0)
    all_attack_ids = np.concatenate(all_attack_ids, axis=0)

    print(f"Total eval embeddings: {all_embs.shape[0]} (dim={all_embs.shape[1]})")

    # -------- Build attack-label strings --------
    attack_labels = []
    for b, a in zip(all_bin_labels, all_attack_ids):
        if b == 1:
            attack_labels.append("Real")
        else:
            name = idx_to_attack.get(int(a), f"Attack{int(a)}")
            attack_labels.append(name)
    attack_labels = np.array(attack_labels)

    # -------- UMAP to 2D --------
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    embs_2d = reducer.fit_transform(all_embs)

    # -------- Matplotlib PNG (Real=blue, attacks colored separately) --------
    print("Saving PNG plot...")
    plt.figure(figsize=(10, 8))

    unique_labels = sorted(set(attack_labels))

    # Plot Real first in blue
    if "Real" in unique_labels:
        mask_real = (attack_labels == "Real")
        plt.scatter(
            embs_2d[mask_real, 0],
            embs_2d[mask_real, 1],
            s=8,
            alpha=0.6,
            c="blue",
            label="Real",
        )

    # Plot each attack separately (default colors)
    for lab in unique_labels:
        if lab == "Real":
            continue
        mask = (attack_labels == lab)
        if np.any(mask):
            plt.scatter(
                embs_2d[mask, 0],
                embs_2d[mask, 1],
                s=8,
                alpha=0.6,
                label=lab,
            )

    plt.legend(markerscale=2, fontsize=8)
    plt.title(f"Stage-1 {run_tag} + Compression UMAP (Eval) by Attack Type")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    png_path = os.path.join(plots_dir, "stage1_umap_eval_by_attack.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved PNG: {png_path}")

    # -------- Plotly HTML (Real=blue, attacks auto-colored) --------
    print("Saving interactive HTML plot...")
    color_map = {"Real": "blue"}

    fig = px.scatter(
        x=embs_2d[:, 0],
        y=embs_2d[:, 1],
        color=attack_labels,
        hover_name=all_names,
        title=f"Stage-1 {run_tag} + Compression UMAP (Eval) by Attack Type",
        labels={"x": "UMAP-1", "y": "UMAP-2", "color": "Class"},
        color_discrete_map=color_map,
    )

    html_path = os.path.join(plots_dir, "stage1_umap_eval_by_attack.html")
    fig.write_html(html_path)
    print(f"Saved HTML: {html_path}")

    print("Done.")

if __name__ == "__main__":
    main()
