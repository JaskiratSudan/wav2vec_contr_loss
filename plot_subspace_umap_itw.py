# plot_subspace_umap_itw.py
# ------------------------------------------------------------
# Visualize Wav2Vec2 embeddings (pre-CompressionModule) on the
# In-The-Wild (ITW) dataset using UMAP (2D),
# colored by:
#   - "Real" (bonafide) in BLUE
#   - "Spoof" in RED
# ------------------------------------------------------------

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

from data_loader import InTheWildDataset, pad_collate_fn_speaker_source
from encoder import Wav2Vec2Encoder

# =========================
#         CONFIG
# =========================

# ITW paths
ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

# Model config (default; can be overridden via --model_name)
MODEL_NAME = "facebook/wav2vec2-large-960h"
# MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

# Audio / loader
MAX_DURATION_SECONDS = 5
TARGET_SAMPLE_RATE = 16000
BATCH_SIZE = 256
NUM_WORKERS = 4
ITW_NUM_SAMPLES = None  # set to an int to subsample, or None for all

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 1337

# Output (base; a "run_tag" subfolder is created inside)
PLOTS_DIR = f"plots/raw_embeddings/{MODEL_NAME}/ITW"

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


# =========================
#          MAIN
# =========================

def main():
    # Optional CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME,
                        help="HF model id, e.g. facebook/wav2vec2-large-960h")
    parser.add_argument("--plots_dir", type=str, default=PLOTS_DIR,
                        help="Base dir to save plots (a subfolder per model will be created).")
    args = parser.parse_args()

    model_name = args.model_name
    run_tag = model_name.replace("/", "__")
    out_dir = os.path.join(args.plots_dir, run_tag)

    set_seed(SEED)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name}")
    print(f"Saving to: {out_dir}")

    # -------- Dataset & Loader (ITW) --------
    itw_ds = InTheWildDataset(
        root_dir=ITW_ROOT,
        protocol_file=ITW_PROTOCOL,
        subset=None,
        num_samples=ITW_NUM_SAMPLES,
        max_duration_seconds=MAX_DURATION_SECONDS,
        # if your InTheWildDataset supports this:
        # target_sample_rate=TARGET_SAMPLE_RATE,
    )

    itw_loader = DataLoader(
        itw_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source,
    )

    # -------- Encoder (frozen) --------
    encoder = Wav2Vec2Encoder(
        model_name=model_name,
        freeze_encoder=True,
    ).to(DEVICE)
    encoder.eval()

    print("Collecting Wav2Vec2 embeddings on ITW...")

    all_embs = []
    all_bin_labels = []
    all_names = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(itw_loader):
            waveforms, bin_labels, speakers, sources = batch

            waveforms = waveforms.to(DEVICE)           # (B, T)
            bin_labels = bin_labels.to(DEVICE).long()  # (B,)

            attn_mask = (waveforms != 0.0).long()

            # Encoder -> (B, K, F, T)
            hs_4d = encoder(waveforms, attention_mask=attn_mask)

            # Pre-compression embedding:
            # 1) Mean over K layers -> (B, F, T)
            # 2) Mean over time -> (B, F)
            # 3) L2 normalize
            pooled_layers = hs_4d.mean(dim=1)    # (B, F, T)
            z = pooled_layers.mean(dim=-1)       # (B, F)
            z = F.normalize(z, p=2, dim=1)       # (B, F)

            all_embs.append(z.cpu().numpy())
            all_bin_labels.append(bin_labels.cpu().numpy())
            all_names.extend(list(sources))

            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {(batch_idx + 1) * BATCH_SIZE} samples...")

    # Concatenate
    all_embs = np.concatenate(all_embs, axis=0)
    all_bin_labels = np.concatenate(all_bin_labels, axis=0)

    print(f"Total ITW embeddings: {all_embs.shape[0]} (dim={all_embs.shape[1]})")

    # -------- Build class labels: Real / Spoof --------
    class_labels = np.array([
        "Real" if int(b) == 1 else "Spoof"
        for b in all_bin_labels
    ])

    # -------- UMAP to 2D --------
    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        random_state=UMAP_RANDOM_STATE,
    )
    embs_2d = reducer.fit_transform(all_embs)

    # -------- Matplotlib PNG (Real=blue, Spoof=red) --------
    print("Saving PNG plot...")
    plt.figure(figsize=(10, 8))

    mask_real = (class_labels == "Real")
    mask_spoof = (class_labels == "Spoof")

    if np.any(mask_real):
        plt.scatter(
            embs_2d[mask_real, 0],
            embs_2d[mask_real, 1],
            s=8,
            alpha=0.6,
            c="blue",
            label="Real",
        )

    if np.any(mask_spoof):
        plt.scatter(
            embs_2d[mask_spoof, 0],
            embs_2d[mask_spoof, 1],
            s=8,
            alpha=0.6,
            c="red",
            label="Spoof",
        )

    plt.legend(markerscale=2, fontsize=8)
    plt.title(f"Wav2Vec2 UMAP (ITW) — Pre-Compression Real vs Spoof [{run_tag}]")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    png_path = os.path.join(out_dir, "wav2vec_umap_itw_real_vs_spoof.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved PNG: {png_path}")

    # -------- Plotly HTML (hover: file name) --------
    print("Saving interactive HTML plot...")
    color_map = {"Real": "blue", "Spoof": "red"}

    fig = px.scatter(
        x=embs_2d[:, 0],
        y=embs_2d[:, 1],
        color=class_labels,
        hover_name=all_names,
        title=f"Wav2Vec2 UMAP (ITW) — Pre-Compression Real vs Spoof [{run_tag}]",
        labels={"x": "UMAP-1", "y": "UMAP-2", "color": "Class"},
        color_discrete_map=color_map,
    )

    html_path = os.path.join(out_dir, "wav2vec_umap_itw_real_vs_spoof.html")
    fig.write_html(html_path)
    print(f"Saved HTML: {html_path}")

    print("Done.")

if __name__ == "__main__":
    main()
