# plot_stage1_umap_itw.py

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
import pandas as pd

from data_loader import InTheWildDataset, pad_collate_fn_speaker_source
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule

# =========================
#         CONFIG
# =========================

# ITW paths
ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

MODEL_NAME = "facebook/wav2vec2-large-960h"
# MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

CKPT_PATH = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost/facebook__wav2vec2-large-960h/stage1_head_best.pt"

if "wav2vec2-xls-r-300m" in MODEL_NAME.lower():
    CKPT_PATH = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost/facebook__wav2vec2-xls-r-300m/facebook__wav2vec2-xls-r-300m_stage1_head_best.pt"


INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

# Audio / loader
MAX_DURATION_SECONDS = 5
TARGET_SAMPLE_RATE = 16000  # kept for reference
BATCH_SIZE = 256
NUM_WORKERS = 4
ITW_NUM_SAMPLES = None  # e.g., 500 to subsample, or None for all

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 1337

# Output
PLOTS_DIR = "plots/dep_embeddings/ITW"

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
    """
    Load finetuned encoder weights if present in the checkpoint.
    Returns True if weights were loaded.
    """
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


# =========================
#          MAIN
# =========================

def main():
    # Optional CLI overrides
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

    # -------- Dataset & Loader (ITW) --------
    itw_ds = InTheWildDataset(
        root_dir=ITW_ROOT,
        protocol_file=ITW_PROTOCOL,
        subset=None,
        num_samples=ITW_NUM_SAMPLES,
        max_duration_seconds=MAX_DURATION_SECONDS,
        # If your InTheWildDataset supports this, you can pass:
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

    # -------- Compression head --------
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

    print("Collecting embeddings on ITW set...")

    all_embs = []
    all_bin_labels = []
    all_speakers = []
    all_sources = []

    num_seen = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(itw_loader):
            # batch: (waveforms, bin_labels, speakers, sources)
            waveforms, bin_labels, speakers, sources = batch

            waveforms = waveforms.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE).long()

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

            # speakers/sources should already be iterables of length B
            all_speakers.extend([str(s) for s in speakers])
            all_sources.extend([str(s) for s in sources])

            num_seen += waveforms.size(0)
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {num_seen} samples...")

    # Concatenate
    all_embs = np.concatenate(all_embs, axis=0)
    all_bin_labels = np.concatenate(all_bin_labels, axis=0)

    print(f"Total ITW embeddings: {all_embs.shape[0]} (dim={all_embs.shape[1]})")

    # -------- Map binary labels -> string classes --------
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

    # -------- Matplotlib PNG (Real vs Spoof) --------
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
    plt.title(f"Stage-1 {run_tag} + Compression UMAP (ITW) – Real vs Spoof")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.tight_layout()

    png_path = os.path.join(plots_dir, "stage1_umap_itw_real_vs_spoof.png")
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved PNG: {png_path}")

    # -------- Plotly HTML (hover: speaker + source) --------
    print("Saving interactive HTML plot...")

    df = pd.DataFrame({
        "UMAP-1": embs_2d[:, 0],
        "UMAP-2": embs_2d[:, 1],
        "Class": class_labels,
        "Speaker": all_speakers,
        "Source": all_sources,
    })

    color_map = {"Real": "blue", "Spoof": "red"}

    fig = px.scatter(
        df,
        x="UMAP-1",
        y="UMAP-2",
        color="Class",
        hover_data=["Speaker", "Source"],
        title=f"Stage-1 {run_tag} + Compression UMAP (ITW) – Real vs Spoof",
        labels={"UMAP-1": "UMAP-1", "UMAP-2": "UMAP-2", "Class": "Class"},
        color_discrete_map=color_map,
    )

    html_path = os.path.join(plots_dir, "stage1_umap_itw_real_vs_spoof.html")
    fig.write_html(html_path)
    print(f"Saved HTML: {html_path}")

    print("Done.")


if __name__ == "__main__":
    main()
