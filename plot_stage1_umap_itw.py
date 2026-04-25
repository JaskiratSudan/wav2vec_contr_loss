# plot_stage1_umap_itw.py

import os
import argparse
import random

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
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

CKPT_PATH = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost/facebook__wav2vec2-large-960h/stage1_head_best.pt"

if "wav2vec2-xls-r-300m" in MODEL_NAME.lower():
    CKPT_PATH = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost/facebook__wav2vec2-xls-r-300m/facebook__wav2vec2-xls-r-300m_stage1_head_best.pt"


INPUT_DIM = 1024
HIDDEN_DIM = 256
DROPOUT = 0.1

# Audio / loader
MAX_DURATION_SECONDS = 5
TARGET_SAMPLE_RATE = 16000
BATCH_SIZE = 64
NUM_WORKERS = 4
ITW_NUM_SAMPLES = None

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


# =========================
#          MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH)
    parser.add_argument("--plots_dir", type=str, default=PLOTS_DIR)
    parser.add_argument("--exp_name", type=str,
                        default=os.environ.get("EXP_NAME", "unknown"))
    parser.add_argument("--dr_method", type=str, default="umap",
                        choices=["umap", "tsne"])
    # --clean: strips title, axis labels, ticks, legend from the PNG
    # so panels can be composited into a paper figure grid
    parser.add_argument("--clean", action="store_true",
                        help="Save a clean panel (no title/axes/legend) for paper figures.")
    parser.add_argument("--itw_root", type=str, default=ITW_ROOT,
                        help="ITW audio root directory.")
    parser.add_argument("--itw_protocol", type=str, default=ITW_PROTOCOL,
                        help="ITW protocol file.")
    args = parser.parse_args()

    model_name = args.model_name
    run_tag = model_name.replace("/", "__")
    ckpt_path = resolve_ckpt_path(args.ckpt_path, run_tag)
    plots_dir = os.path.join(args.plots_dir, run_tag)
    exp_name = args.exp_name
    dr_method = args.dr_method.lower()
    clean = args.clean

    set_seed(SEED)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Saving to: {plots_dir}")
    print(f"Clean mode: {clean}")

    # -------- Dataset & Loader --------
    itw_ds = InTheWildDataset(
        root_dir=args.itw_root,
        protocol_file=args.itw_protocol,
        subset=None,
        num_samples=ITW_NUM_SAMPLES,
        max_duration_seconds=MAX_DURATION_SECONDS,
    )

    itw_loader = DataLoader(
        itw_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source,
    )

    # -------- Encoder --------
    encoder = Wav2Vec2Encoder(
        model_name=model_name,
        freeze_encoder=True,
    ).to(DEVICE)
    encoder.eval()

    # -------- Head --------
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
            waveforms, bin_labels, speakers, sources = batch
            waveforms = waveforms.to(DEVICE)
            bin_labels = bin_labels.to(DEVICE).long()
            attn_mask = (waveforms != 0.0).long()

            hs_4d = encoder(waveforms, attention_mask=attn_mask)
            seq = head(hs_4d)
            z = seq.mean(dim=-1)
            z = F.normalize(z, p=2, dim=1)

            all_embs.append(z.cpu().numpy())
            all_bin_labels.append(bin_labels.cpu().numpy())
            all_speakers.extend([str(s) for s in speakers])
            all_sources.extend([str(s) for s in sources])

            num_seen += waveforms.size(0)
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {num_seen} samples...")

    all_embs = np.concatenate(all_embs, axis=0)
    all_bin_labels = np.concatenate(all_bin_labels, axis=0)

    print(f"Total ITW embeddings: {all_embs.shape[0]} (dim={all_embs.shape[1]})")

    class_labels = np.array([
        "Real" if int(b) == 1 else "Spoof"
        for b in all_bin_labels
    ])

    # -------- Dimensionality reduction --------
    if dr_method == "umap":
        print("Running UMAP...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE,
        )
        embs_2d = reducer.fit_transform(all_embs)

    elif dr_method == "tsne":
        print("Running t-SNE...")
        reducer = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=UMAP_RANDOM_STATE,
        )
        embs_2d = reducer.fit_transform(all_embs)

    # -------- Matplotlib PNG --------
    print("Saving PNG plot...")

    mask_real  = (class_labels == "Real")
    mask_spoof = (class_labels == "Spoof")

    if clean:
        # ── Clean panel for paper figure grid ──────────────────────────────
        # Square figure, no padding, no decorations.
        # All labelling (row/col headers, shared legend) is handled
        # by the compositing script.
        fig, ax = plt.subplots(figsize=(4, 4))

        if np.any(mask_real):
            ax.scatter(embs_2d[mask_real,  0], embs_2d[mask_real,  1],
                       s=2, alpha=0.5, c="royalblue", rasterized=True, label="Real")
        if np.any(mask_spoof):
            ax.scatter(embs_2d[mask_spoof, 0], embs_2d[mask_spoof, 1],
                       s=2, alpha=0.5, c="crimson",   rasterized=True, label="Spoof")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")

        # Thin border only — helps visually separate panels in the grid
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#aaaaaa")

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        png_path = os.path.join(plots_dir, "stage1_umap_itw_real_vs_spoof_clean.png")

    else:
        # ── Standard annotated plot (unchanged from original) ──────────────
        fig, ax = plt.subplots(figsize=(10, 8))

        if np.any(mask_real):
            ax.scatter(embs_2d[mask_real,  0], embs_2d[mask_real,  1],
                       s=8, alpha=0.6, c="blue",  label="Real")
        if np.any(mask_spoof):
            ax.scatter(embs_2d[mask_spoof, 0], embs_2d[mask_spoof, 1],
                       s=8, alpha=0.6, c="red",   label="Spoof")

        ax.legend(markerscale=2, fontsize=8)
        ax.set_title(f"{dr_method.upper()}")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        plt.tight_layout()
        png_path = os.path.join(plots_dir, "stage1_umap_itw_real_vs_spoof.png")

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PNG: {png_path}")

    # -------- Plotly HTML --------
    print("Saving interactive HTML plot...")

    df = pd.DataFrame({
        "UMAP-1": embs_2d[:, 0],
        "UMAP-2": embs_2d[:, 1],
        "Class":  class_labels,
        "Speaker": all_speakers,
        "Source":  all_sources,
    })

    color_map = {"Real": "blue", "Spoof": "red"}

    fig = px.scatter(
        df,
        x="UMAP-1", y="UMAP-2",
        color="Class",
        hover_data=["Speaker", "Source"],
        title=f"Trained {dr_method.upper()} ITW EXP: {exp_name}",
        color_discrete_map=color_map,
    )

    html_path = os.path.join(plots_dir, "stage1_umap_itw_real_vs_spoof.html")
    fig.write_html(html_path)
    print(f"Saved HTML: {html_path}")

    print("Done.")


if __name__ == "__main__":
    main()