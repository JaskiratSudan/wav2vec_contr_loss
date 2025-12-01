# extract_stage1_embeddings.py
# --------------------------------------------------------
# Use frozen Stage-1 Wav2Vec2 + CompressionModule to extract
# clip-level embeddings for:
#   - ASVspoof2019 train/dev/eval splits
#   - In-The-Wild (ITW) dataset
#
# Saves:
#   stage1_embeddings/ASV/train_embeddings.npy    (N_train, D)
#   stage1_embeddings/ASV/train_labels.npy        (N_train,)
#   stage1_embeddings/ASV/dev_embeddings.npy      (N_dev, D)
#   stage1_embeddings/ASV/dev_labels.npy          (N_dev,)
#   stage1_embeddings/ASV/eval_embeddings.npy     (N_eval, D)
#   stage1_embeddings/ASV/eval_labels.npy         (N_eval,)
#
#   stage1_embeddings/ITW/itw_embeddings.npy      (N_itw, D)
#   stage1_embeddings/ITW/itw_labels.npy          (N_itw,)
#
# If embeddings already exist for a split/dataset, that part is skipped.
#
# Requires:
#   - encoder.py (Wav2Vec2Encoder)
#   - compression_module.py (CompressionModule)
#   - data_loader.py:
#       ASVspoof2019Dataset, InTheWildDataset
#       pad_collate_fn_speaker_source_multiclass, pad_collate_fn_speaker_source
#   - stage1_head_best.pt from train_stage1.py
# --------------------------------------------------------

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import (
    ASVspoof2019Dataset,
    InTheWildDataset,
    pad_collate_fn_speaker_source_multiclass,
    pad_collate_fn_speaker_source,
)
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule

# ============================================================
#                       CONFIG
# ============================================================

# ----- ASVspoof2019 paths -----
TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"

DEV_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac"
DEV_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt"

EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

# ----- In-The-Wild paths -----
ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"
ITW_NUM_SAMPLES = None  # None = all, or set to an int to subsample

# Stage-1 checkpoint
STAGE1_CKPT = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage1/supcon/with_rawboost/facebook__wav2vec2-xls-r-300m/facebook__wav2vec2-xls-r-300m_stage1_head_best.pt"

# Extraction
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
MAX_DURATION_SECONDS = 5
TARGET_SAMPLE_RATE = 16000  # kept for reference; ITW loader may already be 16k
BATCH_SIZE = 256
NUM_WORKERS = 4
SEED = 1337

# Output dirs
ASV_OUT_DIR = f"stage1_embeddings/ASV/{MODEL_NAME}"
ITW_OUT_DIR = f"stage1_embeddings/ITW/{MODEL_NAME}"

# ============================================================
#                       UTILS
# ============================================================

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Stage1Backbone(torch.nn.Module):
    """
    Wrapper around frozen Wav2Vec2Encoder + CompressionModule
    that outputs a single L2-normalized clip embedding per utterance.
    """
    def __init__(self, ckpt_path: str, device: torch.device):
        super().__init__()

        # Frozen encoder
        self.encoder = Wav2Vec2Encoder(model_name=MODEL_NAME, freeze_encoder=True).to(device)

        # Load Stage-1 head config to match dims
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        input_dim = cfg.get("INPUT_DIM", 1024)
        hidden_dim = cfg.get("HIDDEN_DIM", 256)
        dropout = cfg.get("DROPOUT", 0.1)

        # Compression head
        self.head = CompressionModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
        ).to(device)

        self.head.load_state_dict(ckpt["compression_state_dict"])

        # Freeze everything
        self.encoder.eval()
        self.head.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        waveforms:     (B, T)
        attention_mask:(B, T)
        Returns:
            z:          (B, D) L2-normalized clip embeddings
        """
        # (B, K, F, T)
        hs_4d = self.encoder(waveforms, attention_mask=attention_mask)
        # (B, D, T)
        seq = self.head(hs_4d)
        # mean-pool over time -> (B, D)
        z = seq.mean(dim=-1)
        # L2-normalize
        z = F.normalize(z, p=2, dim=1)
        return z


# ----------------- ASV extraction -----------------

def extract_asv_split(
    root_dir: str,
    protocol_file: str,
    split_name: str,
    backbone: Stage1Backbone,
    device: torch.device,
):
    """
    Extract embeddings for one ASVspoof2019 split and save as .npy.
    Skips if files already exist.
    """
    os.makedirs(ASV_OUT_DIR, exist_ok=True)
    emb_path = os.path.join(ASV_OUT_DIR, f"{split_name}_embeddings.npy")
    lab_path = os.path.join(ASV_OUT_DIR, f"{split_name}_labels.npy")

    if os.path.exists(emb_path) and os.path.exists(lab_path):
        print(f"[SKIP][ASV] Existing {split_name} embeddings found:")
        print(f"       {emb_path}")
        print(f"       {lab_path}")
        return

    print(f"==> Building ASV dataset for split: {split_name}")
    ds = ASVspoof2019Dataset(
        root_dir=root_dir,
        protocol_file=protocol_file,
        subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS,
        target_sample_rate=TARGET_SAMPLE_RATE,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=pad_collate_fn_speaker_source_multiclass,
    )

    all_embs = []
    all_labels = []

    backbone.eval()

    for waveforms, labels, *_ in tqdm(loader, desc=f"[ASV] Extracting {split_name}"):
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        attn_mask = (waveforms != 0.0).long()

        z = backbone(waveforms, attn_mask)  # (B, D)

        all_embs.append(z.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if len(all_embs) == 0:
        print(f"[WARN][ASV] No samples found for split '{split_name}'. Skipping.")
        return

    embs = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    np.save(emb_path, embs)
    np.save(lab_path, labels)

    print(f"[OK][ASV] Saved {split_name}: embeddings {embs.shape}, labels {labels.shape}")
    print(f"     -> {emb_path}")
    print(f"     -> {lab_path}")


# ----------------- ITW extraction -----------------

def extract_itw(backbone: Stage1Backbone, device: torch.device):
    """
    Extract embeddings for the full In-The-Wild dataset and save as:
        stage1_embeddings/ITW/itw_embeddings.npy
        stage1_embeddings/ITW/itw_labels.npy

    Skips if files already exist.
    """
    os.makedirs(ITW_OUT_DIR, exist_ok=True)
    emb_path = os.path.join(ITW_OUT_DIR, "itw_embeddings.npy")
    lab_path = os.path.join(ITW_OUT_DIR, "itw_labels.npy")

    if os.path.exists(emb_path) and os.path.exists(lab_path):
        print(f"[SKIP][ITW] Existing ITW embeddings found:")
        print(f"       {emb_path}")
        print(f"       {lab_path}")
        return

    print("==> Building In-The-Wild dataset...")
    itw_ds = InTheWildDataset(
        root_dir=ITW_ROOT,
        protocol_file=ITW_PROTOCOL,
        subset=None,
        num_samples=ITW_NUM_SAMPLES,
        max_duration_seconds=MAX_DURATION_SECONDS,
        # if your InTheWildDataset supports it:
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

    all_embs = []
    all_labels = []

    backbone.eval()

    for waveforms, bin_labels, speakers, sources in tqdm(itw_loader, desc="[ITW] Extracting"):
        waveforms = waveforms.to(device)
        bin_labels = bin_labels.to(device)

        attn_mask = (waveforms != 0.0).long()

        z = backbone(waveforms, attn_mask)  # (B, D)

        all_embs.append(z.cpu().numpy())
        all_labels.append(bin_labels.cpu().numpy())

    if len(all_embs) == 0:
        print("[WARN][ITW] No samples found in ITW dataset. Skipping.")
        return

    embs = np.concatenate(all_embs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    np.save(emb_path, embs)
    np.save(lab_path, labels)

    print(f"[OK][ITW] Saved ITW embeddings: {embs.shape}, labels {labels.shape}")
    print(f"     -> {emb_path}")
    print(f"     -> {lab_path}")


# ============================================================
#                       MAIN
# ============================================================

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading Stage-1 checkpoint from: {STAGE1_CKPT}")
    backbone = Stage1Backbone(STAGE1_CKPT, device=device)

    # ASVspoof2019 splits
    extract_asv_split(TRAIN_ROOT, TRAIN_PROTOCOL, "train", backbone, device)
    extract_asv_split(DEV_ROOT,   DEV_PROTOCOL,   "dev",   backbone, device)
    extract_asv_split(EVAL_ROOT,  EVAL_PROTOCOL,  "eval",  backbone, device)

    # In-The-Wild
    extract_itw(backbone, device)


if __name__ == "__main__":
    main()
