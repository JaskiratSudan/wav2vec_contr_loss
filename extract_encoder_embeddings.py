import os
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from numpy.lib.format import open_memmap  # <--- Key import for direct disk writing

from data_loader import ASVspoof2019Dataset, pad_collate_fn_speaker_source_multiclass
from encoder import Wav2Vec2Encoder
from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise

# =======================
#        CONFIG
# =======================
SAVE_DIR = "/scratch/hafiz_root/hafiz1/jsudan/encoder_embeddings"

TRAIN_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_train_protocol_with_speaker.txt"
DEV_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_dev/flac"
DEV_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_dev_protocol_with_speaker.txt"

TARGET_SAMPLE_RATE = 16000
MAX_DURATION_SECONDS = 5
BATCH_SIZE = 256
NUM_WORKERS = 4
USE_RAWBOOST = True
RAWBOOST_PROB = 0.9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output Dimensions
FIXED_TIME_DIM = 250  # 5s * 16k / 320 stride
FEAT_DIM = 1024       # Wav2Vec2 Large feature dim

def apply_rawboost_batch(x: torch.Tensor) -> torch.Tensor:
    if not USE_RAWBOOST: return x
    device = x.device
    pad_mask = (x != 0.0)
    a = x.detach().cpu().numpy()
    for i in range(a.shape[0]):
        if random.random() < RAWBOOST_PROB:
            y = LnL_convolutive_noise(a[i], N_f=5, nBands=5, minF=20.0, maxF=8000.0, minBW=100.0, maxBW=1000.0, minCoeff=10, maxCoeff=100, minG=0.0, maxG=0.0, minBiasLinNonLin=5.0, maxBiasLinNonLin=20.0, fs=TARGET_SAMPLE_RATE)
            if random.random() < 0.5: y = SSI_additive_noise(y, SNRmin=10.0, SNRmax=40.0, nBands=5, minF=20.0, maxF=8000.0, minBW=100.0, maxBW=1000.0, minCoeff=10, maxCoeff=100, minG=0.0, maxG=0.0, fs=TARGET_SAMPLE_RATE)
            if random.random() < 0.5: y = ISD_additive_noise(y, P=10.0, g_sd=2.0)
            a[i] = y
    y = torch.from_numpy(a).to(device=device, dtype=x.dtype)
    return y * pad_mask.to(device=y.device, dtype=y.dtype)

def extract_safe(loader, encoder, split_name, apply_aug):
    print(f"[{split_name}] Initializing safe extraction (Augmentation={apply_aug})...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    N = len(loader.dataset)
    emb_path = os.path.join(SAVE_DIR, f"{split_name}_embeddings_mean.npy")
    lab_path = os.path.join(SAVE_DIR, f"{split_name}_labels.npy")
    
    print(f"[{split_name}] Creating memmap files on disk...")
    # 'w+' creates new file. This allocates space on DISK, not RAM.
    # Shape is (N, 1024, 250) because we are taking the mean over layers.
    mm_embs = open_memmap(emb_path, mode='w+', dtype='float32', shape=(N, FEAT_DIM, FIXED_TIME_DIM))
    mm_labels = open_memmap(lab_path, mode='w+', dtype='int64', shape=(N,))

    encoder.eval()
    start_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Extracting {split_name}"):
            waveforms, labels, *_ = batch
            waveforms = waveforms.to(DEVICE)
            
            if apply_aug:
                waveforms = apply_rawboost_batch(waveforms)
            
            attn_mask = (waveforms != 0.0).long()

            # Forward -> (B, K, F, T)
            hs_4d = encoder(waveforms, attention_mask=attn_mask)
            
            # 1. Take Mean over layers (Dim 1) -> (B, F, T)
            hs_mean = hs_4d.mean(dim=1)

            # 2. Fix Time Dimension (Pad/Crop to 250)
            B, F_dim, T_actual = hs_mean.shape
            batch_fixed = torch.zeros((B, F_dim, FIXED_TIME_DIM), dtype=hs_mean.dtype, device=hs_mean.device)
            valid_t = min(T_actual, FIXED_TIME_DIM)
            batch_fixed[:, :, :valid_t] = hs_mean[:, :, :valid_t]

            # 3. Write directly to DISK (Memmap)
            end_idx = start_idx + B
            mm_embs[start_idx:end_idx] = batch_fixed.cpu().numpy()
            mm_labels[start_idx:end_idx] = labels.numpy()
            
            # 4. Flush to ensure data hits the disk
            mm_embs.flush()
            mm_labels.flush()
            
            start_idx = end_idx

    print(f"[{split_name}] COMPLETE.")
    print(f"  -> Saved to {emb_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    args = parser.parse_args()

    print(f"Model: {args.model_name} | Device: {DEVICE}")

    # Datasets
    train_ds = ASVspoof2019Dataset(
        root_dir=TRAIN_ROOT, protocol_file=TRAIN_PROTOCOL, subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS, target_sample_rate=TARGET_SAMPLE_RATE
    )
    dev_ds = ASVspoof2019Dataset(
        root_dir=DEV_ROOT, protocol_file=DEV_PROTOCOL, subset="all",
        max_duration_seconds=MAX_DURATION_SECONDS, target_sample_rate=TARGET_SAMPLE_RATE
    )

    # Loaders
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=pad_collate_fn_speaker_source_multiclass
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, collate_fn=pad_collate_fn_speaker_source_multiclass
    )

    encoder = Wav2Vec2Encoder(model_name=args.model_name, freeze_encoder=True).to(DEVICE)

    # Extract
    extract_safe(train_loader, encoder, "train", apply_aug=USE_RAWBOOST)
    extract_safe(dev_loader, encoder, "dev", apply_aug=False)

if __name__ == "__main__":
    main()