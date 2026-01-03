# eval_baseline_score_file.py
# --------------------------------------------------------
# Generate ASVspoof-style CM score files using the *current baseline model*:
# XLSR (frozen) + CompressionModule + Linear classifier head (logits)
#
# Output line format:
#   <utt_id> <source> <key> <score>
# where score = logits (higher => more bonafide-like, assuming label 1=bonafide)
# --------------------------------------------------------

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_loader import (
    ASVspoof2019Dataset,
    InTheWildDataset,
    pad_collate_fn_speaker_source_multiclass,
    pad_collate_fn_speaker_source,
)
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule

# ------------------ Paths (edit if needed) ------------------
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

ASV_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
ASV_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

MAX_DURATION_SECONDS = 5
TARGET_SAMPLE_RATE = 16000
BATCH_SIZE = 128
NUM_WORKERS = 4

# Output
ASV_SCORE_PATH = f"scores/baseline/{MODEL_NAME}/score_cm_eval.txt"
ITW_SCORE_PATH = f"scores/baseline/{MODEL_NAME}/score_cm_itw.txt"


def safe_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class End2EndBCEModel(nn.Module):
    """
    encoder: (B,T) -> (B,K,F,T)
    compression: (B,K,F,T) -> (B,H,T)
    mean-pool time: (B,H,T) -> (B,H)
    classifier: (B,H) -> (B,) logits
    """
    def __init__(self, encoder: nn.Module, compression: nn.Module, hidden_dim: int):
        super().__init__()
        self.encoder = encoder
        self.compression = compression
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor):
        # encoder is frozen, so keep it no_grad
        with torch.no_grad():
            hs = self.encoder(waveforms, attention_mask=attention_mask)  # (B,K,F,T)
        seq = self.compression(hs)                                       # (B,H,T)
        emb = seq.mean(dim=-1)                                           # (B,H)
        logits = self.classifier(emb).squeeze(-1)                        # (B,)
        return logits


@torch.no_grad()
def score_loader_and_write(model: nn.Module, loader: DataLoader, device: torch.device, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.eval()

    with open(out_path, "w") as f:
        for batch in tqdm(loader, desc=f"Scoring -> {os.path.basename(out_path)}"):
            # ASV: (wav, bin_label, attack_id, speaker, audio_name)
            # ITW: (wav, label, speaker, audio_name)
            waveforms = batch[0].to(device)
            labels = batch[1].to(device)

            # Use audio_name as utt_id if present (last element)
            utt_ids = batch[-1]  # tuple/list of strings

            attn = (waveforms != 0.0).long()
            logits = model(waveforms, attn)  # (B,)
            scores = logits.detach().cpu().numpy()

            labels_np = labels.detach().cpu().numpy().astype(int)

            for i in range(len(scores)):
                utt_id = str(utt_ids[i])
                source = "NA"
                key = "bonafide" if labels_np[i] == 1 else "spoof"
                f.write(f"{utt_id} {source} {key} {scores[i]:.6f}\n")

    print(f"[OK] Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="baseline checkpoint with model_state_dict", default="wav2vec_contr_loss/checkpoints_baseline/bce/facebook__wav2vec2-xls-r-300m/facebook__wav2vec2-xls-r-300m_baseline_bce_best.pt")
    ap.add_argument("--asv_score_path", type=str, default=ASV_SCORE_PATH, help="Output score file for ASV eval.")
    ap.add_argument("--itw_score_path", type=str, default=ITW_SCORE_PATH, help="Output score file for ITW.")
    ap.add_argument("--run_asv_eval", action="store_true")
    ap.add_argument("--run_itw", action="store_true")
    args = ap.parse_args()

    if not args.run_asv_eval and not args.run_itw:
        print("Nothing to do. Add --run_asv_eval and/or --run_itw")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Build model (must match training)
    encoder = Wav2Vec2Encoder(model_name=MODEL_NAME, freeze_encoder=True).to(device)
    head = CompressionModule(input_dim=1024, hidden_dim=256, dropout_rate=0.1).to(device)
    model = End2EndBCEModel(encoder=encoder, compression=head, hidden_dim=256).to(device)

    ckpt = safe_load(args.ckpt, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)  # allow loading raw state_dict too
    model.load_state_dict(sd, strict=True)
    print("[OK] Loaded baseline checkpoint:", args.ckpt)

    # ---------------- ASV EVAL ----------------
    if args.run_asv_eval:
        ds = ASVspoof2019Dataset(
            root_dir=ASV_EVAL_ROOT,
            protocol_file=ASV_EVAL_PROTOCOL,
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
        score_loader_and_write(model, loader, device, args.asv_score_path)

    # ---------------- ITW ----------------
    if args.run_itw:
        ds = InTheWildDataset(
            root_dir=ITW_ROOT,
            protocol_file=ITW_PROTOCOL,
            subset=None,
            num_samples=None,
            max_duration_seconds=MAX_DURATION_SECONDS,
        )
        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            collate_fn=pad_collate_fn_speaker_source,
        )
        score_loader_and_write(model, loader, device, args.itw_score_path)


if __name__ == "__main__":
    main()
