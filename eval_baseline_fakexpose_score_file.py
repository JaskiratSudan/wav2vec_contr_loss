# eval_baseline_fakexpose_score_file.py
# --------------------------------------------------------
# Generate CM score file for Fakexpose using baseline model:
# XLSR (frozen) + CompressionModule + Linear classifier head (logits)
#
# Output line format:
#   <utt_id> <source> <key> <score>
# --------------------------------------------------------

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from data_loader import FakeXposeDataset
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from evaluation import calculate_EER, compute_eer


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
        with torch.no_grad():
            hs = self.encoder(waveforms, attention_mask=attention_mask)
        seq = self.compression(hs)
        emb = seq.mean(dim=-1)
        logits = self.classifier(emb).squeeze(-1)
        return logits


def pad_collate_fn_fakexpose(batch):
    waveforms, labels, sources, utt_ids = zip(*batch)
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    labels = torch.stack(list(labels))
    return padded_waveforms, labels, sources, utt_ids


def aggregate_scores(scores: np.ndarray, method: str) -> float:
    if method == "mean":
        return float(scores.mean())
    if method == "median":
        return float(np.median(scores))
    if method == "max":
        return float(scores.max())
    raise ValueError(f"Unknown aggregation method: {method}")


@torch.no_grad()
def score_loader_and_write(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    agg: str = "mean",
    aggregate_by_utt: bool = False,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.eval()

    scores_by_utt = {}
    sources_by_utt = {}
    labels_by_utt = {}

    with open(out_path, "w") as f:
        for batch in loader:
            waveforms = batch[0].to(device)
            labels = batch[1].to(device)
            sources = batch[2]
            utt_ids = batch[3]

            attn = (waveforms != 0.0).long()
            logits = model(waveforms, attn)
            scores = logits.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)

            for i in range(len(scores)):
                utt_id = str(utt_ids[i])
                source = str(sources[i])
                label = int(labels_np[i])
                if aggregate_by_utt:
                    scores_by_utt.setdefault(utt_id, []).append(float(scores[i]))
                    sources_by_utt.setdefault(utt_id, source)
                    labels_by_utt.setdefault(utt_id, label)
                else:
                    key = "bonafide" if label == 1 else "spoof"
                    f.write(f"{utt_id} {source} {key} {scores[i]:.6f}\n")

        if aggregate_by_utt:
            for utt_id, score_list in scores_by_utt.items():
                agg_score = aggregate_scores(np.asarray(score_list), agg)
                source = sources_by_utt[utt_id]
                label = labels_by_utt[utt_id]
                key = "bonafide" if label == 1 else "spoof"
                f.write(f"{utt_id} {source} {key} {agg_score:.6f}\n")

    print(f"[OK] Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="baseline checkpoint with model_state_dict")
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--score_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_duration_seconds", type=int, default=5)
    ap.add_argument("--target_sample_rate", type=int, default=16000)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--subset", type=str, default="all", choices=["all", "bonafide", "spoof"])
    ap.add_argument("--chunk_agg", type=str, default="mean", choices=["mean", "median", "max"])
    ap.add_argument("--repeat_short", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    encoder = Wav2Vec2Encoder(model_name=args.model_name, freeze_encoder=True).to(device)
    head = CompressionModule(input_dim=1024, hidden_dim=256, dropout_rate=0.1).to(device)
    model = End2EndBCEModel(encoder=encoder, compression=head, hidden_dim=256).to(device)

    ckpt = safe_load(args.ckpt, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd, strict=True)
    print("[OK] Loaded baseline checkpoint:", args.ckpt)

    repeat_short = bool(args.repeat_short)

    ds = FakeXposeDataset(
        root_dir=args.root_dir,
        subset=args.subset,
        num_samples=args.num_samples,
        max_duration_seconds=args.max_duration_seconds,
        target_sample_rate=args.target_sample_rate,
        return_audio_name=True,
        repeat_short=repeat_short,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_fakexpose,
    )

    score_loader_and_write(
        model,
        loader,
        device,
        args.score_path,
        agg=args.chunk_agg,
        aggregate_by_utt=True,
    )
    eer = calculate_EER(args.score_path)

    cm_data = np.genfromtxt(args.score_path, dtype=str)
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(float)
    labels = (cm_keys == "bonafide").astype(int)
    bonafide_scores = cm_scores[labels == 1]
    spoof_scores = cm_scores[labels == 0]
    eer_frac, eer_thresh = compute_eer(bonafide_scores, spoof_scores)
    preds = (cm_scores >= eer_thresh).astype(int)
    acc = (preds == labels).mean() * 100.0

    print(f"EER: {eer}")
    print(f"EER threshold: {eer_thresh}")
    print(f"Accuracy@EERth: {acc:.2f}%")


if __name__ == "__main__":
    main()
