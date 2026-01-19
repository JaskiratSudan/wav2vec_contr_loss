# eval_fakexpose_score_file.py
# --------------------------------------------------------
# End-to-end scoring for Fakexpose dataset:
#   Stage-1 (encoder + compression) -> embeddings
#   Stage-2 (binary head) -> logits -> score file
#
# Output line format:
#   <utt_id> <source> <key> <score>
# --------------------------------------------------------

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import FakeXposeDataset
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from evaluation import calculate_EER, compute_eer
import numpy as np


def safe_load(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=True)


class Stage1Backbone(nn.Module):
    def __init__(self, ckpt_path: str, model_name: str, device: torch.device):
        super().__init__()

        self.encoder = Wav2Vec2Encoder(model_name=model_name, freeze_encoder=True).to(device)

        ckpt = safe_load(ckpt_path, map_location=device)
        cfg = ckpt.get("config", {})
        input_dim = cfg.get("INPUT_DIM", 1024)
        hidden_dim = cfg.get("HIDDEN_DIM", 256)
        dropout = cfg.get("DROPOUT", 0.1)

        if "encoder_state_dict" in ckpt:
            load_state_dict_flexible(self.encoder, ckpt["encoder_state_dict"])

        self.head = CompressionModule(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout,
        ).to(device)

        load_state_dict_flexible(self.head, ckpt["compression_state_dict"])

        self.encoder.eval()
        self.head.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, waveforms: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hs_4d = self.encoder(waveforms, attention_mask=attention_mask)
        seq = self.head(hs_4d)
        z = seq.mean(dim=-1)
        z = F.normalize(z, p=2, dim=1)
        return z


class LinearBinaryHead(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class SmallMLPBinaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_stage2_head(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = safe_load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    head_type = cfg.get("HEAD_TYPE", "linear")
    in_dim = cfg.get("IN_DIM", 256)
    hidden_dim = cfg.get("HIDDEN_DIM", 128)
    dropout = cfg.get("DROPOUT", 0.2)

    if head_type == "linear":
        clf = LinearBinaryHead(in_dim=in_dim).to(device)
    elif head_type == "mlp":
        clf = SmallMLPBinaryHead(in_dim=in_dim, hidden=hidden_dim, dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown HEAD_TYPE in Stage-2 ckpt: {head_type}")

    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()
    for p in clf.parameters():
        p.requires_grad = False

    print(
        f"Loaded Stage-2 head: type={head_type}, in_dim={in_dim}, "
        f"hidden_dim={hidden_dim}, dropout={dropout}"
    )
    return clf


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
def score_and_write(
    stage1: nn.Module,
    stage2: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
    agg: str = "mean",
    aggregate_by_utt: bool = False,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stage1.eval()
    stage2.eval()

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
            embs = stage1(waveforms, attn)
            logits = stage2(embs)
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
    ap.add_argument("--root_dir", type=str, required=True)
    ap.add_argument("--stage1_ckpt", type=str, required=True)
    ap.add_argument("--stage2_ckpt", type=str, required=True)
    ap.add_argument("--score_path", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--subset", type=str, default="all", choices=["all", "bonafide", "spoof"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_duration_seconds", type=int, default=5)
    ap.add_argument("--target_sample_rate", type=int, default=16000)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--print_eer", action="store_true")
    ap.add_argument("--chunk_agg", type=str, default="mean", choices=["mean", "median", "max"])
    ap.add_argument("--repeat_short", type=int, default=1, choices=[0, 1])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

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
    labels = [label for _, label, _ in ds.items]
    num_real = sum(1 for lbl in labels if lbl == 1)
    num_fake = sum(1 for lbl in labels if lbl == 0)
    print(f"Fakexpose samples: real={num_real}, fake={num_fake}")
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_fakexpose,
    )

    stage1 = Stage1Backbone(args.stage1_ckpt, model_name=args.model_name, device=device)
    stage2 = load_stage2_head(args.stage2_ckpt, device=device)

    score_and_write(
        stage1,
        stage2,
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
