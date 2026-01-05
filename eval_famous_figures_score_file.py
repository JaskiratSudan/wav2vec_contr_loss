# eval_famous_figures_score_file.py
# --------------------------------------------------------
# End-to-end scoring for FamousFigures dataset:
#   Stage-1 (encoder + compression) -> embeddings
#   Stage-2 (binary head) -> logits -> score file
#
# Output line format:
#   <utt_id> <source> <key> <score>
# where utt_id is the audio file name and score = logits
# --------------------------------------------------------

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loader import FamousFiguresDataset
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from evaluation import calculate_EER


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


def pad_collate_fn_famous(batch):
    waveforms, labels, speakers, sources, utt_ids = zip(*batch)
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        list(waveforms), batch_first=True, padding_value=0.0
    )
    labels = torch.stack(list(labels))
    return padded_waveforms, labels, speakers, sources, utt_ids


@torch.no_grad()
def score_and_write(
    stage1: nn.Module,
    stage2: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: str,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stage1.eval()
    stage2.eval()

    with open(out_path, "w") as f:
        for batch in loader:
            waveforms = batch[0].to(device)
            labels = batch[1].to(device)
            sources = batch[3]
            utt_ids = batch[4]

            attn = (waveforms != 0.0).long()
            embs = stage1(waveforms, attn)
            logits = stage2(embs)
            scores = logits.detach().cpu().numpy()

            labels_np = labels.detach().cpu().numpy().astype(int)
            for i in range(len(scores)):
                utt_id = str(utt_ids[i])
                source = str(sources[i])
                key = "bonafide" if labels_np[i] == 1 else "spoof"
                f.write(f"{utt_id} {source} {key} {scores[i]:.6f}\n")

    print(f"[OK] Wrote: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol_file", type=str, required=True)
    ap.add_argument("--root_dir", type=str, default="")
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
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = FamousFiguresDataset(
        protocol_file=args.protocol_file,
        root_dir=args.root_dir,
        subset=args.subset,
        num_samples=args.num_samples,
        max_duration_seconds=args.max_duration_seconds,
        target_sample_rate=args.target_sample_rate,
        return_audio_name=True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=pad_collate_fn_famous,
    )

    stage1 = Stage1Backbone(args.stage1_ckpt, model_name=args.model_name, device=device)
    stage2 = load_stage2_head(args.stage2_ckpt, device=device)

    score_and_write(stage1, stage2, loader, device, args.score_path)

    if args.print_eer:
        eer = calculate_EER(args.score_path)
        print(f"EER: {eer}")


if __name__ == "__main__":
    main()
