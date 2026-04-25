import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from data_loader import (
    ASVspoof2019Dataset,
    ASVspoof5Dataset,
    ASVspoof2021DFDataset,
    ASVspoof2021LADataset,
    InTheWildDataset,
    FamousFiguresDataset,
    FakeXposeDataset,
    MLAADMailabsDataset,
    DeepfakeEval2024Dataset,
)
from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from evaluation import calculate_EER

DEFAULT_ASV19_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_eval/flac"
DEFAULT_ASV19_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/AsvSpoofData_2019/train/LA/ASVspoof2019_eval_protocol_with_speaker.txt"

DEFAULT_ASV5_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/No_Laundering_eval/flac"
DEFAULT_ASV5_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof5/protocols/ASVspoof5.eval.track_1.tsv"

DEFAULT_ASV21_DF_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/flac"
DEFAULT_ASV21_DF_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/DF/ASVspoof2021_DF_eval/trial_metadata.txt"

DEFAULT_ASV21_LA_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/flac"
DEFAULT_ASV21_LA_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ASVSpoof2021_complete/LA/ASVspoof2021_LA_eval/trial_metadata.txt"

DEFAULT_ITW_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild"
DEFAULT_ITW_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv"

DEFAULT_FF_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/famousfigures/protocol.txt"
DEFAULT_FF_ROOT = ""

print(f"FF_PROTOCOL: {DEFAULT_FF_PROTOCOL}")

DEFAULT_FAKEXPOSE_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/fakexpose"

DEFAULT_MLAAD_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/multilingual"
DEFAULT_MLAAD_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/multilingual/protocol_MLAAD_MAILabs_total_balanced.txt"

DEFAULT_DEEPFAKE_EVAL_ROOT = "/nfs/turbo/umd-hafiz/issf_server_data/Deepfake_Eval_2024/audio-data"
DEFAULT_DEEPFAKE_EVAL_PROTOCOL = "/nfs/turbo/umd-hafiz/issf_server_data/Deepfake_Eval_2024/audio-metadata-publish.csv"


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


class DeepMLPBinaryHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        layers = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_stage2_head(ckpt_path: str, device: torch.device) -> nn.Module:
    ckpt = safe_load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})

    head_type = cfg.get("HEAD_TYPE", "linear")
    in_dim = cfg.get("IN_DIM", 256)
    hidden_dim = cfg.get("HIDDEN_DIM", 128)
    num_layers = cfg.get("NUM_LAYERS", 3)
    dropout = cfg.get("DROPOUT", 0.2)

    if head_type == "linear":
        clf = LinearBinaryHead(in_dim=in_dim).to(device)
    elif head_type == "mlp":
        clf = SmallMLPBinaryHead(in_dim=in_dim, hidden=hidden_dim, dropout=dropout).to(device)
    elif head_type == "deep_mlp":
        clf = DeepMLPBinaryHead(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout).to(device)
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


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        return True, rank, world_size, local_rank
    return False, 0, 1, 0


def pad_collate_fn_generic(batch):
    waveforms = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sources = []
    utt_ids = []
    for item in batch:
        if len(item) >= 4:
            sources.append(item[-2])
            utt_ids.append(item[-1])
        elif len(item) == 3:
            sources.append("NA")
            utt_ids.append(item[2])
        else:
            sources.append("NA")
            utt_ids.append("unknown")
    padded_waveforms = torch.nn.utils.rnn.pad_sequence(
        waveforms, batch_first=True, padding_value=0.0
    )
    labels = torch.stack(labels)
    return padded_waveforms, labels, None, sources, utt_ids


@torch.no_grad()
def _repeat_pad_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(target_len, device=x.device, dtype=x.dtype)
    if x.numel() >= target_len:
        return x[:target_len]
    reps = (target_len + x.numel() - 1) // x.numel()
    return x.repeat(reps)[:target_len]


def _window_starts(total_len: int, win_len: int, hop_len: int) -> list:
    if total_len <= win_len:
        return [0]
    return list(range(0, total_len, hop_len))


def _even_spacing_indices(n: int, k: int) -> list:
    if k <= 0 or k >= n:
        return list(range(n))
    import numpy as _np
    idx = _np.linspace(0, n - 1, k).astype(int)
    return idx.tolist()


@torch.no_grad()
def score_and_write_windowed(
    stage1: nn.Module,
    stage2: nn.Module,
    ds,
    device: torch.device,
    out_path: str,
    window_sec: float,
    hop_sec: float,
    agg: str,
    topk_frac: float,
    max_windows_per_utt: int,
    target_sample_rate: int,
    window_batch_size: int,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stage1.eval()
    stage2.eval()

    win_len = max(1, int(round(window_sec * target_sample_rate)))
    hop_len = max(1, int(round(hop_sec * target_sample_rate)))

    scores_by_utt = {}
    label_by_utt = {}
    source_by_utt = {}

    buf_wavs = []
    buf_utt = []

    def flush():
        if not buf_wavs:
            return
        batch = torch.stack(buf_wavs, dim=0).to(device)
        attn = torch.ones_like(batch, dtype=torch.long)
        embs = stage1(batch, attn)
        logits = stage2(embs)
        scores = logits.detach().cpu().tolist()
        for s, uid in zip(scores, buf_utt):
            scores_by_utt.setdefault(uid, []).append(float(s))
        buf_wavs.clear()
        buf_utt.clear()

    for idx in range(len(ds)):
        item = ds[idx]
        if len(item) >= 4:
            wav, label, source, utt_id = item[0], item[1], item[-2], item[-1]
        elif len(item) == 3:
            wav, label, utt_id = item[0], item[1], item[2]
            source = "NA"
        else:
            wav, label = item[0], item[1]
            utt_id = f"utt_{idx:06d}"
            source = "NA"

        utt_id = str(utt_id).strip().replace(" ", "_")
        source = str(source).strip().replace(" ", "_")
        label_val = int(label) if not isinstance(label, torch.Tensor) else int(label.item())

        if utt_id not in label_by_utt:
            label_by_utt[utt_id] = label_val
            source_by_utt[utt_id] = source

        wav = wav.to(device)
        total_len = int(wav.numel())
        starts = _window_starts(total_len, win_len, hop_len)
        if max_windows_per_utt > 0 and len(starts) > max_windows_per_utt:
            idxs = _even_spacing_indices(len(starts), max_windows_per_utt)
            starts = [starts[i] for i in idxs]

        for st in starts:
            seg = wav[st:st + win_len]
            if seg.numel() < win_len:
                seg = _repeat_pad_1d(seg, win_len)
            buf_wavs.append(seg)
            buf_utt.append(utt_id)
            if len(buf_wavs) >= window_batch_size:
                flush()

    flush()

    def aggregate(vals):
        if agg == "mean":
            return sum(vals) / len(vals)
        if agg == "max":
            return max(vals)
        if agg == "topk":
            k = max(1, int(round(len(vals) * topk_frac)))
            top = sorted(vals, reverse=True)[:k]
            return sum(top) / len(top)
        return sum(vals) / len(vals)

    with open(out_path, "w") as f:
        for utt_id, vals in scores_by_utt.items():
            score = aggregate(vals)
            key = "bonafide" if label_by_utt.get(utt_id, 0) == 1 else "spoof"
            source = source_by_utt.get(utt_id, "NA")
            f.write(f"{utt_id} {source} {key} {score:.6f}\n")

    print(f"[OK] Wrote (windowed): {out_path}")


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
                # Ensure fixed 4-column score format by stripping whitespace.
                utt_id = str(utt_ids[i]).strip().replace(" ", "_")
                source = str(sources[i]).strip().replace(" ", "_")
                key = "bonafide" if labels_np[i] == 1 else "spoof"
                f.write(f"{utt_id} {source} {key} {scores[i]:.6f}\n")

    print(f"[OK] Wrote: {out_path}")


def _merge_ranked_scores(out_path: str, world_size: int) -> None:
    with open(out_path, "w") as fout:
        for rank in range(world_size):
            part = f"{out_path}.rank{rank}"
            if not os.path.isfile(part):
                continue
            with open(part, "r") as fin:
                shutil.copyfileobj(fin, fout)
            os.remove(part)

def resolve_ckpts(exp_name: str, model_name: str, stage1_ckpt: str, stage2_ckpt: str,
                  stage1_root: str, stage2_root: str):
    model_id = os.path.basename(model_name.rstrip("/"))
    if not stage1_ckpt:
        stage1_ckpt = os.path.join(stage1_root, exp_name, f"{model_id}_stage1_head_best.pt")
    if not stage2_ckpt:
        stage2_ckpt = os.path.join(stage2_root, exp_name, model_id, "stage2_binary_head_best.pt")
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt}")
    if not os.path.isfile(stage2_ckpt):
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_ckpt}")
    return stage1_ckpt, stage2_ckpt

def dataset_specs(args):
    ff_speakers = [s.strip() for s in args.ff_speakers.split(",") if s.strip()]
    return {
        "asv19": {
            "cls": ASVspoof2019Dataset,
            "kwargs": {
                "root_dir": args.asv19_root,
                "protocol_file": args.asv19_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join(args.model_name, "score_cm_eval.txt"),
        },
        "asv5": {
            "cls": ASVspoof5Dataset,
            "kwargs": {
                "root_dir": args.asv5_root,
                "protocol_file": args.asv5_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join(args.model_name, "score_cm_eval_asv5.txt"),
        },
        "asv21_df": {
            "cls": ASVspoof2021DFDataset,
            "kwargs": {
                "root_dir": args.asv21_df_root,
                "protocol_file": args.asv21_df_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
                "skip_missing": args.skip_missing,
            },
            "score_rel": os.path.join("asv21_df", "score_cm_asv21_df.txt"),
        },
        "asv21_la": {
            "cls": ASVspoof2021LADataset,
            "kwargs": {
                "root_dir": args.asv21_la_root,
                "protocol_file": args.asv21_la_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
                "skip_missing": args.skip_missing,
            },
            "score_rel": os.path.join("asv21_la", "score_cm_asv21_la.txt"),
        },
        "itw": {
            "cls": InTheWildDataset,
            "kwargs": {
                "root_dir": args.itw_root,
                "protocol_file": args.itw_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
            },
            "score_rel": os.path.join(args.model_name, "score_cm_itw.txt"),
        },
        "famous_figures": {
            "cls": FamousFiguresDataset,
            "kwargs": {
                "protocol_file": args.ff_protocol,
                "root_dir": args.ff_root,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
                "include_speakers": ff_speakers if ff_speakers else None,
                "return_audio_name": True,
            },
            "score_rel": os.path.join("famous_figures", "score_cm_ff.txt"),
        },
        "fakexpose": {
            "cls": FakeXposeDataset,
            "kwargs": {
                "root_dir": args.fakexpose_root,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join("fakexpose", "score_cm_fakexpose.txt"),
        },
        "mlaad": {
            "cls": MLAADMailabsDataset,
            "kwargs": {
                "root_dir": args.mlaad_root,
                "protocol_file": args.mlaad_protocol,
                "subset": "all",
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join("mlaad", "score_cm_mlaad.txt"),
        },
        "deepfake_eval_2024": {
            "cls": DeepfakeEval2024Dataset,
            "kwargs": {
                "root_dir": args.deepfake_root,
                "protocol_file": args.deepfake_protocol,
                "subset": args.subset,
                "num_samples": args.num_samples,
                "max_duration_seconds": args.max_duration_seconds,
                "target_sample_rate": args.target_sample_rate,
            },
            "score_rel": os.path.join("deepfake_eval_2024", "score_cm_deepfake_eval_2024.txt"),
        },
    }

def _missing_inputs(kwargs):
    missing = []
    proto = kwargs.get("protocol_file")
    if proto and not os.path.isfile(proto):
        missing.append(proto)
    root = kwargs.get("root_dir")
    if root and not os.path.isdir(root):
        missing.append(root)
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_name", type=str, default="")
    ap.add_argument("--datasets", type=str, default="asv19,asv5,asv21_df,asv21_la,itw,famous_figures,fakexpose,mlaad")
    ap.add_argument("--stage1_ckpt", type=str, default="")
    ap.add_argument("--stage2_ckpt", type=str, default="")
    ap.add_argument("--stage1_root", type=str, default="checkpoints_stage1")
    ap.add_argument("--stage2_root", type=str, default="checkpoints_stage2")
    ap.add_argument("--scores_dir", type=str, default="scores")
    ap.add_argument("--model_name", type=str, default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--subset", type=str, default="all", choices=["all", "bonafide", "spoof"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_duration_seconds", type=int, default=5)
    ap.add_argument("--target_sample_rate", type=int, default=16000)
    ap.add_argument("--num_samples", type=int, default=None)
    ap.add_argument("--print_eer", action="store_true")
    ap.add_argument("--windowed_eval", type=int, default=0, choices=[0,1])
    ap.add_argument("--window_sec", type=float, default=None)
    ap.add_argument("--hop_sec", type=float, default=None)
    ap.add_argument("--agg", type=str, default="mean", choices=["mean","max","topk"])
    ap.add_argument("--topk_frac", type=float, default=0.3)
    ap.add_argument("--max_windows_per_utt", type=int, default=0)
    ap.add_argument("--window_batch_size", type=int, default=None)
    ap.add_argument("--skip_missing", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--asv19_root", type=str, default=DEFAULT_ASV19_EVAL_ROOT)
    ap.add_argument("--asv19_protocol", type=str, default=DEFAULT_ASV19_EVAL_PROTOCOL)
    ap.add_argument("--asv5_root", type=str, default=DEFAULT_ASV5_EVAL_ROOT)
    ap.add_argument("--asv5_protocol", type=str, default=DEFAULT_ASV5_EVAL_PROTOCOL)
    ap.add_argument("--asv21_df_root", type=str, default=DEFAULT_ASV21_DF_ROOT)
    ap.add_argument("--asv21_df_protocol", type=str, default=DEFAULT_ASV21_DF_PROTOCOL)
    ap.add_argument("--asv21_la_root", type=str, default=DEFAULT_ASV21_LA_ROOT)
    ap.add_argument("--asv21_la_protocol", type=str, default=DEFAULT_ASV21_LA_PROTOCOL)
    ap.add_argument("--itw_root", type=str, default=DEFAULT_ITW_ROOT)
    ap.add_argument("--itw_protocol", type=str, default=DEFAULT_ITW_PROTOCOL)
    ap.add_argument("--ff_protocol", type=str, default=DEFAULT_FF_PROTOCOL)
    ap.add_argument("--ff_root", type=str, default=DEFAULT_FF_ROOT)
    ap.add_argument(
        "--ff_speakers",
        type=str,
        default="",
        help="Comma-separated list of FamousFigures speaker names to include (optional).",
    )
    ap.add_argument("--fakexpose_root", type=str, default=DEFAULT_FAKEXPOSE_ROOT)
    ap.add_argument("--mlaad_root", type=str, default=DEFAULT_MLAAD_ROOT)
    ap.add_argument("--mlaad_protocol", type=str, default=DEFAULT_MLAAD_PROTOCOL)
    ap.add_argument("--deepfake_root", type=str, default=DEFAULT_DEEPFAKE_EVAL_ROOT)
    ap.add_argument("--deepfake_protocol", type=str, default=DEFAULT_DEEPFAKE_EVAL_PROTOCOL)
    ap.add_argument(
        "--force_rescore",
        action="store_true",
        help="Regenerate score files even if they already exist.",
    )
    args = ap.parse_args()
    if args.window_sec is None:
        args.window_sec = float(args.max_duration_seconds)
    if args.hop_sec is None:
        args.hop_sec = float(args.window_sec)
    if args.window_batch_size is None:
        args.window_batch_size = int(args.batch_size)

    is_distributed, rank, world_size, local_rank = setup_distributed()
    if torch.cuda.is_available():
        if is_distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    if args.windowed_eval and is_distributed:
        if rank != 0:
            return
    if not args.stage1_ckpt or not args.stage2_ckpt:
        if not args.exp_name:
            raise ValueError("Provide --exp_name or explicit --stage1_ckpt/--stage2_ckpt.")
    stage1_ckpt, stage2_ckpt = resolve_ckpts(
        args.exp_name,
        args.model_name,
        args.stage1_ckpt,
        args.stage2_ckpt,
        args.stage1_root,
        args.stage2_root,
    )

    stage1 = Stage1Backbone(stage1_ckpt, model_name=args.model_name, device=device)
    stage2 = load_stage2_head(stage2_ckpt, device=device)

    if not is_distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        stage1 = nn.DataParallel(stage1)
        stage2 = nn.DataParallel(stage2)

    spec_map = dataset_specs(args)
    score_exp = args.exp_name or "custom"
    dataset_list = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    for name in dataset_list:
        if name not in spec_map:
            print(f"[WARN] Unknown dataset '{name}', skipping.")
            continue
        spec = spec_map[name]
        missing = _missing_inputs(spec["kwargs"])
        if missing:
            print(f"[WARN] Missing inputs for '{name}': {missing}. Skipping.")
            continue
        ds = spec["cls"](**spec["kwargs"])
        if is_distributed:
            sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False)
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=pad_collate_fn_generic,
            )
        else:
            loader = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=pad_collate_fn_generic,
            )
        score_path = os.path.join(args.scores_dir, score_exp, spec["score_rel"])
        skip = False

        if is_distributed:
            flag = torch.tensor(int(skip), device=device)
            dist.broadcast(flag, src=0)
            skip = bool(flag.item())

        if rank == 0:
            print(f"[INFO] Scoring dataset '{name}' -> {score_path}")
        if args.windowed_eval:
            score_and_write_windowed(
                stage1,
                stage2,
                ds,
                device,
                score_path,
                window_sec=args.window_sec,
                hop_sec=args.hop_sec,
                agg=args.agg,
                topk_frac=args.topk_frac,
                max_windows_per_utt=args.max_windows_per_utt,
                target_sample_rate=args.target_sample_rate,
                window_batch_size=args.window_batch_size,
            )
        else:
            out_path = score_path
            if is_distributed:
                out_path = f"{score_path}.rank{rank}"
            score_and_write(stage1, stage2, loader, device, out_path)

        if is_distributed and not args.windowed_eval:
            dist.barrier()
            if rank == 0 and not skip:
                _merge_ranked_scores(score_path, world_size)
            dist.barrier()

        if args.print_eer and rank == 0:
            if os.path.isfile(score_path):
                eer = calculate_EER(score_path)
                print(f"EER ({name}): {eer}")
            else:
                print(f"[WARN] Score file missing for '{name}', skipping EER.")


if __name__ == "__main__":
    main()
