import os
from typing import List, Dict, Tuple
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv

from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule
from evaluation import compute_eer

# ---------------------------
# USER CONFIG (edit these)
# ---------------------------
AUDIO_PATH = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/Analyse_chinese.wav"
EXP_NAMES = [
    # "supcon_temp_0.07_uni_weight_0.1_uni_t_2",
    # "mem_bank_16384_supcon_temp_0.5",
    "mem_bank_512_supcon_temp_0.2",
    "mem_bank_0_supcon_geodesic_temp_0.2",
    # "mem_bank_512_supcon_geodesic_temp_0.2",
    # "mem_bank_512_supcon_temp_0.5",
    # "supcon_geodesic_temp_0.07",
    # "mem_bank_1024_supcon_temp_0.5",
    # "mem_bank_16384_supcon_geodesic_temp_0.2",
    # "hbm_256_mem_bank_16384_supcon_geodesic_temp_0.2",
    # "mem_bank_1024_supcon_temp_0.2",
    # "mem_bank_16384_supcon_temp_0.2",
    # "hbm_256_mem_bank_16384_supcon_temp_0.2",
    # "mem_bank_512_supcon_geodesic_temp_0.5",
    # "mem_bank_1024_supcon_geodesic_temp_0.5",
    # "mem_bank_1024_supcon_geodesic_temp_0.2",
    "asv5_supcon_geodesic_temp_0.07"
]

# ---------------------------
# HEATMAP FROM CSV OPTION
# ---------------------------
PLOT_HEATMAP_FROM_CSV = False   # <-- set True to skip model inference and plot directly from CSVs
HEATMAP_CSV_FILES = ["/home/jsudan/wav2vec_contr_loss/one_audio_analysis/multilingual_model.csv"]          # e.g. [".../model_outputs_model_1.csv", ".../model_outputs_model_2.csv"]
HEATMAP_CSV_NAMES = None        # optional: same length as HEATMAP_CSV_FILES; if None, uses filenames

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
STAGE1_ROOT = "/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1"
STAGE2_ROOT = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage2"

TARGET_SAMPLE_RATE = 16000
WINDOW_SEC = 10.0
HOP_SEC = 5
WINDOW_BATCH_SIZE = 16
TOPK_FRAC = 0.3  # used for "topk" aggregate
OUT_DIR = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis"
HEATMAP_FILE = "window_predictions_heatmap.png"
LOGITS_FILE = "window_logits.txt"
CSV_PREFIX = "model_outputs"
# EXTRA_CSV_PATH = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/multilingual_model.csv"
EXTRA_CSV_PATH = None
SCORES_ROOT = "/home/jsudan/wav2vec_contr_loss/scores"
ITW_SCORE_REL = "facebook/wav2vec2-xls-r-300m/score_cm_itw.txt"
ASV_SCORE_REL = "facebook/wav2vec2-xls-r-300m/score_cm_eval.txt"


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
    return clf


def resolve_ckpts(exp_name: str, model_name: str, stage1_root: str, stage2_root: str):
    model_id = os.path.basename(model_name.rstrip("/"))
    stage1_ckpt = os.path.join(stage1_root, exp_name, f"{model_id}_stage1_head_best.pt")
    stage2_ckpt = os.path.join(stage2_root, exp_name, model_id, "stage2_binary_head_best.pt")
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt}")
    if not os.path.isfile(stage2_ckpt):
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_ckpt}")
    return stage1_ckpt, stage2_ckpt


@torch.no_grad()
def _repeat_pad_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(target_len, device=x.device, dtype=x.dtype)
    if x.numel() >= target_len:
        return x[:target_len]
    reps = (target_len + x.numel() - 1) // x.numel()
    return x.repeat(reps)[:target_len]


def _window_starts(total_len: int, win_len: int, hop_len: int) -> List[int]:
    # hop_len <= 0 => treat as single-window mode
    if hop_len <= 0:
        return [0]
    if total_len <= win_len:
        return [0]
    return list(range(0, total_len, hop_len))


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(y).float()


def make_windows(wav: torch.Tensor, win_len: int, hop_len: int) -> List[torch.Tensor]:
    total_len = int(wav.numel())
    starts = _window_starts(total_len, win_len, hop_len)
    windows = []
    for st in starts:
        seg = wav[st:st + win_len]
        if seg.numel() < win_len:
            seg = _repeat_pad_1d(seg, win_len)
        windows.append(seg)
    return windows


@torch.no_grad()
def score_windows(stage1: nn.Module, stage2: nn.Module, windows: List[torch.Tensor],
                  device: torch.device, batch_size: int) -> List[float]:
    scores = []
    buf = []
    for seg in windows:
        buf.append(seg)
        if len(buf) >= batch_size:
            batch = torch.stack(buf, dim=0).to(device)
            attn = torch.ones_like(batch, dtype=torch.long)
            embs = stage1(batch, attn)
            logits = stage2(embs)
            scores.extend(logits.detach().cpu().tolist())
            buf.clear()
    if buf:
        batch = torch.stack(buf, dim=0).to(device)
        attn = torch.ones_like(batch, dtype=torch.long)
        embs = stage1(batch, attn)
        logits = stage2(embs)
        scores.extend(logits.detach().cpu().tolist())
    return [float(s) for s in scores]


def aggregate(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"mean": float("nan"), "max": float("nan"), "topk": float("nan")}
    mean = float(sum(scores) / len(scores))
    maxv = float(max(scores))
    k = max(1, int(round(len(scores) * TOPK_FRAC)))
    top = sorted(scores, reverse=True)[:k]
    topk = float(sum(top) / len(top))
    return {"mean": mean, "max": maxv, "topk": topk}


def print_table(rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("No results.")
        return
    headers = list(rows[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}
    def fmt_row(r):
        return " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers)
    sep = "-+-".join("-" * col_widths[h] for h in headers)
    print(fmt_row({h: h for h in headers}))
    print(sep)
    for r in rows:
        print(fmt_row(r))


def plot_label_heatmap(exp_names: List[str], all_labels: List[List[str]],
                       window_sec: float, hop_sec: float, out_path: str) -> None:
    if not all_labels:
        print("No labels to plot.")
        return
    max_windows = max(len(s) for s in all_labels)
    if max_windows == 0:
        print("No windows to plot.")
        return

    label_mat = np.full((len(all_labels), max_windows), np.nan, dtype=np.float32)
    for i, labels in enumerate(all_labels):
        for j, lab in enumerate(labels):
            if lab == "spoof":
                label_mat[i, j] = 1.0
            elif lab == "bonafide":
                label_mat[i, j] = 0.0

    plt.figure(figsize=(max(15, max_windows * 0.3), max(5, len(exp_names) * 0.6)))
    cmap = plt.matplotlib.colors.ListedColormap(["blue", "red"])
    bounds = [-0.5, 0.5, 1.5]
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = plt.imshow(
        label_mat,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    cbar = plt.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(["bonafide", "spoof"])

    y_labels = []
    multilingual_first = len(exp_names) > 0 and exp_names[0] == "multilingual_model"
    for i in range(len(exp_names)):
        if multilingual_first and i == 0:
            y_labels.append("Multilingual_Model")
        elif multilingual_first and 1 <= i <= 4:
            y_labels.append(f"Model_EN {i}")
        elif not multilingual_first and i <= 3:
            y_labels.append(f"Model_EN {i+1}")
        else:
            y_labels.append(f"Model {i+1}")
    plt.yticks(np.arange(len(exp_names)), y_labels)

    tick_idx = np.linspace(0, max_windows - 1, num=min(10, max_windows), dtype=int)

    def _fmt_mmss(t):
        m = int(t // 60)
        s = int(round(t % 60))
        return f"{m:02d}:{s:02d}"

    if hop_sec <= 0:
        # hop=0 => only one window or no time progression; show window indices
        plt.xticks(ticks=tick_idx, labels=[str(i) for i in tick_idx])
        plt.xlabel("Window index")
    else:
        tick_times = tick_idx * hop_sec
        plt.xticks(ticks=tick_idx, labels=[_fmt_mmss(t) for t in tick_times])
        plt.xlabel("Time (window start)")

    plt.ylabel("Model / Experiment")
    plt.title(f"Windowed Labels (win={window_sec:.1f}s, hop={hop_sec:.1f}s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()
    print(f"Saved heatmap: {out_path}")


def load_external_logits(csv_path: str) -> List[float]:
    if not csv_path or not os.path.isfile(csv_path):
        return []
    scores = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "logit_spoof" in row and row["logit_spoof"] != "":
                scores.append(float(row["logit_spoof"]))
            elif "score" in row and row["score"] != "":
                scores.append(float(row["score"]))
    return scores


def load_external_labels(csv_path: str) -> List[str]:
    if not csv_path or not os.path.isfile(csv_path):
        return []
    labels = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "label" in row and row["label"] != "":
                labels.append(row["label"].strip())
    return labels


def load_itw_thresholds(scores_root: str) -> Dict[str, float]:
    thresholds = {}
    root = Path(scores_root)
    if not root.exists():
        return thresholds
    for exp_dir in root.iterdir():
        if not exp_dir.is_dir():
            continue
        score_path = exp_dir / ASV_SCORE_REL
        if not score_path.is_file():
            continue
        try:
            cm_data = np.genfromtxt(score_path, dtype=str)
            cm_keys = cm_data[:, 2]
            cm_scores = cm_data[:, 3].astype(float)
            bona = cm_scores[cm_keys == "bonafide"]
            spoof = cm_scores[cm_keys == "spoof"]
            eer, thr = compute_eer(bona, spoof)
            thresholds[exp_dir.name] = float(thr)
        except Exception:
            continue
    return thresholds


def main():
    if not AUDIO_PATH or AUDIO_PATH == "/path/to/your_audio.wav":
        raise ValueError("Set AUDIO_PATH at the top of this script.")
    if not EXP_NAMES:
        raise ValueError("Add at least one experiment name to EXP_NAMES.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Device:", device)

    wav = load_audio_mono(AUDIO_PATH, TARGET_SAMPLE_RATE)
    win_len = int(round(WINDOW_SEC * TARGET_SAMPLE_RATE))
    hop_len = int(round(HOP_SEC * TARGET_SAMPLE_RATE))
    windows = make_windows(wav, win_len, hop_len)
    print(f"Audio windows: {len(windows)} (win={WINDOW_SEC}s, hop={HOP_SEC}s)")

    rows = []
    model_names = []
    model_scores = []
    model_labels = []
    thresholds = load_itw_thresholds(SCORES_ROOT)
    if not thresholds:
        print("[WARN] No ITW thresholds found. Defaulting to 0.0 for classification.")

    # Load multilingual model first (if available)
    extra_scores = load_external_logits(EXTRA_CSV_PATH)
    extra_labels = load_external_labels(EXTRA_CSV_PATH)
    if extra_scores:
        model_names.append("multilingual_model")
        model_scores.append(extra_scores)
        # Ensure labels align with windows
        labels = []
        for i in range(len(extra_scores)):
            if i < len(extra_labels):
                labels.append(extra_labels[i])
            else:
                labels.append("unknown")
        model_labels.append(labels)

    # Score experiments
    for exp in EXP_NAMES:
        stage1_ckpt, stage2_ckpt = resolve_ckpts(exp, MODEL_NAME, STAGE1_ROOT, STAGE2_ROOT)
        stage1 = Stage1Backbone(stage1_ckpt, model_name=MODEL_NAME, device=device)
        stage2 = load_stage2_head(stage2_ckpt, device=device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            stage1 = nn.DataParallel(stage1)
            stage2 = nn.DataParallel(stage2)

        scores = score_windows(stage1, stage2, windows, device, WINDOW_BATCH_SIZE)
        agg = aggregate(scores)

        model_names.append(exp)
        model_scores.append(scores)

        thr = thresholds.get(exp, 0.0)
        labels = ["bonafide" if s >= thr else "spoof" for s in scores]
        model_labels.append(labels)

        rows.append({
            "exp_name": exp,
            "n_windows": str(len(scores)),
            "mean": f"{agg['mean']:.6f}",
            "max": f"{agg['max']:.6f}",
            "topk": f"{agg['topk']:.6f}",
        })

    print_table(rows)

    os.makedirs(OUT_DIR, exist_ok=True)
    heatmap_path = os.path.join(OUT_DIR, HEATMAP_FILE)
    plot_label_heatmap(model_names, model_labels, WINDOW_SEC, HOP_SEC, heatmap_path)

    logits_path = os.path.join(OUT_DIR, LOGITS_FILE)
    with open(logits_path, "w") as f:
        # Header with window start times (mm:ss)
        def _fmt_mmss(t):
            m = int(t // 60)
            s = int(round(t % 60))
            return f"{m:02d}:{s:02d}"
        max_windows = max((len(s) for s in model_scores), default=0)
        time_labels = [_fmt_mmss(i * HOP_SEC) for i in range(max_windows)]
        header = "model\t" + "\t".join(time_labels) + "\n"
        f.write(header)

        for idx, scores in enumerate(model_scores):
            model_label = f"Model {idx+1}"
            score_str = " ".join(f"{s:.6f}" for s in scores)
            # Use tabs to align with header
            f.write(f"{model_label}\t" + "\t".join(f"{s:.6f}" for s in scores) + "\n")
    print(f"Saved logits: {logits_path}")

    # Save per-model CSVs with multilingual-style columns
    headers = [
        "filename",
        "label",
        "score",
        "spoof_prob",
        "bonafide_prob",
        "logit_spoof",
        "logit_bonafide",
        "start_time",
        "end_time",
    ]

    for model_idx, scores in enumerate(model_scores):
        csv_path = os.path.join(OUT_DIR, f"{CSV_PREFIX}_model_{model_idx+1}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for i, logit in enumerate(scores):
                start = i * HOP_SEC
                end = start + WINDOW_SEC
                label = model_labels[model_idx][i] if i < len(model_labels[model_idx]) else "unknown"
                bonafide_prob = 1.0 / (1.0 + np.exp(-logit))   # sigmoid(logit)
                spoof_prob = 1.0 - bonafide_prob
                score = bonafide_prob
                writer.writerow([
                    f"audio {i+1}",
                    label,
                    f"{score:.9f}",
                    f"{spoof_prob:.9f}",
                    f"{bonafide_prob:.9f}",
                    f"{-logit:.9f}",    # logit_spoof
                    f"{logit:.9f}",     # logit_bonafide
                    f"{start:.6f}",
                    f"{end:.6f}",
                ])
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
