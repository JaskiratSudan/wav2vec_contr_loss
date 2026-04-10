import csv
import os
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Wav2Vec2Encoder
from compression_module import CompressionModule


# -------------------------------------------------
# USER CONFIG
# -------------------------------------------------
AUDIO_PATH = "/home/jsudan/wav2vec_contr_loss/test_audio_samples/Benjamin_Netanyahu_audio.wav"

EXP_NAME = "asv5_hbm_1024_warmepo_5_mem_16384_qst_2_supcon_geodesic_temp_0.07"
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

STAGE1_ROOT = "/scratch/hafiz_root/hafiz1/jsudan/wav2vec_contr_loss/checkpoints_stage1"
STAGE2_ROOT = "/home/jsudan/wav2vec_contr_loss/checkpoints_stage2"

TARGET_SAMPLE_RATE = 16000
WINDOW_SEC = 10.0
HOP_SEC = 5
BATCH_SIZE = 8

# THRESHOLD = -9.399353       # ITW threshold for this model
THRESHOLD = -8.399353         # Custom threshold

# From your EER table:
# higher score means bonafide
HIGHER_SCORE_MEANS_BONAFIDE = True

# -------------------------------------------------
# CSV CONFIG
# -------------------------------------------------
# Set to None to be prompted at runtime, or supply a path directly.
# Example: CSV_PATH = "/home/jsudan/results/trump_1990.csv"
CSV_PATH = "/home/jsudan/wav2vec_contr_loss/test_audio_samples/model_logs/Benjamin_Netanyahu_audio.csv"


# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def safe_load(path: str, map_location: torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_state_dict_flexible(model: torch.nn.Module, state_dict: dict) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        cleaned = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }
        model.load_state_dict(cleaned, strict=True)


def resolve_ckpts(exp_name: str, model_name: str, stage1_root: str, stage2_root: str):
    model_id = os.path.basename(model_name.rstrip("/"))
    stage1_ckpt = os.path.join(stage1_root, exp_name, f"{model_id}_stage1_head_best.pt")
    stage2_ckpt = os.path.join(stage2_root, exp_name, model_id, "stage2_binary_head_best.pt")

    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt}")
    if not os.path.isfile(stage2_ckpt):
        raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_ckpt}")

    return stage1_ckpt, stage2_ckpt


def load_audio_mono(path: str, target_sr: int) -> torch.Tensor:
    y, _ = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(y).float()


def repeat_pad_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
    if x.numel() == 0:
        return torch.zeros(target_len, dtype=x.dtype)
    if x.numel() >= target_len:
        return x[:target_len]
    reps = (target_len + x.numel() - 1) // x.numel()
    return x.repeat(reps)[:target_len]


def make_windows(wav: torch.Tensor, win_len: int, hop_len: int):
    total_len = int(wav.numel())
    starts = [0] if total_len <= win_len else list(range(0, total_len, hop_len))

    windows = []
    for st in starts:
        seg = wav[st:st + win_len]
        if seg.numel() < win_len:
            seg = repeat_pad_1d(seg, win_len)
        windows.append((st, seg))
    return windows


def format_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


# -------------------------------------------------
# MODEL
# -------------------------------------------------
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
        raise ValueError(f"Unknown HEAD_TYPE: {head_type}")

    clf.load_state_dict(ckpt["model_state_dict"])
    clf.eval()

    for p in clf.parameters():
        p.requires_grad = False

    return clf


# -------------------------------------------------
# INFERENCE
# -------------------------------------------------
@torch.no_grad()
def score_windows(stage1, stage2, windows, device, batch_size):
    results = []

    batch_segments = []
    batch_starts = []

    def flush():
        if not batch_segments:
            return

        batch = torch.stack(batch_segments, dim=0).to(device)
        attn = torch.ones_like(batch, dtype=torch.long)
        embs = stage1(batch, attn)
        logits = stage2(embs).detach().cpu().tolist()

        for st, logit in zip(batch_starts, logits):
            results.append((st, float(logit)))

        batch_segments.clear()
        batch_starts.clear()

    for st, seg in windows:
        batch_segments.append(seg)
        batch_starts.append(st)
        if len(batch_segments) >= batch_size:
            flush()

    flush()
    return results


def predict_label(logit: float, threshold: float) -> str:
    if HIGHER_SCORE_MEANS_BONAFIDE:
        return "bonafide" if logit >= threshold else "spoof"
    return "spoof" if logit >= threshold else "bonafide"


# -------------------------------------------------
# CSV SAVE
# -------------------------------------------------
def save_csv(results, csv_path):
    """
    Writes inference results to a CSV file.

    Format
    ------
    Metadata lines (prefixed with '#') carry all the run config so the
    plot script can reconstruct the heatmap without any hardcoded values.
    Data lines follow as standard CSV rows.

    # audio_path=...
    # exp_name=...
    # threshold=...
    # window_sec=...
    # hop_sec=...
    # higher_score_means_bonafide=...
    # sample_rate=...
    chunk,start_sec,end_sec,logit,label
    1,0.0,10.0,-8.912345,spoof
    ...
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        # -- metadata header (readable by plot_heatmap.py) --
        f.write(f"# audio_path={AUDIO_PATH}\n")
        f.write(f"# exp_name={EXP_NAME}\n")
        f.write(f"# threshold={THRESHOLD}\n")
        f.write(f"# window_sec={WINDOW_SEC}\n")
        f.write(f"# hop_sec={HOP_SEC}\n")
        f.write(f"# higher_score_means_bonafide={HIGHER_SCORE_MEANS_BONAFIDE}\n")
        f.write(f"# sample_rate={TARGET_SAMPLE_RATE}\n")

        # -- data rows --
        writer = csv.writer(f)
        writer.writerow(["chunk", "start_sec", "end_sec", "logit", "label"])

        for i, (st, logit) in enumerate(results):
            start_sec = st / TARGET_SAMPLE_RATE
            end_sec   = start_sec + WINDOW_SEC
            label     = predict_label(logit, THRESHOLD)
            writer.writerow([i + 1, f"{start_sec:.6f}", f"{end_sec:.6f}", f"{logit:.6f}", label])

    print(f"\nResults saved to: {csv_path}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    stage1_ckpt, stage2_ckpt = resolve_ckpts(
        EXP_NAME, MODEL_NAME, STAGE1_ROOT, STAGE2_ROOT
    )

    stage1 = Stage1Backbone(stage1_ckpt, MODEL_NAME, device)
    stage2 = load_stage2_head(stage2_ckpt, device)

    wav = load_audio_mono(AUDIO_PATH, TARGET_SAMPLE_RATE)

    win_len = int(WINDOW_SEC * TARGET_SAMPLE_RATE)
    hop_len = int(HOP_SEC * TARGET_SAMPLE_RATE)

    windows = make_windows(wav, win_len, hop_len)
    results = score_windows(stage1, stage2, windows, device, BATCH_SIZE)

    print(f"\nAudio: {AUDIO_PATH}")
    print(f"Experiment: {EXP_NAME}")
    print(f"Threshold: {THRESHOLD}")
    print(f"Window: {WINDOW_SEC}s, Hop: {HOP_SEC}s")
    print("-" * 80)
    print(f"{'chunk':<8} {'start':<10} {'end':<10} {'logit':<12} {'label':<10}")
    print("-" * 80)

    for i, (st, logit) in enumerate(results):
        start_sec = st / TARGET_SAMPLE_RATE
        end_sec   = start_sec + WINDOW_SEC
        label     = predict_label(logit, THRESHOLD)
        print(
            f"{i+1:<8} "
            f"{format_time(start_sec):<10} "
            f"{format_time(end_sec):<10} "
            f"{logit:<12.6f} "
            f"{label:<10}"
        )

    # -- save CSV --
    csv_path = CSV_PATH
    if not csv_path:
        csv_path = input("\nEnter path to save CSV (e.g. /home/jsudan/results/trump_1990.csv): ").strip()

    save_csv(results, csv_path)


if __name__ == "__main__":
    main()