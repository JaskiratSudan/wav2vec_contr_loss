import csv
from pathlib import Path

import numpy as np
import librosa
import matplotlib.pyplot as plt

from evaluation import compute_eer  # same as your training code


# ===== CONFIG =====
AUDIO_PATH = "one_audio_analysis/audio_2.wav"
CSV_PATH   = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/model_outputs_model_2.csv"
OUT_PNG    = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/fake_timeline_itw_thr.png"

# IMPORTANT: set this to the exp that produced THIS CSV
EXP_NAME = "mem_bank_16384_supcon_temp_0.5"

SCORES_ROOT = "/home/jsudan/wav2vec_contr_loss/scores"
ITW_SCORE_REL = "facebook/wav2vec2-xls-r-300m/score_cm_itw.txt"

SR = 16000
TIME_GRID_STEP = 0.02  # 20ms grid for smoother curve

# Which logit to use from CSV:
# - If you FIXED your CSV saving and write logit_bonafide correctly, prefer "logit_bonafide".
# - If you did NOT fix it yet, your current CSV has the model score in "logit_spoof" (misnamed).
PREFERRED_LOGIT_COL = "logit_bonafide"  # try this first, fallback below
FALLBACK_LOGIT_COL  = "logit_spoof"     # your current CSV likely has the score here
# ==================


def load_itw_thr(exp_name: str) -> float:
    score_path = Path(SCORES_ROOT) / exp_name / ITW_SCORE_REL
    if not score_path.is_file():
        raise FileNotFoundError(f"ITW score file not found:\n{score_path}")

    cm_data = np.genfromtxt(score_path, dtype=str)
    # expected columns: [utt_id, ..., key, score] -> you used [:,2] key and [:,3] score
    keys = cm_data[:, 2]
    scores = cm_data[:, 3].astype(float)

    bona = scores[keys == "bonafide"]
    spoof = scores[keys == "spoof"]
    eer, thr = compute_eer(bona, spoof)
    return float(thr)


def load_windows(csv_path: str):
    """
    Returns list of (start_time, end_time, logit_score).
    logit_score is assumed to be the SAME score you threshold with:
        spoof if logit < thr
    """
    wins = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            st = float(row["start_time"])
            en = float(row["end_time"])

            logit = None
            if PREFERRED_LOGIT_COL in row and row[PREFERRED_LOGIT_COL] != "":
                logit = float(row[PREFERRED_LOGIT_COL])
            elif FALLBACK_LOGIT_COL in row and row[FALLBACK_LOGIT_COL] != "":
                logit = float(row[FALLBACK_LOGIT_COL])
            elif "score" in row and row["score"] != "":
                # last resort: use score column (not ideal)
                logit = float(row["score"])
            else:
                continue

            wins.append((st, en, logit))
    return wins


def shade_runs(ax, t_grid: np.ndarray, mask: np.ndarray, alpha: float = 0.18):
    """
    Shade contiguous True regions of mask on ax as axvspan.
    """
    if mask.sum() == 0:
        return
    m = mask.astype(np.int32)
    diff = np.diff(m, prepend=m[0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    # if mask starts True at t=0
    if mask[0] and (len(starts) == 0 or starts[0] != 0):
        starts = np.insert(starts, 0, 0)
    # if mask ends True at end
    if mask[-1]:
        ends = np.append(ends, len(mask) - 1)

    for s, e in zip(starts, ends):
        ax.axvspan(t_grid[s], t_grid[e], alpha=alpha)


def main():
    thr = load_itw_thr(EXP_NAME)
    print(f"[ITW] exp={EXP_NAME}  thr={thr:.6f}")

    y, _ = librosa.load(AUDIO_PATH, sr=SR, mono=True)
    dur = len(y) / SR
    t_audio = np.arange(len(y)) / SR

    wins = load_windows(CSV_PATH)
    if not wins:
        raise ValueError("No windows loaded from CSV (check columns/start_time/end_time).")

    # aggregated logit curve over time by averaging all windows covering each time point
    t_grid = np.arange(0.0, dur + 1e-9, TIME_GRID_STEP)
    logit_sum = np.zeros_like(t_grid)
    count = np.zeros_like(t_grid)

    for st, en, logit in wins:
        mask = (t_grid >= st) & (t_grid <= en)
        logit_sum[mask] += logit
        count[mask] += 1

    valid = count > 0
    logit_grid = np.full_like(t_grid, np.nan, dtype=np.float64)
    logit_grid[valid] = logit_sum[valid] / count[valid]

    # spoof region (consistent with your labeling rule)
    spoof_mask = valid & (logit_grid < thr)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    # waveform
    ax1.plot(t_audio, y, linewidth=0.8)
    ax1.set_xlim(0, dur)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Waveform")

    # shade spoof regions (based on aggregated logit < thr)
    shade_runs(ax1, t_grid, spoof_mask, alpha=0.18)

    # logit curve + thr
    ax2 = ax1.twinx()
    ax2.plot(t_grid, logit_grid, linewidth=2.0)
    ax2.axhline(thr, linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Aggregated model score (logit)")

    plt.title(f"Localization via ITW threshold: spoof if logit < thr (thr={thr:.4f})")
    plt.tight_layout()
    Path(OUT_PNG).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=250)
    plt.close()
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
