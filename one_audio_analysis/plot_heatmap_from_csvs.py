import re
import csv
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


# ====== CONFIG ======
CSV_DIR = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/model_outputs"
OUT_PATH = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/window_predictions_heatmap.png"

WINDOW_SEC = 10.0
HOP_SEC = 5.0

FILENAME_PREFIX = None  # e.g. "one_audio__" or None
# ====================


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def norm_label(x: str) -> str:
    t = (x or "").strip().lower()
    if t in {"bonafide", "bona fide", "bona_fide", "real", "genuine"}:
        return "bonafide"
    if t in {"spoof", "fake", "synthetic"}:
        return "spoof"
    return "unknown"


def read_labels(csv_path: str) -> List[str]:
    labels = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        if "label" not in (r.fieldnames or []):
            raise ValueError(f"'label' column missing in {csv_path}. Found: {r.fieldnames}")
        for row in r:
            labels.append(norm_label(row.get("label", "")))
    return labels


def fmt_mmss(t: float) -> str:
    m = int(t // 60)
    s = int(round(t % 60))
    return f"{m:02d}:{s:02d}"


def main():
    root = Path(CSV_DIR)
    if not root.exists():
        raise FileNotFoundError(f"CSV_DIR not found: {CSV_DIR}")

    csv_files = [p for p in root.glob("*.csv") if p.is_file()]
    if FILENAME_PREFIX:
        csv_files = [p for p in csv_files if p.name.startswith(FILENAME_PREFIX)]

    csv_files = sorted(csv_files, key=lambda p: natural_key(p.name))
    if not csv_files:
        raise ValueError("No CSV files found to plot.")

    all_labels = [read_labels(str(p)) for p in csv_files]
    max_windows = max((len(x) for x in all_labels), default=0)
    n_models = len(all_labels)

    if max_windows == 0:
        raise ValueError("All CSVs have 0 windows (no rows).")

    # matrix: 0=bonafide (blue), 1=spoof (red), nan=unknown
    mat = np.full((n_models, max_windows), np.nan, dtype=np.float32)
    for i, labels in enumerate(all_labels):
        for j, lab in enumerate(labels):
            if lab == "bonafide":
                mat[i, j] = 0.0
            elif lab == "spoof":
                mat[i, j] = 1.0

    y_labels = [f"Model {i+1}" for i in range(n_models)]

    plt.figure(figsize=(max(12, max_windows * 0.3), max(4, n_models * 0.55)))
    cmap = plt.matplotlib.colors.ListedColormap(["blue", "red"])
    norm = plt.matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, ticks=[0, 1])
    cbar.ax.set_yticklabels(["bonafide", "spoof"])

    plt.yticks(np.arange(n_models), y_labels)

    # how many time labels you want on the x-axis
    N_TIME_TICKS = 25  # try 20/25/30 depending on how dense you want

    tick_idx = np.linspace(0, max_windows - 1, num=min(N_TIME_TICKS, max_windows), dtype=int)
    tick_times = tick_idx * HOP_SEC
    plt.xticks(tick_idx, [fmt_mmss(t) for t in tick_times], rotation=45, ha="right")

    plt.xlabel("Time (window start)")
    plt.ylabel("Models (sequential order)")
    plt.title(f"Window Labels Heatmap (win={WINDOW_SEC}s, hop={HOP_SEC}s)")

    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=250)
    plt.close()

    print(f"Saved heatmap: {OUT_PATH}")
    print("Row order used (top to bottom):")
    for i, p in enumerate(csv_files, start=1):
        print(f"  Model {i}: {p.name}")


if __name__ == "__main__":
    main()
