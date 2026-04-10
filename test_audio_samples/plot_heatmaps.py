import csv
import os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
# Path to the CSV saved by infer.py
CSV_PATH = "/home/jsudan/wav2vec_contr_loss/test_audio_samples/model_logs/Benjamin_Netanyahu_audio.csv"

# Where to save the heatmap image
PLOT_PATH = "/home/jsudan/wav2vec_contr_loss/test_audio_samples/plots/Benjamin_Netanyahu_audio.png"

# Whether to skip plotting the final chunk
SKIP_LAST_CHUNK = True


# -------------------------------------------------
# CSV LOADER
# -------------------------------------------------
def load_csv(csv_path: str):
 
    meta = {}
    rows = []

    with open(csv_path, newline="") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("#"):
                content = stripped.lstrip("#").strip()
                if "=" in content:
                    key, _, value = content.partition("=")
                    meta[key.strip()] = value.strip()
                continue

            if not stripped:
                continue

            remaining = f.read()
            reader = csv.DictReader([stripped] + remaining.splitlines())
            for row in reader:
                rows.append({
                    "chunk": int(row["chunk"]),
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "logit": float(row["logit"]),
                    "label": row["label"].strip(),
                })
            break

    return meta, rows


# -------------------------------------------------
# HEATMAP
# -------------------------------------------------
def format_seconds(seconds: float) -> str:
    """
    Format x-axis labels as seconds only.
    Example: 0, 5, 10, 15
    """
    if abs(seconds - round(seconds)) < 1e-6:
        return f"{int(round(seconds))}"
    return f"{seconds:.1f}"


def save_heatmap(rows, meta, plot_path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    if SKIP_LAST_CHUNK and len(rows) > 1:
        rows = rows[:-1]

    if not rows:
        raise ValueError("No rows available to plot.")

    # metadata
    threshold = float(meta.get("threshold", 0.0))
    window_sec = float(meta.get("window_sec", 10.0))
    hop_sec = float(meta.get("hop_sec", 5.0))
    audio_path = meta.get("audio_path", "unknown")

    logits = [r["logit"] for r in rows]
    starts_sec = [r["start_sec"] for r in rows]
    n = len(rows)

    tick_labels = [format_seconds(s) for s in starts_sec]

    # data for imshow
    data = np.array(logits).reshape(1, n)

    # color scaling centered at threshold
    vmin = min(logits) - 0.5
    vmax = max(logits) + 0.5
    if vmin >= threshold:
        vmin = threshold - 1.0
    if vmax <= threshold:
        vmax = threshold + 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=threshold, vmax=vmax)

    # figure
    fig_width = max(12, n * 1.15)
    fig_height = 4.1
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdYlGn",
        norm=norm,
        interpolation="nearest",
    )

    # bottom x-axis
    ax.set_xticks(range(n))
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=11)
    ax.set_yticks([])
    ax.set_xlabel(
        f"Window start time in seconds (window={window_sec}s, hop={hop_sec}s)",
        fontsize=13,
        labelpad=10,
    )

    # top x-axis: chunk indices
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(range(n))
    ax_top.set_xticklabels([f"#{r['chunk']}" for r in rows], fontsize=12)
    ax_top.set_xlabel("Chunk index", fontsize=13, labelpad=8)

    # title
    # audio_name = os.path.basename(audio_path)
    # fig.suptitle(
    #     f"Deepfake detection heatmap - {audio_name}",
    #     fontsize=18,
    #     fontweight="bold",
    #     y=0.97,
    # )

    # layout
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.23, top=0.76)

    # colorbar only
    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        pad=0.02,
        fraction=0.035
    )

    # remove numeric ticks and default label
    cbar.set_ticks([])
    cbar.ax.set_ylabel("")

    # threshold marker line only
    cbar.ax.axhline(threshold, color="black", linewidth=1.5, linestyle="--")

    # top and bottom labels on colorbar
    cbar.ax.text(
        1.8, 0.98, "bonafide",
        transform=cbar.ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold"
    )
    cbar.ax.text(
        1.8, 0.02, "spoof",
        transform=cbar.ax.transAxes,
        ha="left", va="bottom",
        fontsize=11, fontweight="bold"
    )

    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved to: {plot_path}")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    csv_path = CSV_PATH
    if not csv_path:
        csv_path = input("Enter path to results CSV: ").strip()

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    meta, rows = load_csv(csv_path)

    print(f"\nLoaded {len(rows)} windows from: {csv_path}")
    print(f"  audio      : {meta.get('audio_path', 'N/A')}")
    print(f"  exp_name   : {meta.get('exp_name', 'N/A')}")
    print(f"  threshold  : {meta.get('threshold', 'N/A')}")
    print(f"  window_sec : {meta.get('window_sec', 'N/A')}")
    print(f"  hop_sec    : {meta.get('hop_sec', 'N/A')}")

    if SKIP_LAST_CHUNK and len(rows) > 1:
        print("  plotting    : all chunks except the last one")

    plot_path = PLOT_PATH
    if not plot_path:
        plot_path = input("Enter path to save heatmap (e.g. /home/jsudan/plots/out.png): ").strip()

    save_heatmap(rows, meta, plot_path)


if __name__ == "__main__":
    main()