from PIL import Image, ImageDraw, ImageFont
import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE = "plots/dep_embeddings/ITW"
MODEL = "facebook__wav2vec2-xls-r-300m"
FILENAME = "stage1_umap_itw_real_vs_spoof_clean.png"


def get_path(folder):
    return os.path.join(BASE, folder, MODEL, FILENAME)


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------
def measure_text(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        try:
            return int(draw.textlength(text, font=font)), font.size
        except AttributeError:
            return draw.textsize(text, font=font)


def draw_text_centered(draw, text, center_xy, font, fill=(0, 0, 0)):
    w, h = measure_text(draw, text, font)
    x = center_xy[0] - w / 2
    y = center_xy[1] - h / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_text_right_aligned(draw, text, right_x, center_y, font, fill=(0, 0, 0)):
    w, h = measure_text(draw, text, font)
    x = right_x - w
    y = center_y - h / 2
    draw.text((x, y), text, font=font, fill=fill)


# -----------------------------------------------------------------------------
# Main grid function
# -----------------------------------------------------------------------------
def make_grid(
    paths,
    row_labels,
    col_labels,
    title,
    out_path,
    cell_size=(260, 260),
    x_gap=6,
    y_gap=6,
    top_margin=6,
    bottom_margin=6,
    left_margin=6,
    right_margin=8,
    row_label_gap=12,
    col_label_h=38,
    title_h=0,
    font_size=24,
    legend_font_size=18,
    legend_dot_r=5,
    legend_line_gap=4,
    bg_color=(255, 255, 255),
):
    # -------------------------------------------------------------------------
    # Fonts
    # -------------------------------------------------------------------------
    try:
        font_title = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size + 2,
        )
        font_col = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size,
        )
        font_row = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size - 2,
        )
        font_leg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            legend_font_size,
        )
    except Exception:
        font_title = font_col = font_row = font_leg = ImageFont.load_default()

    tmp = Image.new("RGB", (10, 10), bg_color)
    tmp_draw = ImageDraw.Draw(tmp)

    cell_w, cell_h = cell_size
    n_rows = len(paths)
    n_cols = len(paths[0])

    # -------------------------------------------------------------------------
    # Measure row label width tightly
    # -------------------------------------------------------------------------
    max_row_w = 0
    for label in row_labels:
        for line in label.split("\n"):
            w, _ = measure_text(tmp_draw, line, font_row)
            max_row_w = max(max_row_w, w)

    # -------------------------------------------------------------------------
    # Legend setup, 2 lines and smaller
    # -------------------------------------------------------------------------
    legend_items = [("Real", (65, 105, 225)), ("Spoof", (220, 20, 60))]
    leg_text_ws = [measure_text(tmp_draw, txt, font_leg)[0] for txt, _ in legend_items]
    leg_text_h = max(measure_text(tmp_draw, txt, font_leg)[1] for txt, _ in legend_items)

    legend_block_w = max(2 * legend_dot_r + 8 + w for w in leg_text_ws) + 4
    legend_block_h = 2 * leg_text_h + legend_line_gap + 4

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------
    grid_left = left_margin + max_row_w + row_label_gap
    grid_top = title_h + top_margin + col_label_h + y_gap
    grid_w = n_cols * cell_w + (n_cols - 1) * x_gap
    grid_h = n_rows * cell_h + (n_rows - 1) * y_gap

    total_w = grid_left + grid_w + right_margin + legend_block_w
    total_h = title_h + top_margin + col_label_h + y_gap + grid_h + bottom_margin

    canvas = Image.new("RGB", (total_w, total_h), color=bg_color)
    draw = ImageDraw.Draw(canvas)

    # -------------------------------------------------------------------------
    # Title
    # -------------------------------------------------------------------------
    if title_h > 0 and title:
        draw_text_centered(draw, title, (total_w / 2, title_h / 2), font_title)

    # -------------------------------------------------------------------------
    # Column labels
    # -------------------------------------------------------------------------
    col_label_y = title_h + top_margin + col_label_h / 2
    for c, col_label in enumerate(col_labels):
        x_center = grid_left + c * (cell_w + x_gap) + cell_w / 2
        draw_text_centered(draw, col_label, (x_center, col_label_y), font_col)

    # -------------------------------------------------------------------------
    # Legend at top-right
    # -------------------------------------------------------------------------
    legend_x = grid_left + grid_w + right_margin
    legend_y = title_h + top_margin + 2

    for i, (txt, color) in enumerate(legend_items):
        cy = legend_y + i * (leg_text_h + legend_line_gap) + leg_text_h / 2

        dot_box = [
            legend_x,
            cy - legend_dot_r,
            legend_x + 2 * legend_dot_r,
            cy + legend_dot_r,
        ]
        draw.ellipse(dot_box, fill=color)

        text_x = legend_x + 2 * legend_dot_r + 8
        draw_text_right_aligned(
            draw,
            txt,
            right_x=text_x + leg_text_ws[i],
            center_y=cy,
            font=font_leg,
        )

    # -------------------------------------------------------------------------
    # Row labels and images
    # -------------------------------------------------------------------------
    label_right_x = left_margin + max_row_w

    for r, (path_row, row_label) in enumerate(zip(paths, row_labels)):
        y_top = grid_top + r * (cell_h + y_gap)
        y_center = y_top + cell_h / 2

        lines = row_label.split("\n")
        line_heights = [measure_text(tmp_draw, line, font_row)[1] for line in lines]
        total_text_h = sum(line_heights) + (len(lines) - 1) * 2

        cur_y = y_center - total_text_h / 2
        for line, lh in zip(lines, line_heights):
            draw_text_right_aligned(
                draw,
                line,
                right_x=label_right_x,
                center_y=cur_y + lh / 2,
                font=font_row,
            )
            cur_y += lh + 2

        for c, p in enumerate(path_row):
            x_left = grid_left + c * (cell_w + x_gap)

            if os.path.exists(p):
                img = Image.open(p).convert("RGB").resize(cell_size, Image.LANCZOS)
            else:
                print(f"MISSING: {p}")
                img = Image.new("RGB", cell_size, color=(210, 210, 210))

            canvas.paste(img, (int(x_left), int(y_top)))

    canvas.save(out_path, dpi=(300, 300))
    print(f"Saved: {out_path}  ({total_w}x{total_h}px)")


# -----------------------------------------------------------------------------
# Figure A: Temperature Sweep
# -----------------------------------------------------------------------------
temps = ["0.07", "0.1", "0.3", "0.6"]

paths_A = [
    [get_path(f"supcon_temp_{t}") for t in temps],
    [get_path(f"supcon_geodesic_temp_{t}") for t in temps],
]

make_grid(
    paths=paths_A,
    row_labels=["Cosine", "Geodesic"],
    col_labels=[f"τ = {t}" for t in temps],
    title="",
    out_path="temperature_sweep.pdf",
    cell_size=(260, 260),
    x_gap=6,
    y_gap=6,
    row_label_gap=12,
    col_label_h=38,
    font_size=24,
    legend_font_size=18,
    legend_dot_r=5,
)


# -----------------------------------------------------------------------------
# Figure B: Queue Ablation
# -----------------------------------------------------------------------------
queues = [None, "256", "1024", "4096", "8192"]
col_labels_B = ["No Queue", "|Q|=128", "|Q|=512", "|Q|=2048", "|Q|=4096"]


def queue_path(similarity, temp, q):
    if q is None:
        folder = (
            f"supcon_geodesic_temp_{temp}"
            if similarity == "geodesic"
            else f"supcon_temp_{temp}"
        )
    else:
        folder = (
            f"mem_{q}_supcon_geodesic_temp_{temp}"
            if similarity == "geodesic"
            else f"mem_{q}_supcon_temp_{temp}"
        )
    return get_path(folder)


paths_B = [
    [queue_path("cosine", "0.3", q) for q in queues],
    [queue_path("geodesic", "0.07", q) for q in queues],
]

make_grid(
    paths=paths_B,
    row_labels=["Cosine\nτ = 0.30", "Geodesic\nτ = 0.07"],
    col_labels=col_labels_B,
    title="",
    out_path="queue_ablation.pdf",
    cell_size=(260, 260),
    x_gap=6,
    y_gap=6,
    row_label_gap=12,
    col_label_h=38,
    font_size=24,
    legend_font_size=18,
    legend_dot_r=5,
)