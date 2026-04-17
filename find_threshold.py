import os
import csv
import glob
import numpy as np

from evaluation import compute_eer

# -------------------------------------------------
# USER CONFIG
# -------------------------------------------------
EXP_NAMES = [
    # "asv5_hbm_1024_warmepo_5_mem_16384_qst_2_supcon_geodesic_temp_0.07",
    "asv5_hbm_1024_mem_16384_supcon_geodesic_temp_0.07",
    # "asv5_mem_4096_qst_3_supcon_geodesic_temp_0.07",
    # "mem_4096_supcon_temp_0.3",
]

DATASETS = [
    "itw",
    "asv19",
    "asv5",
    "asv21_df",
    "asv21_la",
    "famous_figures",
    "fakexpose",
    "mlaad",
    "deepfake_eval_2024",
]

SCORES_DIR = "/home/jsudan/wav2vec_contr_loss/scores"
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

SAVE_CSV = False
CSV_OUT = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/eer_threshold_results.csv"

# Set to True to pool all datasets and compute a single deployment threshold.
# Set to False for the original per-dataset EER mode.
POOLED_MODE = True


# -------------------------------------------------
# SCORE FILE NAMES
# -------------------------------------------------
DATASET_SCORE_FILE_MAP = {
    "asv19": "score_cm_eval.txt",
    "asv5": "score_cm_eval_asv5.txt",
    "asv21_df": "score_cm_asv21_df.txt",
    "asv21_la": "score_cm_asv21_la.txt",
    "itw": "score_cm_itw.txt",
    "famous_figures": "score_cm_ff.txt",
    "fakexpose": "score_cm_fakexpose.txt",
    "mlaad": "score_cm_mlaad.txt",
    "deepfake_eval_2024": "score_cm_deepfake_eval_2024.txt",
}


def build_candidate_paths(exp_name: str, dataset_name: str, scores_dir: str, model_name: str):
    if dataset_name not in DATASET_SCORE_FILE_MAP:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    score_filename = DATASET_SCORE_FILE_MAP[dataset_name]
    model_name = model_name.strip("/")
    model_base = os.path.basename(model_name)

    candidates = []
    candidates.append(os.path.join(scores_dir, exp_name, model_name, score_filename))
    candidates.append(os.path.join(scores_dir, exp_name, model_base, score_filename))
    candidates.append(os.path.join(scores_dir, exp_name, dataset_name, score_filename))
    candidates.append(os.path.join(scores_dir, exp_name, score_filename))

    return candidates


def resolve_score_file(exp_name: str, dataset_name: str, scores_dir: str, model_name: str):
    candidates = build_candidate_paths(exp_name, dataset_name, scores_dir, model_name)

    for path in candidates:
        if os.path.isfile(path):
            return path

    score_filename = DATASET_SCORE_FILE_MAP[dataset_name]
    exp_root = os.path.join(scores_dir, exp_name)
    hits = glob.glob(os.path.join(exp_root, "**", score_filename), recursive=True)

    if hits:
        return hits[0]

    return None


def load_score_file(score_file: str):
    if not os.path.isfile(score_file):
        raise FileNotFoundError(f"Score file not found: {score_file}")

    data = np.genfromtxt(score_file, dtype=str)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        raise ValueError(f"Expected at least 4 columns in {score_file}, got shape {data.shape}")

    labels = data[:, 2]
    scores = data[:, 3].astype(float)

    return labels, scores


def calculate_eer_and_threshold(score_file: str):
    labels, scores = load_score_file(score_file)

    bona_scores = scores[labels == "bonafide"]
    spoof_scores = scores[labels == "spoof"]

    if len(bona_scores) == 0:
        raise ValueError(f"No bonafide samples found in {score_file}")
    if len(spoof_scores) == 0:
        raise ValueError(f"No spoof samples found in {score_file}")

    eer, threshold = compute_eer(bona_scores, spoof_scores)

    bona_mean = float(np.mean(bona_scores))
    spoof_mean = float(np.mean(spoof_scores))

    if bona_mean > spoof_mean:
        direction = "higher score means bonafide"
    elif spoof_mean > bona_mean:
        direction = "higher score means spoof"
    else:
        direction = "same mean score"

    return {
        "eer_percent": float(eer * 100.0),
        "threshold": float(threshold),
        "bonafide_mean": bona_mean,
        "spoof_mean": spoof_mean,
        "n_bonafide": int(len(bona_scores)),
        "n_spoof": int(len(spoof_scores)),
        "direction": direction,
    }


def per_dataset_stats_at_threshold(labels, scores, threshold, higher_is_bonafide):
    """Compute FAR, FRR, and accuracy at a fixed threshold for one dataset."""
    bona = scores[labels == "bonafide"]
    spoof = scores[labels == "spoof"]
    if higher_is_bonafide:
        # bonafide accepted when score >= threshold; spoof accepted when score >= threshold
        fa = float(np.sum(spoof >= threshold))   # false accepts (spoof above thresh)
        fr = float(np.sum(bona < threshold))      # false rejects (bona below thresh)
    else:
        fa = float(np.sum(spoof < threshold))
        fr = float(np.sum(bona >= threshold))
    far = fa / len(spoof) * 100.0 if len(spoof) > 0 else float("nan")
    frr = fr / len(bona) * 100.0 if len(bona) > 0 else float("nan")
    correct = (len(bona) - fr) + (len(spoof) - fa)
    acc = correct / (len(bona) + len(spoof)) * 100.0 if (len(bona) + len(spoof)) > 0 else float("nan")
    return {"far_pct": far, "frr_pct": frr, "acc_pct": acc}


def calculate_pooled_threshold(exp_name, datasets, scores_dir, model_name):
    """
    Load score files for all datasets, pool them, and compute a single EER + threshold.

    Returns
    -------
    threshold : float
    eer_pct : float
    higher_is_bonafide : bool
    per_ds : list of dicts  (one per dataset, with raw labels/scores + stats)
    """
    all_bona = []
    all_spoof = []
    per_ds = []

    for ds_name in datasets:
        score_file = resolve_score_file(exp_name, ds_name, scores_dir, model_name)
        if score_file is None:
            print(f"  [WARN] Missing score file for '{ds_name}' — skipping.")
            per_ds.append({"dataset": ds_name, "status": "missing"})
            continue
        try:
            labels, scores = load_score_file(score_file)
        except Exception as e:
            print(f"  [WARN] Could not load '{ds_name}': {e} — skipping.")
            per_ds.append({"dataset": ds_name, "status": f"error: {e}"})
            continue

        bona = scores[labels == "bonafide"]
        spoof = scores[labels == "spoof"]
        all_bona.append(bona)
        all_spoof.append(spoof)
        per_ds.append({
            "dataset": ds_name,
            "status": "ok",
            "labels": labels,
            "scores": scores,
            "n_bonafide": len(bona),
            "n_spoof": len(spoof),
        })
        print(f"  [OK] Loaded '{ds_name}': {len(bona)} bonafide, {len(spoof)} spoof")

    if not all_bona or not all_spoof:
        raise RuntimeError("No score files could be loaded — cannot compute pooled threshold.")

    pooled_bona = np.concatenate(all_bona)
    pooled_spoof = np.concatenate(all_spoof)

    eer, threshold = compute_eer(pooled_bona, pooled_spoof)
    higher_is_bonafide = float(np.mean(pooled_bona)) >= float(np.mean(pooled_spoof))

    return float(threshold), float(eer) * 100.0, higher_is_bonafide, per_ds, int(len(pooled_bona)), int(len(pooled_spoof))


def print_table(rows):
    if not rows:
        print("No results.")
        return

    headers = list(rows[0].keys())
    widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    def fmt(row):
        return " | ".join(str(row[h]).ljust(widths[h]) for h in headers)

    sep = "-+-".join("-" * widths[h] for h in headers)

    print(fmt({h: h for h in headers}))
    print(sep)
    for row in rows:
        print(fmt(row))


def save_csv(rows, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {out_path}")


def main():
    if POOLED_MODE:
        _main_pooled()
    else:
        _main_per_dataset()


def _main_per_dataset():
    rows = []

    for exp_name in EXP_NAMES:
        for dataset_name in DATASETS:
            print("\n" + "=" * 100)
            print(f"Experiment: {exp_name}")
            print(f"Dataset:    {dataset_name}")

            score_file = resolve_score_file(
                exp_name=exp_name,
                dataset_name=dataset_name,
                scores_dir=SCORES_DIR,
                model_name=MODEL_NAME,
            )

            if score_file is None:
                print("Status: missing score file")
                rows.append({
                    "exp_name": exp_name,
                    "dataset": dataset_name,
                    "eer_percent": "NA",
                    "threshold": "NA",
                    "bonafide_mean": "NA",
                    "spoof_mean": "NA",
                    "direction": "missing file",
                })
                continue

            try:
                result = calculate_eer_and_threshold(score_file)

                print(f"EER (%):       {result['eer_percent']:.4f}")
                print(f"Threshold:     {result['threshold']:.6f}")
                print(f"Bonafide mean: {result['bonafide_mean']:.6f}")
                print(f"Spoof mean:    {result['spoof_mean']:.6f}")
                print(f"Direction:     {result['direction']}")

                rows.append({
                    "exp_name": exp_name,
                    "dataset": dataset_name,
                    "eer_percent": f"{result['eer_percent']:.4f}",
                    "threshold": f"{result['threshold']:.6f}",
                    "bonafide_mean": f"{result['bonafide_mean']:.6f}",
                    "spoof_mean": f"{result['spoof_mean']:.6f}",
                    "direction": result["direction"],
                })

            except Exception as e:
                print(f"Status: failed: {e}")
                rows.append({
                    "exp_name": exp_name,
                    "dataset": dataset_name,
                    "eer_percent": "ERR",
                    "threshold": "ERR",
                    "bonafide_mean": "ERR",
                    "spoof_mean": "ERR",
                    "direction": str(e),
                })

    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    print_table(rows)

    if SAVE_CSV and rows:
        save_csv(rows, CSV_OUT)


def _main_pooled():
    all_rows = []

    for exp_name in EXP_NAMES:
        print("\n" + "=" * 100)
        print(f"Experiment (pooled): {exp_name}")
        print(f"Datasets: {DATASETS}")

        try:
            threshold, pooled_eer_pct, higher_is_bonafide, per_ds, n_bona_total, n_spoof_total = \
                calculate_pooled_threshold(exp_name, DATASETS, SCORES_DIR, MODEL_NAME)
        except RuntimeError as e:
            print(f"  [ERROR] {e}")
            continue

        direction = "higher score means bonafide" if higher_is_bonafide else "higher score means spoof"
        print(f"\n  POOLED EER:  {pooled_eer_pct:.4f}%")
        print(f"  THRESHOLD:   {threshold:.6f}")
        print(f"  DIRECTION:   {direction}")
        print(f"  TOTAL BONA:  {n_bona_total}   TOTAL SPOOF: {n_spoof_total}")

        # Per-dataset stats at the pooled threshold
        ds_rows = []
        pooled_correct = 0
        pooled_total = 0

        for ds_info in per_ds:
            ds_name = ds_info["dataset"]
            if ds_info["status"] != "ok":
                ds_rows.append({
                    "dataset": ds_name,
                    "n_bona": "NA",
                    "n_spoof": "NA",
                    "per_eer_pct": "NA",
                    "acc_at_thresh_pct": "NA",
                    "far_pct": "NA",
                    "frr_pct": "NA",
                })
                continue

            labels = ds_info["labels"]
            scores = ds_info["scores"]

            # Per-dataset EER (for reference)
            try:
                bona = scores[labels == "bonafide"]
                spoof = scores[labels == "spoof"]
                per_eer, _ = compute_eer(bona, spoof)
                per_eer_pct = per_eer * 100.0
            except Exception:
                per_eer_pct = float("nan")

            stats = per_dataset_stats_at_threshold(labels, scores, threshold, higher_is_bonafide)
            n_b = ds_info["n_bonafide"]
            n_s = ds_info["n_spoof"]

            # Accumulate for pooled accuracy row
            total = n_b + n_s
            if not np.isnan(stats["acc_pct"]):
                pooled_correct += stats["acc_pct"] / 100.0 * total
                pooled_total += total

            ds_rows.append({
                "dataset": ds_name,
                "n_bona": n_b,
                "n_spoof": n_s,
                "per_eer_pct": f"{per_eer_pct:.4f}",
                "acc_at_thresh_pct": f"{stats['acc_pct']:.2f}",
                "far_pct": f"{stats['far_pct']:.2f}",
                "frr_pct": f"{stats['frr_pct']:.2f}",
            })

        # Pooled accuracy row
        pooled_acc = (pooled_correct / pooled_total * 100.0) if pooled_total > 0 else float("nan")
        ds_rows.append({
            "dataset": "POOLED",
            "n_bona": n_bona_total,
            "n_spoof": n_spoof_total,
            "per_eer_pct": f"{pooled_eer_pct:.4f}",
            "acc_at_thresh_pct": f"{pooled_acc:.2f}",
            "far_pct": "—",
            "frr_pct": "—",
        })

        print()
        print_table(ds_rows)

        # Collect for CSV
        for row in ds_rows:
            all_rows.append({"exp_name": exp_name, "threshold": f"{threshold:.6f}",
                             "direction": direction, **row})

    if SAVE_CSV and all_rows:
        save_csv(all_rows, CSV_OUT)


if __name__ == "__main__":
    main()