import os
import csv
import glob
import numpy as np

from evaluation import compute_eer

# -------------------------------------------------
# USER CONFIG
# -------------------------------------------------
EXP_NAMES = [
    "asv5_hbm_1024_warmepo_5_mem_16384_qst_2_supcon_geodesic_temp_0.07",
    "asv5_hbm_1024_mem_16384_supcon_geodesic_temp_0.07",
    "asv5_mem_4096_qst_3_supcon_geodesic_temp_0.07",
    "mem_4096_supcon_temp_0.3",
]

DATASETS = [
    "itw",
    # "asv19",
    # "asv5",
    "asv21_df",
    "asv21_la",
    # "famous_figures",
    # "fakexpose",
    # "mlaad",
    "deepfake_eval_2024",
]

SCORES_DIR = "/home/jsudan/wav2vec_contr_loss/scores"
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"

SAVE_CSV = False
CSV_OUT = "/home/jsudan/wav2vec_contr_loss/one_audio_analysis/eer_threshold_results.csv"


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


if __name__ == "__main__":
    main()