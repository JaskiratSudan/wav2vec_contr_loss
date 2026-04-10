"""
filter_itw_speaker.py

Filters the In-The-Wild meta.csv to a single speaker and writes
a protocol file compatible with InTheWildDataset.

Usage
-----
    python filter_itw_speaker.py \
        --protocol /nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv \
        --speaker  "Donald Trump" \
        --out_dir  /home/jsudan/wav2vec_contr_loss/famous_figures/splits

Output
------
    Donald_Trump_itw.txt  — same format as meta.csv (filename, speaker, label)
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True,
                        help="Path to ITW meta.csv")
    parser.add_argument("--speaker",  required=True,
                        help="Speaker name as it appears in meta.csv, e.g. 'Donald Trump'")
    parser.add_argument("--out_dir",  required=True,
                        help="Directory to write the filtered protocol file")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize speaker name for filename (spaces → underscores)
    speaker_tag = args.speaker.strip().replace(" ", "_")
    out_path = out_dir / f"{speaker_tag}_itw.txt"

    rows = []
    bonafide = spoof = 0

    with open(args.protocol, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            filename, speaker, label = parts[0], parts[1], parts[2]
            label = label.strip().lower().replace("bona-fide", "bonafide").replace("bonafilde", "bonafide")

            if speaker.strip() != args.speaker.strip():
                continue

            rows.append(f"{filename},{speaker},{label}")
            if label == "bonafide":
                bonafide += 1
            else:
                spoof += 1

    if not rows:
        print(f"No rows found for speaker '{args.speaker}'.")
        print("Check that the name matches exactly (case-sensitive).")
        return

    with open(out_path, "w") as f:
        f.write("\n".join(rows) + "\n")

    print(f"Speaker : {args.speaker}")
    print(f"Bonafide: {bonafide}")
    print(f"Spoof   : {spoof}")
    print(f"Total   : {len(rows)}")
    print(f"Written : {out_path}")


if __name__ == "__main__":
    main()