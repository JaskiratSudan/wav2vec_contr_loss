#!/usr/bin/env python3
"""Print EER from a single score file."""

import argparse
import os

from evaluation import calculate_EER


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print EER from a single score file."
    )
    parser.add_argument(
        "score_path",
        help="Path to score file (format: <utt_id> <source> <key> <score>).",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Number of decimal places to print (default: 4).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.score_path):
        raise SystemExit(f"Score file not found: {args.score_path}")

    eer = calculate_EER(args.score_path)
    fmt = f"{{:.{args.precision}f}}"
    print(f"EER: {fmt.format(eer)}")


if __name__ == "__main__":
    main()
