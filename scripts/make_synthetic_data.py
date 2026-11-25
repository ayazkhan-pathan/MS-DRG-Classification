"""Write obviously fake note CSVs for a smoke run of drg train, evaluate and predict.

Usage:
    python scripts/make_synthetic_data.py --out-dir data/synthetic
    python scripts/make_synthetic_data.py --out-dir data/synthetic --train 400 --seed 1

The notes are invented words labelled S01 to S04 (see src/drg/synthetic.py). Scores on them
only show that the pipeline runs.
"""

from __future__ import annotations

import argparse

from drg.synthetic import write_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="data/synthetic")
    parser.add_argument("--train", type=int, default=200, help="training notes")
    parser.add_argument("--val", type=int, default=60, help="validation notes")
    parser.add_argument("--test", type=int, default=60, help="test notes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    paths = write_splits(args.out_dir, args.train, args.val, args.test, args.seed)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
