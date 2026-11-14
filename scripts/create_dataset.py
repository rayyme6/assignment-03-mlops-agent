"""Recreate the public Iris CSV; never overwrite an existing dataset implicitly."""

import argparse
from pathlib import Path

from sklearn.datasets import load_iris


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/iris.csv"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}; choose a new --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    load_iris(as_frame=True).frame.to_csv(args.output, index=False)
    print(f"Created {args.output}; run dvc add and git commit to record this version")


if __name__ == "__main__":
    main()
