"""Command-line interface for evaluating a JSONL prediction file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import evaluate_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL file with expected and predicted fields")
    args = parser.parse_args()
    with args.dataset.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    print(json.dumps(evaluate_records(records), indent=2))


if __name__ == "__main__":
    main()

