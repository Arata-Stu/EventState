#!/usr/bin/env python3
"""Summarize native event counts, rates, and observed low/high activity runs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_state.evaluation.activity_counts import print_summary, summarize_activity_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("dsec", "m3ed"), required=True)
    parser.add_argument("--task", choices=("semantic", "depth"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output; preserving {args.output}")
    result = summarize_activity_counts(args.activity_manifest, dataset=args.dataset, task=args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print_summary(result)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
