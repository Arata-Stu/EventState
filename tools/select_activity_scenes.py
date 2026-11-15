#!/usr/bin/env python3
"""Choose continuous low/high-activity video scenes without GPU inference."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from event_state.evaluation.activity_scenes import select_scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("dsec", "m3ed"), required=True)
    parser.add_argument("--task", choices=("semantic", "depth"), required=True)
    parser.add_argument("--duration-seconds", type=float, default=10)
    parser.add_argument("--min-fraction", type=float, default=0.8)
    parser.add_argument("--per-kind", type=int, default=1)
    parser.add_argument("--max-gap-seconds", type=float, default=0.25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output; preserving {args.output}")
    result = select_scenes(args.activity_manifest, dataset=args.dataset, task=args.task,
        duration_seconds=args.duration_seconds, min_fraction=args.min_fraction,
        per_kind=args.per_kind, max_gap_seconds=args.max_gap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {len(result['scenes'])} continuous scenes: {args.output}")
    for scene in result["scenes"]:
        print(f"{scene['id']}: {scene['sequence']} [{scene['start_us']},{scene['end_us']}) "
              f"{scene['duration_us']/1e6:g}s, {scene['frames']} frames, "
              f"low={scene['low_fraction']:.1%}, high={scene['high_fraction']:.1%}")
    for item in result["shortfalls"]:
        print(f"Insufficient {item['kind']} scenes: {item['selected']}/{item['requested']}; "
              "duration and activity thresholds were not relaxed.")


if __name__ == "__main__":
    main()
