#!/usr/bin/env python3
"""Choose full evaluation sequences or low/high-activity scenes without GPU inference."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from event_state.evaluation.activity_scenes import select_full_sequences, select_scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activity-manifest", type=Path, required=True)
    parser.add_argument("--dataset", choices=("dsec", "m3ed"), required=True)
    parser.add_argument("--task", choices=("semantic", "depth"), required=True)
    parser.add_argument("--full-sequences", action="store_true",
                        help="Use every native input of every current task evaluation sequence")
    parser.add_argument("--duration-seconds", type=float, help="Activity window duration (default: 10)")
    parser.add_argument("--min-fraction", type=float, help="Minimum low/high fraction (default: 0.8)")
    parser.add_argument("--per-kind", type=int, help="Number of windows per activity kind (default: 1)")
    parser.add_argument("--max-gap-seconds", type=float, help="Maximum window input gap (default: 0.25)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output; preserving {args.output}")
    window_options = {key: getattr(args, key)
                      for key in ("duration_seconds", "min_fraction", "per_kind", "max_gap_seconds")
                      if getattr(args, key) is not None}
    if args.full_sequences:
        if window_options:
            parser.error("--full-sequences cannot be combined with activity-window options")
        result = select_full_sequences(args.activity_manifest, dataset=args.dataset, task=args.task)
    else:
        result = select_scenes(args.activity_manifest, dataset=args.dataset, task=args.task, **window_options)
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
