#!/usr/bin/env python3
"""Show the published split, or audit a checkpoint before any target/cache work."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_state.dense.protocol import audit_pretraining, load_protocol, load_statistics_provenance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--statistics-provenance", type=Path)
    a = p.parse_args()
    protocol = load_protocol(a.protocol)
    print(json.dumps(protocol, indent=2))
    if a.checkpoint:
        from event_state.training import load_checkpoint_config_metadata
        cfg = load_checkpoint_config_metadata(a.checkpoint).config
        # Check sequence leakage first, so legacy train41 failures are visible
        # even when its original normalization provenance has not been recovered.
        provisional_stats = protocol["splits"]["train"]
        try:
            audit_pretraining(cfg, protocol, statistics_sequences=provisional_stats)
            if a.statistics_provenance is None:
                raise ValueError("Sequence check passed; --statistics-provenance is still required for certification")
            stats = load_statistics_provenance(a.statistics_provenance, cfg)
            print(json.dumps(audit_pretraining(cfg, protocol, statistics_sequences=stats.get("sequences")), indent=2))
        except ValueError as error:
            print(f"AUDIT REJECTED: {error}", file=sys.stderr)
            raise SystemExit(1) from error


if __name__ == "__main__":
    main()
