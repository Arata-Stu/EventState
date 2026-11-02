#!/usr/bin/env python3
"""Resume one unchanged dense pipeline command, preserving incomplete artifacts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(command):
    script = Path(command[1]).name
    def arg(key):
        return command[command.index(key) + 1]
    if script == 'cache_dense_features.py':
        out = Path(arg('--output-dir'))
        if out.exists():
            if not (out / '_SUCCESS').is_file():
                raise RuntimeError(f'Incomplete cache preserved: {out}. Move it aside before retrying.')
            meta = json.loads((out / 'metadata.json').read_text())
            for key, expected in [('checkpoint_sha256', sha(arg('--checkpoint'))),
                                  ('target_index_sha256', sha(arg('--target-index'))),
                                  ('state_policy', arg('--state-policy'))]:
                if meta[key] != expected:
                    raise ValueError(f'Cache provenance mismatch: {key}: {out}')
            if any(not Path(r['feature_path']).is_file() for r in meta['records']):
                raise ValueError(f'Missing cached features: {out}')
            return None
    elif script == 'train_dense_probe.py':
        out = Path(arg('--output-dir'))
        if out.exists():
            # Trainer validates full settings even for a completed head.
            return command + ['--resume']
    elif script == 'evaluate_dense_probe.py':
        out = Path(arg('--output'))
        if out.exists():
            report = json.loads(out.read_text())
            if (report['head_sha256'] != sha(arg('--checkpoint')) or
                    report['cache_metadata_sha256'] != sha(Path(arg('--cache')) / 'metadata.json')):
                raise ValueError(f'Evaluation provenance mismatch: {out}')
            return None
    elif script == 'compare_dense_probes.py':
        out = Path(arg('--output'))
        if out.exists():
            from compare_dense_probes import compare
            value = compare(json.loads(Path(arg('--baseline')).read_text()),
                            json.loads(Path(arg('--candidate')).read_text()), arg('--mode'))
            old = json.loads(out.read_text())
            if old['differences'] != value['differences']:
                raise ValueError(f'Comparison changed: {out}')
            return None
    return command


if __name__ == '__main__':
    command = prepare(sys.argv[1:])
    if command is None:
        print('[resume] verified existing output; skip', flush=True)
    else:
        subprocess.run(command, check=True)
