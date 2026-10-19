#!/usr/bin/env python3
"""Compare continuous activity strata of two independently trained model/head pairs."""
import argparse
import json
from pathlib import Path


def compare(baseline, candidate):
    rows = []
    for feature in ('z', 'h', 'concat'):
        a, b = [json.loads((root/f'{feature}.json').read_text()) for root in (baseline, candidate)]
        for result in (a, b):
            if result['state_policy'] != 'continuous' or result['feature'] != feature:
                raise ValueError('Expected continuous evaluation with matching feature')
        # Model/head hashes MUST be allowed to differ: these are separate trained models.
        for key in ('dataset', 'role', 'sequences', 'samples_sha256', 'activity_sha256', 'low_activity_threshold'):
            if a[key] != b[key]:
                raise ValueError(f'Comparison mismatch: {feature} {key}')
        if set(a['metrics']) != set(b['metrics']):
            raise ValueError('Metric groups differ')
        for group, am in a['metrics'].items():
            bm = b['metrics'][group]
            for key in ('class_support', 'evaluated_pixels', 'frames'):
                if am[key] != bm[key]:
                    raise ValueError(f'Comparison mismatch: {feature} {group} {key}')
            x, y = am['mIoU'], bm['mIoU']
            rows.append(dict(feature=feature, group=group, baseline_mIoU=x, candidate_mIoU=y,
                             delta_points=None if x is None or y is None else 100*(y-x),
                             pixels=am['evaluated_pixels'], class_support=am['class_support'],
                             baseline_class_IoU=am['class_IoU'], candidate_class_IoU=bm['class_IoU'],
                             baseline_backbone_sha256=a['backbone_sha256'], candidate_backbone_sha256=b['backbone_sha256'],
                             baseline_head_sha256=a['head_sha256'], candidate_head_sha256=b['head_sha256']))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True, help='Existing MODEL/continuous directory')
    p.add_argument('--candidate', type=Path, required=True, help='New MODEL/continuous directory')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rows = compare(a.baseline, a.candidate)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(baseline=str(a.baseline.resolve()), candidate=str(a.candidate.resolve()),
                                     rows=rows), indent=2, allow_nan=False)+'\n')
    print('mIoU(%), difference = candidate - baseline (points)')
    print('feature group                  baseline candidate   delta       pixels')
    def fmt(v, scale=1):
        return '    N/A' if v is None else f'{v*scale:7.3f}'
    for r in rows:
        print(f"{r['feature']:6} {r['group']:22} {fmt(r['baseline_mIoU'],100)} {fmt(r['candidate_mIoU'],100)} {fmt(r['delta_points'])} {r['pixels']:12,}")

if __name__ == '__main__':
    main()
