# DSEC-Semantic frozen-head evaluation

This protocol measures whether a frozen EventState representation supports
pixel-level semantics.  It does not fine-tune the EventState backbone: only a
segmentation head is trained. The default head is a true linear probe: one
learned 1x1 classifier followed by parameter-free bilinear upsampling. Pass
`--head-type nonlinear` to retain the compact convolutional decoder as a
separate frozen-backbone baseline.

## Protocol

- Classes: the official 11-class labels.
- Development split: six official training sequences for head training and two
  official training sequences for validation. This selects the epoch count.
- Final fit: the head is reinitialized and trained on all eight official train
  sequences for the selected number of epochs.
- Test split: all three official test sequences, untouched until final evaluation.
- Model selection: highest validation mIoU.
- Metrics: mIoU, pixel accuracy, mean class accuracy, and per-class IoU.
- Temporal state: continuous across every frame in a sequence and reset only at
  sequence boundaries.
- Storage: features are written only for frames with semantic labels. Unlabeled
  frames still update recurrent state.

The fixed development split is recorded in
`tools/manifests/dsec_semantic_split.yaml`. Because the official archive has no
validation partition, this two-stage procedure preserves the comparable
8-train/3-test benchmark while avoiding test-driven epoch selection. Test
labels are never used to train or select the head.

## Geometry

DSEC-Semantic masks are 640 x 440. The prepared event tensor is cropped to the
corresponding top 440 rows. If the EventState checkpoint expects 640 x 448,
eight zero rows are added at the bottom instead of vertically resizing the
image. This keeps native pixel coordinates unchanged. The head maps the
stride-16 feature grid back to 640 x 440 logits.

## One-command run

```bash
bash tools/run_dsec_semantic_frozen.sh \
  --checkpoint /path/to/EventState/checkpoint.pt \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --labels-root /path/to/DSEC/task_labels/semantic \
  --feature-cache-dir /path/to/DSEC_cache/semantic_features/MODEL \
  --output-dir outputs/dsec_semantic_frozen/MODEL/seed_0 \
  --feature h \
  --teacher-checkpoint /path/to/dinov3_vits16.pth \
  --gpu 0 \
  --seed 0
```

The run is restartable at sequence granularity during feature export and from
`last.pt` during both head-training stages. Final official-test metrics are
written to `test_metrics.json`.

## Full-sequence visualization

The visualization command renders every labeled frame from all three official
test sequences without temporal subsampling. One PCA basis and one percentile
range are fitted jointly across the selected sequences, so feature colors are
comparable between videos. RGB is displayed only as a visual reference;
semantic predictions consume the cached event feature alone.

```bash
python tools/visualize_dsec_semantic_sequences.py \
  --checkpoint outputs/dsec_semantic_frozen/MODEL/seed_0/final/last.pt \
  --feature-cache-dir /path/to/DSEC_cache/semantic_features/MODEL \
  --event-cache-dir /path/to/DSEC_cache/events/gep_rgb \
  --labels-root /path/to/DSEC/task_labels/semantic \
  --dataset-root /path/to/DSEC \
  --output-dir outputs/dsec_semantic_frozen/MODEL/seed_0/visualization/test \
  --role test \
  --model-label MODEL \
  --device cuda
```

Each sequence produces an MP4 plus a per-frame CSV and a JSON provenance file.
The six panels show aligned RGB reference, GEP event input, normalized feature
PCA, official ground truth, event-only prediction, and a correct/error overlay.
Use `--max-frames 100` for a quick preview, or `--sequences NAME ...` to select
specific sequences.
