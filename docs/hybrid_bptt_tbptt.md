# DSEC/M3ED hybrid BPTT/TBPTT training

## Semantics

`training.sampling.mode` controls the pretraining input schedule:

- `random`: the existing behavior. Overlapping clips are shuffled and every clip
  starts from an empty recurrent state. Gradients span the frames inside one clip.
- `stream`: non-overlapping clips are read chronologically. Recurrent state is
  carried to the next clip of the same recording and detached at every clip
  boundary (TBPTT).
- `mixed`: one random micro-batch and one stream micro-batch contribute to every
  optimizer update. Their losses are combined with `random_weight` and
  `stream_weight`.

The stream loader tracks state by sequence name rather than by batch position.
It is therefore safe to replace a finished sequence in one batch lane or to emit
a smaller final batch. Stream iterator position and recurrent states are included
in checkpoints for faithful resume.

The final incomplete tail of each training recording is omitted so all samples in
a stream batch have the same temporal length. Evaluation continues to include the
incomplete final clip.

## Augmentation contract

Random clips sample one crop/flip per clip. Stream clips deterministically reuse
one crop/flip for the complete recording within an epoch, so recurrent state is
never carried between incompatible coordinate systems. A new geometry is sampled
when the next stream epoch starts; the random half of mixed training resamples
every clip.

Spatial augmentation requires `teacher.cache_features=false`. In that mode the
same transform is applied to event and RGB tensors and the frozen teacher is run
online on the transformed RGB frames. Cached DINOv3 tokens must not be combined
with stochastic geometry.

## Recommended first ablation

Use `tools/run_dsec_hybrid_augmentation_ablation.sh` to launch:

1. hybrid training without augmentation and with the cached teacher;
2. hybrid training without augmentation and with the online teacher;
3. hybrid training with the online teacher and sequence-consistent augmentation.

Compare (1) with the existing E1 random/no-augmentation checkpoint to measure the
sampling effect. Compare (2) and (3) to measure augmentation without confounding
cached float16 tokens and online teacher inference. Comparing (1) and (2) also
quantifies that teacher-delivery difference directly.

The default branch sizes are four random and four stream sequences, preserving
the former effective local batch size of eight. Because the augmented run uses an
online teacher, compare throughput and GPU memory as well as representation and
downstream metrics.

The mixed schedule is an independent implementation inspired by the public RVT
random/stream sampling design; no runtime dependency on the reference repository
is introduced.

## M3ED

M3ED uses the same sampler, state bank, checkpoint, and augmentation contracts as
DSEC. Its stream clips are read from the prepared M3ED cache in timestamp order.
Set `dataset=m3ed_half_dagr`; `dataset.sequence_length` controls both the random
BPTT clip length and the stream TBPTT truncation interval. The provided M3ED
configuration uses 16 frames by default.

Prepared depth, semantic, or pose targets are not required for representation
pretraining. Stochastic augmentation remains intentionally incompatible with
those aligned downstream targets; use the target-free pretraining dataset when
running the augmentation ablation.

`tools/run_m3ed_hybrid_augmentation_ablation.sh` launches a controlled four-run
comparison. The cached-teacher pair isolates random versus hybrid sampling, and
the online-teacher pair isolates augmentation. The two online runs execute
sequentially on GPU 2, while the cached runs occupy GPUs 0 and 1.
