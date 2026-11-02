"""Load causal RGB context from the dataset's full (single-sequence) timeline."""

import torch

from event_state.causal_rgb import sampled_history, validate_context


def configure_video_history(dataset, context_frames, input_mode="video", sample_fps=None):
    dataset.teacher_context_frames = int(context_frames)
    dataset.teacher_input_mode = input_mode
    dataset.teacher_sample_fps = sample_fps
    dataset._teacher_timelines = {}
    if not context_frames:
        return
    validate_context(input_mode, context_frames, sample_fps)
    # Even a deterministic random crop with scale=1 can choose a random offset.
    if dataset.transform.training or not dataset.load_images or dataset.feature_cache_dir:
        raise ValueError("Video teacher requires deterministic geometry, online RGB and no teacher cache")


def load_video_history(dataset, sequence_name, start, length):
    k = dataset.teacher_context_frames
    records = dataset._frames_by_sequence[sequence_name]
    if sequence_name not in dataset._teacher_timelines:
        timeline = [int(record.timestamp) for record in records]
        if any(b <= a for a, b in zip(timeline, timeline[1:])):
            raise ValueError("Teacher RGB timeline must be strictly increasing")
        dataset._teacher_timelines[sequence_name] = timeline
    timeline = dataset._teacher_timelines[sequence_name]
    positions, requests, prefix = sampled_history(
        timeline, start, length, mode=dataset.teacher_input_mode,
        context_frames=k, sample_fps=dataset.teacher_sample_fps,
    )
    unique = sorted({i for row in positions for i in row})
    lookup = {i: j for j, i in enumerate(unique)}
    # Decode prefix-padding duplicates only once. Geometry is applied jointly
    # with events in the existing PairedSequenceTransform, including crop.
    images = torch.stack([dataset._load_image(records[i].image_path) for i in unique])
    metadata = {
        "teacher_indices": torch.tensor([[lookup[i] for i in row] for row in positions]),
        "teacher_frame_indices": torch.tensor([[records[i].frame_index for i in row] for row in positions]),
        "teacher_timestamps": torch.tensor([[timeline[i] for i in row] for row in positions]),
        "teacher_requested_timestamps": torch.tensor(requests),
        "teacher_prefix_padding": torch.tensor(prefix, dtype=torch.bool),
        "teacher_image_count": len(unique),
        "teacher_sequence_start_timestamp": timeline[0],
    }
    return images, metadata


def collate_teacher_history(samples):
    """Pad only unique RGB buffers; never replicate B*T*K full-resolution frames."""
    from torch.utils.data import default_collate
    from torch.nn.utils.rnn import pad_sequence

    if "teacher_images" not in samples[0]:
        return default_collate(samples)
    images = pad_sequence([sample["teacher_images"] for sample in samples], batch_first=True)
    result = default_collate([{k: v for k, v in sample.items() if k != "teacher_images"}
                              for sample in samples])
    result["teacher_images"] = images
    return result
