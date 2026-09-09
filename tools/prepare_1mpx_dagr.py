#!/usr/bin/env python3
"""DAGR-downsample a Prophesee 1Mpx event stream and record padding masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401 - registers common event-file compression filters.
import numpy as np

from event_state.data.dagr_downsample import DAGRDownsampler, bottom_padding_masks


FORMAT_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply DAGR's stateful signed-event 2x downsampling to a 1280x720 "
            "Prophesee HDF5 stream. Events remain 640x360; metadata contains "
            "the masks for bottom-padding the model input to 640x448."
        )
    )
    parser.add_argument("input", type=Path, help="Input *_td.h5")
    parser.add_argument("output", type=Path, help="Output downsampled HDF5")
    parser.add_argument("--chunk-events", type=int, default=1_000_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    if len(values) == 0:
        return
    start = len(dataset)
    dataset.resize(start + len(values), axis=0)
    dataset[start:] = values


def _extend_counts(counts: np.ndarray, milliseconds: np.ndarray) -> np.ndarray:
    if milliseconds.size == 0:
        return counts
    required = int(milliseconds[-1]) + 1
    if required > len(counts):
        counts = np.pad(counts, (0, required - len(counts)))
    counts[:required] += np.bincount(milliseconds, minlength=required).astype(np.uint64)
    return counts


def main() -> None:
    args = parse_args()
    source_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"input HDF5 not found: {source_path}")
    if source_path == output_path:
        raise ValueError("input and output paths must differ")
    if args.chunk_events <= 0:
        raise ValueError("--chunk-events must be positive")
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {output_path}")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    downsampler = DAGRDownsampler()
    pixel_mask, patch_mask, patch_valid_fraction = bottom_padding_masks()
    counts = np.zeros(0, dtype=np.uint64)
    timestamp_offset = 0
    input_events = 0
    output_events = 0

    try:
        with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
            required = ("events/x", "events/y", "events/t", "events/p")
            missing = [name for name in required if name not in source]
            if missing:
                raise KeyError(f"input lacks datasets: {missing}")
            lengths = {len(source[name]) for name in required}
            if len(lengths) != 1:
                raise ValueError("input event datasets have different lengths")
            total = lengths.pop()
            if total:
                timestamp_offset = int(source["events/t"][0])
            output.create_dataset("t_offset", data=timestamp_offset, dtype=np.int64)

            events = output.create_group("events")
            create = {
                "shape": (0,),
                "maxshape": (None,),
                "chunks": True,
                "compression": "gzip",
                "compression_opts": 1,
            }
            out_x = events.create_dataset("x", dtype=np.uint16, **create)
            out_y = events.create_dataset("y", dtype=np.uint16, **create)
            out_t = events.create_dataset("t", dtype=np.uint64, **create)
            out_p = events.create_dataset("p", dtype=np.uint8, **create)

            geometry = output.create_group("geometry")
            geometry.create_dataset("pixel_mask", data=pixel_mask, compression="gzip")
            geometry.create_dataset("patch_mask", data=patch_mask, compression="gzip")
            geometry.create_dataset(
                "patch_valid_fraction", data=patch_valid_fraction, compression="gzip"
            )

            for start in range(0, total, args.chunk_events):
                end = min(total, start + args.chunk_events)
                x = np.asarray(source["events/x"][start:end])
                y = np.asarray(source["events/y"][start:end])
                t = np.asarray(source["events/t"][start:end], dtype=np.uint64)
                p = np.asarray(source["events/p"][start:end])
                reduced_x, reduced_y, reduced_p, selected = downsampler(x, y, p)
                selected_t = t[selected]
                if selected_t.size:
                    relative_t = selected_t - np.uint64(timestamp_offset)
                    counts = _extend_counts(counts, relative_t // np.uint64(1000))
                else:
                    relative_t = selected_t
                _append(out_x, reduced_x)
                _append(out_y, reduced_y)
                _append(out_t, relative_t)
                _append(out_p, (reduced_p > 0).astype(np.uint8))
                input_events += end - start
                output_events += len(reduced_x)
                print(
                    f"\r{end:,}/{total:,} input events; {output_events:,} retained",
                    end="",
                    flush=True,
                )

            # DSEC/DAGR convention: entry m points to the first event at or after m ms.
            ms_to_idx = np.empty(len(counts) + 1, dtype=np.uint64)
            ms_to_idx[0] = 0
            if len(counts):
                np.cumsum(counts, out=ms_to_idx[1:])
            output.create_dataset("ms_to_idx", data=ms_to_idx, compression="gzip")
            metadata = {
                "format_version": FORMAT_VERSION,
                "algorithm": "dagr_stateful_signed_event_downsampling",
                "input_size": [720, 1280],
                "event_size": [360, 640],
                "model_input_size": [448, 640],
                "downsample_factor": [2, 2],
                "padding": {"top": 0, "bottom": 88, "left": 0, "right": 0},
                "patch_size": 16,
                "mask_policy": "patch valid when it contains at least one real pixel",
                "input_events": input_events,
                "output_events": output_events,
                "source": str(source_path),
            }
            output.attrs["event_state_metadata"] = json.dumps(metadata, sort_keys=True)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    print()
    print(f"Saved {output_events:,} events and padding masks to {output_path}")


if __name__ == "__main__":
    main()
