#!/usr/bin/env python3
"""Plan a causal V-JEPA teacher run from an audited resolved DINO config."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from event_state.vjepa_plan import make_plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", required=True, type=Path, help="Existing resolved_config.yaml")
    p.add_argument("--teacher-repo", required=True, type=Path, help="Local official vjepa2 Git checkout")
    p.add_argument("--teacher-revision", required=True, help="Full 40-character pinned commit")
    p.add_argument("--teacher-checkpoint", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--stage", choices=("smoke", "full"), default="smoke")
    p.add_argument("--input-mode", choices=("image", "video"), required=True)
    p.add_argument("--context-frames", type=int, help="Video default 16; image must be 1")
    p.add_argument("--sample-fps", type=float, help="Video default 4; not applicable to image mode")
    p.add_argument("--teacher-micro-batch", type=int, default=1)
    p.add_argument("--batch-size", type=int, help="Omit to preserve base experiment batch size")
    p.add_argument("--execute", action="store_true", help="Otherwise print plan only")
    a = p.parse_args()
    if a.input_mode == "image" and (a.sample_fps is not None or a.context_frames not in (None, 1)):
        p.error("Image mode uses exactly one current frame, without --sample-fps")
    context_frames = a.context_frames if a.context_frames is not None else (1 if a.input_mode == "image" else 16)
    sample_fps = (a.sample_fps if a.sample_fps is not None else 4.0) if a.input_mode == "video" else None
    import yaml
    if a.output_dir.exists():
        raise FileExistsError("Choose a fresh output directory; this runner never overwrites a run")
    if len(a.teacher_revision) != 40 or a.teacher_micro_batch < 1:
        raise ValueError("Full commit and positive teacher micro-batch required")
    teacher = yaml.safe_load((PROJECT / "configs/teacher/vjepa2_1_vit_base_384.yaml").read_text())
    digest = hashlib.sha256()
    with a.teacher_checkpoint.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    teacher.update(repository=str(a.teacher_repo.resolve()), revision=a.teacher_revision,
                   checkpoint=str(a.teacher_checkpoint.resolve()), checkpoint_sha256=digest.hexdigest(),
                   input_mode=a.input_mode, context_frames=context_frames, sample_fps=sample_fps,
                   feature="image_last_layer" if a.input_mode == "image" else "trailing_last_tubelet",
                   micro_batch_size=a.teacher_micro_batch)
    config = make_plan(yaml.safe_load(a.base_config.read_text()), teacher,
                       output=a.output_dir.resolve(), stage=a.stage, batch_size=a.batch_size)
    print(yaml.safe_dump(config, sort_keys=False), flush=True)
    if not a.execute:
        print("Plan only. Use --execute to run on the training server.")
        return
    from omegaconf import OmegaConf
    from event_state.training.factory import validate_config
    from event_state.models.teacher.vjepa21 import verify_source
    validate_config(OmegaConf.create(config))
    verify_source(teacher["repository"], teacher["revision"], teacher["checkpoint"], teacher["checkpoint_sha256"])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    (a.output_dir / "vjepa_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    command = [sys.executable, "-u", "train.py", "--config-path", str(a.output_dir.resolve()),
               "--config-name", "vjepa_config"]
    # Preserve raw output for loss/throughput/OOM diagnosis. No feature cache.
    with (a.output_dir / "train.log").open("w") as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    final = a.output_dir / "checkpoints" / f"step_{config['training']['max_steps']:08d}.pt"
    if not final.is_file() or final.stat().st_size == 0:
        raise RuntimeError(f"Training exited without final checkpoint: {final}")
    print(f"[vjepa21] complete: {final}")


if __name__ == "__main__":
    main()
