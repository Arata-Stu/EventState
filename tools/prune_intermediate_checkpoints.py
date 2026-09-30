#!/usr/bin/env python3
"""Prune dense intermediate saves from explicitly selected completed runs.

Dry-run by default. Keeps 10k milestones, latest, final, best, all non-step
files and symlinks. Does not touch feature caches or downstream head weights.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import zipfile
from pathlib import Path

STEP = re.compile(r"step_(\d{8})\.pt")


def plan(roots: list[Path], outputs: Path) -> list[Path]:
    candidates: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root == outputs or not root.is_relative_to(outputs) or not root.is_dir():
            raise ValueError(f"Select an existing run strictly inside {outputs}: {root}")
        for directory, subdirs, _files in os.walk(root, followlinks=False):
            parent = Path(directory)
            subdirs[:] = [d for d in subdirs if not (parent / d).is_symlink()]
            if parent.name != "checkpoints":
                continue
            final = parent / "step_00100000.pt"
            # Cheap container check, not a model-load verification.
            if final.is_symlink() or not final.is_file() or not zipfile.is_zipfile(final):
                print(f"SKIP (no valid final ZIP container): {parent}")
                continue
            with zipfile.ZipFile(final) as archive:
                if not any(n.endswith("/data.pkl") for n in archive.namelist()):
                    print(f"SKIP (no checkpoint metadata): {parent}")
                    continue
            steps = [(p, int(m[1])) for p in parent.iterdir()
                     if (m := STEP.fullmatch(p.name)) and p.is_file() and not p.is_symlink()]
            latest = max(n for _, n in steps)
            candidates.update(p for p, n in steps
                              if 0 < n < 100000 and n != latest and n % 10000 != 0)
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    outputs = (Path(__file__).resolve().parents[1] / "outputs").resolve()
    candidates = plan(args.runs, outputs)
    records = []
    groups: dict[str, tuple[int, int]] = {}
    for path in candidates:
        st = path.stat()
        records.append({"path": str(path), "size": st.st_size,
                        "mtime_ns": st.st_mtime_ns, "inode": st.st_ino})
        count, size = groups.get(str(path.parent), (0, 0))
        groups[str(path.parent)] = count + 1, size + st.st_size
    for name, (count, size) in groups.items():
        print(f"{size / 2**30:.2f} GiB  {count} files  {name}")
    print(f"TOTAL: {len(records)} files, {sum(r['size'] for r in records) / 2**30:.2f} GiB logical size")
    print("KEEP: every 10k step, final, latest, best, logs, metrics, caches, other files")
    if not args.apply:
        print("Dry-run only. Use --apply to delete these intermediate files.")
        return
    audit = outputs / f"checkpoint_prune_{time.time_ns()}.jsonl"
    # Persist the candidate list before removing anything; flush each outcome.
    with audit.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps({"planned": records}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        for record in records:
            path = Path(record["path"])
            st = path.lstat()
            if path.is_symlink() or (st.st_size, st.st_mtime_ns, st.st_ino) != (
                record["size"], record["mtime_ns"], record["inode"]
            ):
                raise RuntimeError(f"File changed after planning; stopping: {path}")
            path.unlink()
            handle.write(json.dumps({"deleted": str(path)}) + "\n")
            handle.flush()
    print(f"Deletion audit: {audit}")


if __name__ == "__main__":
    main()
