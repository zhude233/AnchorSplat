#!/usr/bin/env python3
"""Run gsplat simple_trainer with a temporary SuperGaussian scale clamp patch."""

from __future__ import annotations

import os
import runpy
import sys
import tempfile
from pathlib import Path


DEFAULT_SOURCE_TRAINER = "third_party/gsplat/examples/simple_trainer.py"
SOURCE_TRAINER_ENV = "SIMPLE_TRAINER_PATH"

ORIGINAL_LINE = "dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]"
PATCHED_LINE = "dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1).clamp_min(1e-7)  # [N,]"


def parse_wrapper_args(argv: list[str]) -> tuple[Path, list[str]]:
    source_trainer = os.environ.get(SOURCE_TRAINER_ENV, DEFAULT_SOURCE_TRAINER)
    trainer_argv: list[str] = []

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--source-trainer":
            i += 1
            if i >= len(argv):
                raise SystemExit("error: --source-trainer requires a path")
            source_trainer = argv[i]
        elif arg.startswith("--source-trainer="):
            source_trainer = arg.split("=", 1)[1]
            if not source_trainer:
                raise SystemExit("error: --source-trainer requires a non-empty path")
        else:
            trainer_argv.append(arg)
        i += 1

    return Path(source_trainer).expanduser().resolve(), trainer_argv


def patch_trainer_text(source_path: Path) -> str:
    text = source_path.read_text()

    if PATCHED_LINE in text:
        raise SystemExit(
            f"error: source trainer already contains the scale clamp patch: {source_path}"
        )

    count = text.count(ORIGINAL_LINE)
    if count != 1:
        raise SystemExit(
            "error: expected exactly one unpatched scale initialization line in "
            f"{source_path}, found {count}"
        )

    return text.replace(ORIGINAL_LINE, PATCHED_LINE, 1)


def run_patched_trainer(source_path: Path, trainer_argv: list[str]) -> None:
    patched_text = patch_trainer_text(source_path)

    with tempfile.TemporaryDirectory(prefix="simple_trainer_sg_scale_clamp_") as tmpdir:
        patched_path = Path(tmpdir) / source_path.name
        patched_path.write_text(patched_text)

        sys.path.insert(0, str(source_path.parent))
        sys.argv = [str(patched_path), *trainer_argv]
        runpy.run_path(str(patched_path), run_name="__main__")


def main() -> None:
    source_path, trainer_argv = parse_wrapper_args(sys.argv[1:])
    run_patched_trainer(source_path, trainer_argv)


if __name__ == "__main__":
    main()
