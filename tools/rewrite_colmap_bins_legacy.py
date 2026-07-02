#!/usr/bin/env python3
"""Rewrite COLMAP text models into legacy binary files used by this repo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import colmap_utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-model-dir", required=True, help="Directory containing cameras.txt/images.txt/points3D.txt.")
    parser.add_argument("--output-dir", required=True, help="Directory to write cameras.bin/images.bin/points3D.bin.")
    return parser.parse_args()


def read_images_text_with_points(path: Path):
    images = {}
    with path.open("r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            elems = line.split()
            if len(elems) < 10:
                continue

            image_id = int(elems[0])
            qvec = np.array(tuple(map(float, elems[1:5])))
            tvec = np.array(tuple(map(float, elems[5:8])))
            camera_id = int(elems[8])
            image_name = elems[9]

            points_line = fid.readline().strip()
            if points_line:
                points = points_line.split()
                xys = np.column_stack(
                    [
                        tuple(map(float, points[0::3])),
                        tuple(map(float, points[1::3])),
                    ]
                )
                point3D_ids = np.array(tuple(map(int, points[2::3])), dtype=np.int64)
            else:
                xys = np.empty((0, 2), dtype=np.float64)
                point3D_ids = np.empty((0,), dtype=np.int64)

            images[image_id] = colmap_utils.Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def main() -> int:
    args = parse_args()
    text_model_dir = Path(args.text_model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras = colmap_utils.read_cameras_text(text_model_dir / "cameras.txt")
    images = read_images_text_with_points(text_model_dir / "images.txt")
    points3D = colmap_utils.read_points3D_text(text_model_dir / "points3D.txt")

    colmap_utils.write_cameras_binary(cameras, output_dir / "cameras.bin")
    colmap_utils.write_images_binary(images, output_dir / "images.bin")
    colmap_utils.write_points3D_binary(points3D, output_dir / "points3D.bin")

    print(
        f"rewrote legacy COLMAP bins: cameras={len(cameras)} "
        f"images={len(images)} points3D={len(points3D)} -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
