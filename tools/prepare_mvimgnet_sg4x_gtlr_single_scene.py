#!/usr/bin/env python3
"""Prepare one sg4x scene whose low-res input is downsampled from GT-HR renders."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


DEFAULT_LAYOUT_SCENE = (
    "data/mvimgnet_sg4x_splatformer_ready/0__0000f9c2"
)
DEFAULT_SOURCE_SCENE = "data/mvimgnet_testset_500/0/0000f9c2"
DEFAULT_OUTPUT_ROOT = (
    "data/mvimgnet_sg4x_gtlr_splatformer_single"
)
DEFAULT_SCENE_NAME = "0__0000f9c2"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
REQUIRED_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an existing sg4x SplatFormer scene layout, but replace 256/images "
            "with 64x64 bilinear downsampled HR_131072_gaussian renders."
        )
    )
    parser.add_argument("--layout-scene", default=DEFAULT_LAYOUT_SCENE)
    parser.add_argument("--source-scene", default=DEFAULT_SOURCE_SCENE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene-name", default=DEFAULT_SCENE_NAME)
    parser.add_argument("--hr-image-dir", default="HR_131072_gaussian")
    parser.add_argument("--low-size", type=int, default=64)
    parser.add_argument("--high-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_colmap_image_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Malformed image record in {path}: {line}")
        records.append(
            {
                "image_id": int(parts[0]),
                "qvec": tuple(float(value) for value in parts[1:5]),
                "tvec": tuple(float(value) for value in parts[5:8]),
                "camera_id": int(parts[8]),
                "name": parts[9],
                "line": line,
            }
        )
        index += 2
    return records


def read_single_camera(path: Path) -> dict[str, Any]:
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 8:
            raise ValueError(f"Malformed camera record in {path}: {line}")
        return {
            "camera_id": int(parts[0]),
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": tuple(float(value) for value in parts[4:8]),
            "line": stripped,
        }
    raise ValueError(f"No camera record found in {path}")


def find_image(source_dir: Path, name: str) -> Path:
    candidate = source_dir / name
    if candidate.is_file():
        return candidate

    stem = Path(name).stem
    for extension in IMAGE_EXTENSIONS:
        candidate = source_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate

    matches = sorted(path for path in source_dir.glob(f"{stem}.*") if path.suffix in IMAGE_EXTENSIONS)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Missing HR source image for {name} in {source_dir}")


def copy_required_layout(layout_scene: Path, output_scene: Path) -> None:
    for size in ("256", "1024"):
        output_size_dir = output_scene / size
        output_size_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("cameras.txt", "images.txt"):
            shutil.copy2(layout_scene / size / filename, output_size_dir / filename)

    shutil.copytree(layout_scene / "1024" / "images", output_scene / "1024" / "images")

    sparse_src = layout_scene / "256" / "sparse" / "0"
    sparse_dst = output_scene / "256" / "sparse" / "0"
    sparse_dst.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_SPARSE_FILES:
        shutil.copy2(sparse_src / filename, sparse_dst / filename)


def write_low_images_from_hr(
    hr_dir: Path,
    output_images_dir: Path,
    image_names: list[str],
    low_size: int,
) -> list[dict[str, Any]]:
    output_images_dir.mkdir(parents=True, exist_ok=True)
    resampling = getattr(Image, "Resampling", None)
    bilinear = resampling.BILINEAR if resampling is not None else Image.BILINEAR
    written: list[dict[str, Any]] = []

    for name in image_names:
        source = find_image(hr_dir, name)
        destination = output_images_dir / name
        with Image.open(source) as image:
            source_size = [image.width, image.height]
            resized = image.resize((low_size, low_size), resample=bilinear)
            resized.save(destination)
        written.append({"name": name, "source": str(source), "source_size": source_size})
    return written


def assert_file_sets_match(directory: Path, expected_names: list[str]) -> None:
    expected = set(expected_names)
    actual = {path.name for path in directory.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(f"Image set mismatch in {directory}: missing={missing[:5]} extra={extra[:5]}")


def assert_image_sizes(directory: Path, expected_names: list[str], width: int, height: int) -> None:
    for name in expected_names:
        with Image.open(directory / name) as image:
            if image.width != width or image.height != height:
                raise AssertionError(
                    f"Unexpected image size for {directory / name}: "
                    f"{image.width}x{image.height}, expected {width}x{height}"
                )


def assert_cameras(low_camera: dict[str, Any], high_camera: dict[str, Any]) -> None:
    if low_camera["model"] != "PINHOLE" or high_camera["model"] != "PINHOLE":
        raise AssertionError(f"Expected PINHOLE cameras, got {low_camera['model']} and {high_camera['model']}")
    if (low_camera["width"], low_camera["height"]) != (64, 64):
        raise AssertionError(f"Expected low camera 64x64, got {low_camera['width']}x{low_camera['height']}")
    if (high_camera["width"], high_camera["height"]) != (256, 256):
        raise AssertionError(f"Expected high camera 256x256, got {high_camera['width']}x{high_camera['height']}")

    for low_value, high_value in zip(low_camera["params"], high_camera["params"]):
        if not math.isclose(high_value, low_value * 4.0, rel_tol=1e-6, abs_tol=1e-5):
            raise AssertionError(
                "Expected 1024/cameras.txt params to be 4x 256/cameras.txt params, "
                f"got low={low_camera['params']} high={high_camera['params']}"
            )


def assert_image_records(low_records: list[dict[str, Any]], high_records: list[dict[str, Any]]) -> None:
    low_by_name = {record["name"]: record for record in low_records}
    high_by_name = {record["name"]: record for record in high_records}
    if set(low_by_name) != set(high_by_name):
        raise AssertionError(
            "256/images.txt and 1024/images.txt name sets differ: "
            f"low_only={sorted(set(low_by_name) - set(high_by_name))[:5]} "
            f"high_only={sorted(set(high_by_name) - set(low_by_name))[:5]}"
        )
    for name, low_record in low_by_name.items():
        high_record = high_by_name[name]
        if low_record["qvec"] != high_record["qvec"] or low_record["tvec"] != high_record["tvec"]:
            raise AssertionError(f"Pose mismatch between low/high images.txt for {name}")


def validate_output(output_scene: Path, low_size: int, high_size: int) -> dict[str, Any]:
    low_camera = read_single_camera(output_scene / "256" / "cameras.txt")
    high_camera = read_single_camera(output_scene / "1024" / "cameras.txt")
    assert_cameras(low_camera, high_camera)

    low_records = read_colmap_image_records(output_scene / "256" / "images.txt")
    high_records = read_colmap_image_records(output_scene / "1024" / "images.txt")
    assert_image_records(low_records, high_records)

    image_names = [record["name"] for record in low_records]
    assert_file_sets_match(output_scene / "256" / "images", image_names)
    assert_file_sets_match(output_scene / "1024" / "images", image_names)
    assert_image_sizes(output_scene / "256" / "images", image_names, low_size, low_size)
    assert_image_sizes(output_scene / "1024" / "images", image_names, high_size, high_size)

    sparse_dir = output_scene / "256" / "sparse" / "0"
    missing_sparse = [filename for filename in REQUIRED_SPARSE_FILES if not (sparse_dir / filename).exists()]
    if missing_sparse:
        raise AssertionError(f"Missing sparse files in {sparse_dir}: {missing_sparse}")

    return {
        "low_camera": low_camera,
        "high_camera": high_camera,
        "num_images": len(image_names),
        "image_names": image_names,
        "sparse_files": [str(sparse_dir / filename) for filename in REQUIRED_SPARSE_FILES],
    }


def prepare_scene(args: argparse.Namespace) -> dict[str, Any]:
    layout_scene = Path(args.layout_scene).expanduser().resolve()
    source_scene = Path(args.source_scene).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    output_scene = output_root / args.scene_name
    hr_dir = source_scene / args.hr_image_dir

    if not layout_scene.is_dir():
        raise FileNotFoundError(f"Missing layout scene: {layout_scene}")
    if not hr_dir.is_dir():
        raise FileNotFoundError(f"Missing HR image directory: {hr_dir}")
    if output_scene.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output scene exists; pass --overwrite to regenerate: {output_scene}")
        shutil.rmtree(output_scene)

    copy_required_layout(layout_scene, output_scene)
    low_records = read_colmap_image_records(output_scene / "256" / "images.txt")
    image_names = [record["name"] for record in low_records]
    low_images = write_low_images_from_hr(hr_dir, output_scene / "256" / "images", image_names, args.low_size)
    validation = validate_output(output_scene, args.low_size, args.high_size)

    summary = {
        "layout_scene": str(layout_scene),
        "source_scene": str(source_scene),
        "hr_image_dir": str(hr_dir),
        "output_root": str(output_root),
        "output_scene": str(output_scene),
        "low_generation": {
            "source": "HR_131072_gaussian",
            "target_size": [args.low_size, args.low_size],
            "resampling": "BILINEAR",
            "num_images": len(low_images),
            "first_image": low_images[0] if low_images else None,
        },
        "target_generation": {
            "source": "copied from existing sg4x layout 1024/images",
            "target_size": [args.high_size, args.high_size],
        },
        "validation": validation,
    }
    summary_path = output_root / f"prepare_{args.scene_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    args = parse_args()
    summary = prepare_scene(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
