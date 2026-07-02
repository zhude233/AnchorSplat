#!/usr/bin/env python3
"""Prepare SplatFormer layout that initializes gsplat from SuperGaussian xyz/rgb."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import colmap_utils


DEFAULT_SOURCE_ROOT = "data/mvimgnet_testset_500"
DEFAULT_LAYOUT_ROOT = "data/mvimgnet_splatformer_ready"
DEFAULT_OUTPUT_ROOT = "data/mvimgnet_xyzinit_splatformer_ready"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--layout-root", default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene-list", help="Optional file with category/scene or category__scene entries.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file")
    parser.add_argument("--point-count", type=int, default=131072)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument(
        "--supergaussian-root",
        default=None,
        help="Repo root to add to PYTHONPATH for unpickling SuperGaussian classes.",
    )
    return parser.parse_args()


def configure_unpickle_imports(source_root: Path, explicit_root: Optional[str]) -> Optional[Path]:
    candidates = []
    if explicit_root:
        candidates.append(Path(explicit_root).expanduser().resolve())
    candidates.extend([source_root, *source_root.parents])
    for candidate in candidates:
        if (candidate / "sg_utils").is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate
    return None


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def iter_scene_entries(source_root: Path, scene_list: Optional[Path]) -> Iterable[Tuple[Path, str, str]]:
    if scene_list is not None:
        with scene_list.open("r") as f:
            entries = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        for entry in entries:
            if "__" in entry and "/" not in entry:
                category, scene = entry.split("__", 1)
                yield source_root / category / scene, category, scene
            else:
                scene_path = Path(entry)
                if not scene_path.is_absolute():
                    scene_path = source_root / scene_path
                yield scene_path, scene_path.parent.name, scene_path.name
        return

    if (source_root / "xyz.pkl").exists():
        yield source_root, source_root.parent.name, source_root.name
        return

    for category_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            yield scene_dir, category_dir.name, scene_dir.name


def scene_output_name(category: str, scene: str) -> str:
    return f"{category}__{scene}"


def copy_required_layout(layout_scene: Path, output_scene: Path) -> None:
    # Do not copy gaussian_splatting or colmap_work; this experiment must train fresh ckpts.
    required = [
        "256/images",
        "256/cameras.txt",
        "256/images.txt",
        "256/sparse/0/cameras.bin",
        "256/sparse/0/images.bin",
        "1024/images",
        "1024/cameras.txt",
        "1024/images.txt",
    ]
    for rel in required:
        src = layout_scene / rel
        dst = output_scene / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing required layout path: {src}")
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    optional = ["256/sparse/0/cameras.txt", "256/sparse/0/images.txt"]
    for rel in optional:
        src = layout_scene / rel
        if src.exists():
            dst = output_scene / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def normalize_rgb(rgb_np: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_np, dtype=np.float64)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def build_points3d(xyz_np: np.ndarray, rgb_np: np.ndarray) -> Dict[int, colmap_utils.Point3D]:
    xyz = np.asarray(xyz_np, dtype=np.float64)
    rgb = normalize_rgb(rgb_np)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"rgb shape {rgb.shape} does not match xyz shape {xyz.shape}")

    points = {}
    empty_image_ids = np.empty((0,), dtype=np.int32)
    empty_point2d_idxs = np.empty((0,), dtype=np.int32)
    for idx, (point, color) in enumerate(zip(xyz, rgb), start=1):
        points[idx] = colmap_utils.Point3D(
            id=idx,
            xyz=point.astype(np.float64),
            rgb=color.astype(np.uint8),
            error=0.0,
            image_ids=empty_image_ids,
            point2D_idxs=empty_point2d_idxs,
        )
    return points


def write_points(points: Dict[int, colmap_utils.Point3D], sparse_dir: Path) -> None:
    sparse_dir.mkdir(parents=True, exist_ok=True)
    colmap_utils.write_points3D_text(points, sparse_dir / "points3D.txt")
    colmap_utils.write_points3D_binary(points, sparse_dir / "points3D.bin")


def process_scene(scene_dir: Path, category: str, scene: str, args: argparse.Namespace) -> Dict[str, Any]:
    out_name = scene_output_name(category, scene)
    layout_scene = Path(args.layout_root) / out_name
    output_scene = Path(args.output_root) / out_name
    result: Dict[str, Any] = {
        "source_scene": str(scene_dir),
        "layout_scene": str(layout_scene),
        "output_scene": str(output_scene),
        "status": "skipped",
        "reason": None,
    }

    if output_scene.exists() and not args.overwrite:
        result["reason"] = "output exists; pass --overwrite"
        return result

    xyz_path = scene_dir / "xyz.pkl"
    rgb_path = scene_dir / "rgb.pkl"
    if not xyz_path.exists() or not rgb_path.exists():
        result["reason"] = "missing xyz.pkl or rgb.pkl"
        return result
    if not layout_scene.exists():
        result["reason"] = "missing converted layout scene"
        return result

    xyz_np = load_pickle(xyz_path)
    rgb_np = load_pickle(rgb_path)
    raw_xyz = np.asarray(xyz_np)
    raw_rgb = np.asarray(rgb_np)
    xyz = np.asarray(xyz_np).astype(np.float32)
    rgb = np.asarray(rgb_np)
    raw_num_points = len(xyz)
    rng = np.random.default_rng(args.sample_seed) if args.sample_seed is not None else np.random.default_rng()
    replace = len(xyz) < args.point_count
    ind = rng.choice(len(xyz), args.point_count, replace=replace)
    xyz = xyz[ind]
    rgb = rgb[ind]

    points = build_points3d(xyz, rgb)
    result["raw_num_points"] = raw_num_points
    result["num_points"] = len(points)
    result["point_count"] = args.point_count
    result["replace"] = bool(replace)
    result["sample_seed"] = args.sample_seed
    result["raw_xyz_min"] = float(raw_xyz.min())
    result["raw_xyz_max"] = float(raw_xyz.max())
    result["raw_rgb_max"] = float(raw_rgb.max())
    result["xyz_min"] = float(xyz.min())
    result["xyz_max"] = float(xyz.max())
    result["rgb_max"] = float(rgb.max())

    if args.dry_run:
        result["status"] = "processed"
        result["dry_run"] = True
        return result

    if output_scene.exists():
        shutil.rmtree(output_scene)
    output_scene.mkdir(parents=True, exist_ok=True)
    copy_required_layout(layout_scene, output_scene)
    write_points(points, output_scene / "256" / "sparse" / "0")
    result["status"] = "processed"
    return result


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    scene_list = Path(args.scene_list).expanduser().resolve() if args.scene_list else None
    sg_root = configure_unpickle_imports(source_root, args.supergaussian_root)

    results = []
    for idx, (scene_dir, category, scene) in enumerate(iter_scene_entries(source_root, scene_list)):
        if args.limit is not None and idx >= args.limit:
            break
        try:
            results.append(process_scene(scene_dir, category, scene, args))
        except Exception as exc:
            results.append(
                {
                    "source_scene": str(scene_dir),
                    "output_scene": scene_output_name(category, scene),
                    "status": "skipped",
                    "reason": f"unexpected error: {exc}",
                }
            )

    summary = {
        "source_root": str(source_root),
        "layout_root": str(Path(args.layout_root).expanduser().resolve()),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
        "supergaussian_root": str(sg_root) if sg_root is not None else None,
        "point_count": args.point_count,
        "sample_seed": args.sample_seed,
        "processed": sum(r["status"] == "processed" for r in results),
        "skipped": sum(r["status"] != "processed" for r in results),
        "results": results,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
