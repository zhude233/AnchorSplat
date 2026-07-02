#!/usr/bin/env python3
"""Lightweight validation for MVImgNet low-res 3DGS preparation and conversion."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import colmap_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", required=True, help="Prepared MVImgNet low-res 3DGS scene dir.")
    parser.add_argument("--ckpt", help="Optional converted SplatFormer ckpt to validate.")
    parser.add_argument("--splatformer-scene-root", help="Optional SplatFormer scene root to validate.")
    parser.add_argument("--prepare-summary", help="Optional prepare_mvimgnet_lowres_3dgs.py JSON summary/log.")
    parser.add_argument(
        "--skip-splatformer-ckpt",
        action="store_true",
        help="Validate raw SplatFormer layout before the converted ckpt has been written.",
    )
    parser.add_argument(
        "--no-gt-eval",
        type=int,
        default=1,
        help="Expected SuperGaussian train.py --no_gt_eval value. Default: 1 for prepared MVImgNet low-res scenes.",
    )
    parser.add_argument(
        "--init-expected-count",
        type=int,
        default=131072,
        help="Expected point count for the prepared surface init PLY. Default: 131072.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help=(
            "Optional expected Gaussian count for converted/final ckpts. "
            "Unset allows trainer densification to change the final count."
        ),
    )
    parser.add_argument(
        "--allow-traj0",
        action="store_true",
        help="Allow transforms.json frame stems containing traj_0. Default fails because the fork treats them as test split.",
    )
    parser.add_argument("--train-output", help="Optional fork train output dir for render-vs-LR checks.")
    parser.add_argument("--iteration", type=int, help="Iteration folder to inspect under train output.")
    parser.add_argument("--max-image-pairs", type=int, default=8)
    parser.add_argument("--log-file", help="Optional JSON summary path.")
    return parser.parse_args()


def fail(message: str) -> None:
    raise ValueError(message)


def image_to_float(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def validate_resolution_low(
    scene_dir: Path,
    init_expected_count: int,
    no_gt_eval: int,
    allow_traj0: bool,
) -> Dict[str, object]:
    resolution_low = scene_dir / "resolution_low"
    transforms_path = resolution_low / "transforms.json"
    images_dir = resolution_low / "images"
    pcd_path = resolution_low / f"surface_pcd_{init_expected_count}_seed_0.ply"
    gt_dir = resolution_low / "gt" / "high_res_images"
    if not transforms_path.is_file():
        fail(f"missing {transforms_path}")
    if not images_dir.is_dir():
        fail(f"missing {images_dir}")
    image_files = sorted(path for path in images_dir.iterdir() if path.is_file())
    if not image_files:
        fail(f"no LR images under {images_dir}")
    if not pcd_path.is_file():
        fail(f"missing {pcd_path}")
    if no_gt_eval != 1 and not gt_dir.is_dir():
        fail(f"--no_gt_eval {no_gt_eval} requires missing GT dir {gt_dir}")

    meta = json.loads(transforms_path.read_text())
    for key in ("fl_x", "fl_y", "cx", "cy", "w", "h", "frames"):
        if key not in meta:
            fail(f"transforms.json missing {key}")
    frames = meta["frames"]
    if not frames:
        fail("transforms.json has no frames")

    missing_images: List[str] = []
    missing_gt_images: List[str] = []
    bad_sizes: List[str] = []
    traj_0_frames: List[str] = []
    for frame in frames:
        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            fail("frame missing non-empty file_path")
        if Path(file_path).is_absolute():
            fail(f"frame file_path must be relative: {file_path}")
        if "traj_0" in Path(file_path).stem:
            traj_0_frames.append(file_path)
        image_path = resolution_low / file_path
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        with Image.open(image_path) as image:
            if (image.width, image.height) != (int(meta["w"]), int(meta["h"])):
                bad_sizes.append(f"{image_path.name}: {image.width}x{image.height}")
        if no_gt_eval != 1:
            gt_path = Path(str(image_path).replace("/images/", "/gt/high_res_images/"))
            if not gt_path.is_file():
                missing_gt_images.append(str(gt_path))
        matrix = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            fail(f"invalid transform_matrix for frame {frame.get('file_path')}")
    if missing_images:
        fail(f"missing {len(missing_images)} LR images, first={missing_images[0]}")
    if missing_gt_images:
        fail(f"missing {len(missing_gt_images)} GT eval images, first={missing_gt_images[0]}")
    if bad_sizes:
        fail(f"LR image size mismatch, first={bad_sizes[0]}")
    if traj_0_frames and not allow_traj0:
        fail(
            "transforms.json contains frame file_path stems with traj_0; "
            f"first={traj_0_frames[0]}. Pass --allow-traj0 only when this is intentional."
        )

    ply = PlyData.read(str(pcd_path))
    pcd_count = int(len(ply["vertex"]))
    if pcd_count != init_expected_count:
        fail(f"surface PLY point count mismatch: expected {init_expected_count}, got {pcd_count}")

    return {
        "num_frames": len(frames),
        "num_images": len(image_files),
        "lr_size": [int(meta["w"]), int(meta["h"])],
        "surface_ply_points": pcd_count,
        "surface_ply": str(pcd_path),
        "init_expected_count": int(init_expected_count),
        "no_gt_eval": int(no_gt_eval),
        "gt_eval_checked": no_gt_eval != 1,
        "traj_0_frame_count": len(traj_0_frames),
        "allow_traj0": bool(allow_traj0),
    }


def validate_ckpt(ckpt_path: Path, expected_count: Optional[int]) -> Dict[str, object]:
    if not ckpt_path.is_file():
        fail(f"missing ckpt {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "splats" not in ckpt:
        fail("ckpt missing 'splats'")
    splats = ckpt["splats"]
    if "means" not in splats:
        fail("ckpt splats missing means")
    n = int(splats["means"].shape[0])
    if expected_count is not None and n != expected_count:
        fail(f"ckpt Gaussian count mismatch: expected {expected_count}, got {n}")
    required = {
        "means": (n, 3),
        "scales": (n, 3),
        "quats": (n, 4),
        "opacities": (n,),
        "sh0": (n, 1, 3),
        "shN": (n, 15, 3),
    }
    shapes = {}
    for key, expected_shape in required.items():
        if key not in splats:
            fail(f"ckpt splats missing {key}")
        tensor = splats[key]
        shapes[key] = list(tensor.shape)
        if tuple(tensor.shape) != expected_shape:
            fail(f"{key} shape mismatch: expected {expected_shape}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            fail(f"{key} contains NaN or Inf")
    quat_norm = torch.linalg.norm(splats["quats"], dim=1)
    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-4, rtol=1e-4):
        fail("quats are not normalized")
    return {
        "ckpt": str(ckpt_path),
        "step": int(ckpt.get("step", -1)),
        "num_gaussians": n,
        "expected_count": int(expected_count) if expected_count is not None else None,
        "count_matches_expected": None if expected_count is None else n == expected_count,
        "shapes": shapes,
        "quat_norm_min": float(quat_norm.min()),
        "quat_norm_max": float(quat_norm.max()),
    }


def require_file(path: Path, nonempty: bool = False) -> None:
    if not path.is_file():
        fail(f"missing {path}")
    if nonempty and path.stat().st_size <= 0:
        fail(f"empty {path}")


def require_dir(path: Path) -> None:
    if not path.is_dir():
        fail(f"missing {path}")


def non_comment_lines(path: Path) -> List[str]:
    lines = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lines.append(stripped)
    return lines


def parse_single_camera(cameras_txt: Path) -> Tuple[int, str, int, int, Tuple[float, float, float, float]]:
    lines = non_comment_lines(cameras_txt)
    if len(lines) != 1:
        fail(f"{cameras_txt} must contain exactly one camera, got {len(lines)}")
    elems = lines[0].split()
    if len(elems) != 8:
        fail(f"expected one PINHOLE camera line with 4 params in {cameras_txt}, got: {lines[0]}")
    camera_id = int(elems[0])
    model = elems[1]
    width = int(elems[2])
    height = int(elems[3])
    params = tuple(float(value) for value in elems[4:8])
    if model != "PINHOLE":
        fail(f"raw SplatFormer layout must use PINHOLE camera model, got {model}")
    return camera_id, model, width, height, params


def parse_images_text_camera_ids(images_txt: Path) -> Tuple[List[int], List[str]]:
    camera_ids = []
    names = []
    for line in non_comment_lines(images_txt):
        elems = line.split()
        if len(elems) < 10:
            continue
        camera_ids.append(int(elems[8]))
        names.append(elems[9])
    if not names:
        fail(f"{images_txt} has no image records")
    return camera_ids, names


def load_prepare_record(summary_path: Optional[Path], scene_dir: Path, splatformer_scene_root: Optional[Path]) -> Optional[Dict[str, Any]]:
    if summary_path is None:
        return None
    if not summary_path.is_file():
        fail(f"missing prepare summary {summary_path}")
    summary = json.loads(summary_path.read_text())
    records = summary.get("results") or summary.get("scenes") or []
    scene_names = {scene_dir.name}
    if splatformer_scene_root is not None:
        scene_names.add(splatformer_scene_root.name)
    for record in records:
        output_scene = record.get("output_scene")
        splatformer_scene = record.get("splatformer_scene")
        if output_scene and Path(str(output_scene)).name in scene_names:
            return record
        if splatformer_scene and Path(str(splatformer_scene)).name in scene_names:
            return record
    fail(f"could not find scene record in prepare summary for {sorted(scene_names)}")
    return None


def compare_prepare_camera(
    prepare_record: Optional[Dict[str, Any]],
    camera_size: Tuple[int, int],
    camera_params: Tuple[float, float, float, float],
) -> Dict[str, object]:
    if prepare_record is None:
        return {"checked": False}
    if prepare_record.get("intrinsics_mode") == "supergaussian-square":
        fail("prepare summary reports intrinsics_mode=supergaussian-square; raw MVImgNet layout expected")
    if prepare_record.get("gt_image_dir") != "gt_rgb":
        fail(f"prepare summary reports gt_image_dir={prepare_record.get('gt_image_dir')}; raw gt_rgb expected")
    layout = prepare_record.get("splatformer_layout") or {}
    expected_size = layout.get("image_size")
    expected_intrinsics = layout.get("computed_intrinsics") or {}
    expected_params = (
        float(expected_intrinsics["fl_x"]),
        float(expected_intrinsics["fl_y"]),
        float(expected_intrinsics["cx"]),
        float(expected_intrinsics["cy"]),
    )
    if list(camera_size) != list(expected_size):
        fail(f"cameras.txt size {list(camera_size)} differs from prepare summary {expected_size}")
    if not np.allclose(np.asarray(camera_params), np.asarray(expected_params), rtol=1e-6, atol=1e-3):
        fail(f"cameras.txt intrinsics {camera_params} differ from prepare summary {expected_params}")
    return {
        "checked": True,
        "expected_size": expected_size,
        "expected_intrinsics": {
            "fl_x": expected_params[0],
            "fl_y": expected_params[1],
            "cx": expected_params[2],
            "cy": expected_params[3],
        },
    }


def validate_splatformer_layout(
    scene_root: Path,
    expected_count: Optional[int],
    prepare_record: Optional[Dict[str, Any]],
    skip_ckpt: bool,
) -> Dict[str, object]:
    ckpt_path = scene_root / "256" / "gaussian_splatting" / "ckpts" / "ckpt_14999_rank0.pt"
    images_txt = scene_root / "1024" / "images.txt"
    cameras_txt = scene_root / "1024" / "cameras.txt"
    images_dir = scene_root / "1024" / "images"
    sparse_images_bin = scene_root / "256" / "sparse" / "0" / "images.bin"

    require_file(images_txt, nonempty=True)
    require_file(cameras_txt, nonempty=True)
    require_dir(images_dir)
    if not any(path.is_file() for path in images_dir.iterdir()):
        fail(f"no SplatFormer 1024 images under {images_dir}")
    require_file(sparse_images_bin, nonempty=True)
    camera_id, camera_model, camera_width, camera_height, camera_params = parse_single_camera(cameras_txt)
    text_camera_ids, text_image_names = parse_images_text_camera_ids(images_txt)
    if set(text_camera_ids) != {camera_id}:
        fail(f"images.txt uses multiple or mismatched camera ids: {sorted(set(text_camera_ids))}, camera={camera_id}")
    binary_images = colmap_utils.read_images_binary(sparse_images_bin)
    binary_camera_ids = {int(image.camera_id) for image in binary_images.values()}
    if binary_camera_ids != {camera_id}:
        fail(f"images.bin uses multiple or mismatched camera ids: {sorted(binary_camera_ids)}, camera={camera_id}")
    binary_image_names = sorted(image.name for image in binary_images.values())
    if binary_image_names != sorted(text_image_names):
        fail("images.bin image names differ from 1024/images.txt")
    missing_images = [name for name in text_image_names if not (images_dir / name).is_file()]
    if missing_images:
        fail(f"1024/images is missing image referenced by images.txt: {missing_images[0]}")
    prepare_camera = compare_prepare_camera(prepare_record, (camera_width, camera_height), camera_params)

    result = {
        "scene_root": str(scene_root),
        "images_txt": str(images_txt),
        "cameras_txt": str(cameras_txt),
        "images_dir": str(images_dir),
        "sparse_images_bin": str(sparse_images_bin),
        "camera_id": camera_id,
        "camera_model": camera_model,
        "camera_size": [camera_width, camera_height],
        "camera_params": {
            "fl_x": float(camera_params[0]),
            "fl_y": float(camera_params[1]),
            "cx": float(camera_params[2]),
            "cy": float(camera_params[3]),
        },
        "images_txt_count": len(text_image_names),
        "images_bin_count": len(binary_images),
        "single_camera": True,
        "prepare_camera": prepare_camera,
    }
    if not skip_ckpt:
        result["ckpt"] = validate_ckpt(ckpt_path, expected_count)
    else:
        result["ckpt_skipped"] = True
    return result


def find_iteration_dir(train_output: Path, iteration: Optional[int]) -> Optional[Path]:
    point_cloud_dir = train_output / "point_cloud"
    if iteration is not None:
        return point_cloud_dir / f"iteration_{iteration}"
    candidates = sorted(
        [path for path in point_cloud_dir.glob("iteration_*") if path.is_dir()],
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    return candidates[-1] if candidates else None


def validate_render_fit(scene_dir: Path, train_output: Path, iteration: Optional[int], max_pairs: int) -> Dict[str, object]:
    iteration_dir = find_iteration_dir(train_output, iteration)
    if iteration_dir is None or not iteration_dir.is_dir():
        fail(f"missing fork trainer iteration output under {train_output}")
    predicted_dir = iteration_dir / "predicted"
    if not predicted_dir.is_dir():
        fail(f"missing predicted render directory {predicted_dir}")

    lr_images = {path.stem: path for path in (scene_dir / "resolution_low" / "images").iterdir() if path.is_file()}
    pairs = []
    for pred_path in sorted(predicted_dir.glob("*.png")):
        lr_path = lr_images.get(pred_path.stem)
        if lr_path is not None:
            pairs.append((pred_path, lr_path))
        if len(pairs) >= max_pairs:
            break
    if not pairs:
        fail(f"no predicted images match LR supervision names in {predicted_dir}")

    maes = []
    psnrs = []
    for pred_path, lr_path in pairs:
        pred = image_to_float(pred_path)
        target = image_to_float(lr_path)
        if pred.shape != target.shape:
            fail(f"render/LR size mismatch for {pred_path.name}: {pred.shape} vs {target.shape}")
        mse = float(np.mean((pred - target) ** 2))
        mae = float(np.mean(np.abs(pred - target)))
        maes.append(mae)
        psnrs.append(float("inf") if mse <= 0 else -10.0 * math.log10(mse))

    return {
        "train_output": str(train_output),
        "iteration_dir": str(iteration_dir),
        "pairs_checked": len(pairs),
        "mae_mean": float(np.mean(maes)),
        "psnr_mean": float(np.mean(psnrs)),
    }


def main() -> int:
    args = parse_args()
    scene_dir = Path(args.scene_dir).expanduser().resolve()
    splatformer_scene_root = Path(args.splatformer_scene_root).expanduser().resolve() if args.splatformer_scene_root else None
    prepare_record = load_prepare_record(
        Path(args.prepare_summary).expanduser().resolve() if args.prepare_summary else None,
        scene_dir,
        splatformer_scene_root,
    )
    summary: Dict[str, object] = {
        "scene_dir": str(scene_dir),
        "init_expected_count": int(args.init_expected_count),
        "expected_count": int(args.expected_count) if args.expected_count is not None else None,
        "no_gt_eval": int(args.no_gt_eval),
        "allow_traj0": bool(args.allow_traj0),
        "resolution_low": validate_resolution_low(
            scene_dir,
            args.init_expected_count,
            args.no_gt_eval,
            args.allow_traj0,
        ),
    }
    if args.ckpt:
        summary["ckpt"] = validate_ckpt(Path(args.ckpt).expanduser().resolve(), args.expected_count)
    if args.splatformer_scene_root:
        summary["splatformer_layout"] = validate_splatformer_layout(
            splatformer_scene_root,
            args.expected_count,
            prepare_record,
            args.skip_splatformer_ckpt,
        )
    if args.train_output:
        summary["render_fit"] = validate_render_fit(
            scene_dir,
            Path(args.train_output).expanduser().resolve(),
            args.iteration,
            args.max_image_pairs,
        )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
