#!/usr/bin/env python3
"""Prepare MVImgNet raw low-res Gaussian scenes for SplatFormer.

The generated low-res 3DGS scene layout is intentionally compatible with
``SuperGaussian/third_parties/gaussian-splatting/train.py``:

    <output>/<category>__<scene>/resolution_low/images
    <output>/<category>__<scene>/resolution_low/transforms.json
    <output>/<category>__<scene>/resolution_low/surface_pcd_131072_seed_0.ply

Low-res supervision images are resized from raw MVImgNet ``gt_rgb`` images by
default, while ``xyz.pkl``/``rgb.pkl`` only initialize the point cloud.

Train the SuperGaussian fork with ``--no_gt_eval 1`` for this layout because
the script does not materialize ``resolution_low/gt/high_res_images``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.prepare_mvimgnet_splatformer_ready import (  # noqa: E402
    IMAGE_EXTENSIONS,
    Intrinsics,
    build_intrinsics_for_camera_poses,
    configure_unpickle_imports,
    extract_camera_poses,
    find_image_path,
    load_pickle,
    make_scene_name,
    supergaussian_square_intrinsics,
)
from dataset import colmap_utils  # noqa: E402

DEFAULT_SOURCE_ROOT = "data/mvimgnet_testset_500"
DEFAULT_OUTPUT_ROOT = "data/mvimgnet_raw_lowres_3dgs"
DEFAULT_GT_IMAGE_DIRS = ("gt_rgb",)
INTRINSIC_ATOL = 1e-3
INTRINSIC_RTOL = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene-list", help="Optional category/scene, category__scene, or absolute scene paths.")
    parser.add_argument("--limit", type=int, help="Optional maximum number of candidate scenes.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-file", help="Optional JSON summary path.")
    parser.add_argument(
        "--gt-image-dir",
        action="append",
        dest="gt_image_dirs",
        help=(
            "High-resolution GT image directory inside each scene. May be passed more than once; "
            "default is raw gt_rgb. HR_131072_gaussian requires --intrinsics-mode supergaussian-square."
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=64,
        help="Target long-edge size for LR supervision images. Default: 64.",
    )
    parser.add_argument("--point-count", type=int, default=131072)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument(
        "--train-name-prefix",
        default="train",
        help=(
            "Prefix for generated LR supervision image names. The default avoids the fork loader's "
            "'traj_0' test split heuristic."
        ),
    )
    parser.add_argument(
        "--preserve-names",
        action="store_true",
        help="Preserve source image stems where possible; duplicates still get deterministic suffixes.",
    )
    parser.add_argument(
        "--supergaussian-root",
        default=None,
        help="Repo root to add to PYTHONPATH for unpickling SuperGaussian classes.",
    )
    parser.add_argument(
        "--intrinsics-mode",
        choices=("supergaussian-square", "direct-scale"),
        default=None,
        help=(
            "Camera conversion mode. Omit to use raw direct-scale intrinsics. "
            "supergaussian-square is only valid for explicit square SuperGaussian render GT."
        ),
    )
    parser.add_argument(
        "--splatformer-output-root",
        help=(
            "Optional root for an independent raw SplatFormer scene layout. "
            "The converter later writes 256/gaussian_splatting/ckpts/ckpt_14999_rank0.pt there."
        ),
    )
    parser.add_argument(
        "--splatformer-image-long-edge",
        type=int,
        default=1024,
        help="Long-edge size for raw SplatFormer eval images. Default: 1024.",
    )
    parser.add_argument(
        "--splatformer-size-name",
        default="1024",
        help="Directory name for SplatFormer eval images/cameras. Default: 1024.",
    )
    return parser.parse_args()


def iter_scene_entries(source_root: Path, scene_list: Optional[Path]) -> Iterable[Tuple[Path, str, str]]:
    if scene_list is not None:
        with scene_list.open("r") as f:
            entries = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        for entry in entries:
            scene_path = Path(entry)
            if scene_path.is_absolute():
                yield scene_path, scene_path.parent.name, scene_path.name
            elif "__" in entry and "/" not in entry:
                category, scene = entry.split("__", 1)
                yield source_root / category / scene, category, scene
            else:
                yield source_root / scene_path, scene_path.parent.name, scene_path.name
        return

    if (source_root / "cam_extrinsics.pkl").exists():
        yield source_root, source_root.parent.name, source_root.name
        return

    if not source_root.exists():
        return
    for category_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            yield scene_dir, category_dir.name, scene_dir.name


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def colmap_pose_to_supergaussian_transform(qvec: np.ndarray, tvec: np.ndarray) -> List[List[float]]:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec_to_rotmat(qvec)
    w2c[:3, 3] = np.asarray(tvec, dtype=np.float64)
    c2w = np.linalg.inv(w2c)
    # SuperGaussian's NerfStudio reader flips columns 1:3 before inverting.
    # Pre-flip here so its recovered R/T are equivalent to the original COLMAP pose.
    c2w[:3, 1:3] *= -1.0
    return c2w.tolist()


def load_pickle_array(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        return np.asarray(pickle.load(f))


def normalize_rgb(rgb_np: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_np, dtype=np.float64)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def sample_xyz_rgb(xyz: np.ndarray, rgb: np.ndarray, count: int, seed: int) -> Tuple[np.ndarray, np.ndarray, bool]:
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb = normalize_rgb(rgb)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"rgb shape {rgb.shape} does not match xyz shape {xyz.shape}")
    rng = np.random.default_rng(seed)
    replace = len(xyz) < count
    indices = rng.choice(len(xyz), count, replace=replace)
    return xyz[indices], rgb[indices], bool(replace)


def write_surface_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normals = np.zeros_like(xyz, dtype=np.float32)
    elements = np.empty(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    elements[:] = list(map(tuple, np.concatenate([xyz, normals, rgb], axis=1)))
    PlyData([PlyElement.describe(elements, "vertex")]).write(path)


def choose_gt_dir(scene_dir: Path, gt_image_dirs: Sequence[str]) -> Tuple[Optional[Path], Optional[str]]:
    for dirname in gt_image_dirs:
        image_dir = scene_dir / dirname
        if image_dir.is_dir():
            if any(path.is_file() and path.suffix in IMAGE_EXTENSIONS for path in image_dir.iterdir()):
                return image_dir, dirname
    return None, None


def output_size_for_long_edge(width: int, height: int, target_size: int) -> Tuple[int, int]:
    if target_size <= 0:
        raise ValueError("--target-size must be positive")
    scale = min(1.0, float(target_size) / float(max(width, height)))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def resize_image(src: Path, dst: Path, target_size: int) -> Tuple[int, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    resampling = getattr(Image, "Resampling", None)
    lanczos = resampling.LANCZOS if resampling is not None else Image.LANCZOS
    with Image.open(src) as image:
        image = image.convert("RGB")
        width, height = output_size_for_long_edge(image.width, image.height, target_size)
        image.resize((width, height), resample=lanczos).save(dst)
    return width, height


def resize_image_to_size(src: Path, dst: Path, output_size: Tuple[int, int]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    resampling = getattr(Image, "Resampling", None)
    lanczos = resampling.LANCZOS if resampling is not None else Image.LANCZOS
    with Image.open(src) as image:
        image = image.convert("RGB")
        image.resize(output_size, resample=lanczos).save(dst)


def sanitize_file_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem.strip())
    return cleaned.strip("._") or "frame"


def unique_output_name(src_image: Path, frame_index: int, used_names: set[str], args: argparse.Namespace) -> str:
    if args.preserve_names:
        base = sanitize_file_stem(src_image.stem)
    else:
        base = f"{sanitize_file_stem(args.train_name_prefix)}_{frame_index:06d}"

    candidate = f"{base}.png"
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix:02d}.png"
        suffix += 1
    used_names.add(candidate)
    return candidate


def scaled_intrinsics(intrinsics: Intrinsics, scale: float) -> Tuple[float, float, float, float]:
    return (
        float(intrinsics.fx) * scale,
        float(intrinsics.fy) * scale,
        float(intrinsics.cx) * scale,
        float(intrinsics.cy) * scale,
    )


def direct_scale_intrinsics(
    intrinsics: Intrinsics,
    source_size: Tuple[int, int],
    output_size: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    source_width = int(intrinsics.width) if intrinsics.width is not None else int(source_size[0])
    source_height = int(intrinsics.height) if intrinsics.height is not None else int(source_size[1])
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"invalid source intrinsics size: {source_width}x{source_height}")
    sx = float(output_size[0]) / float(source_width)
    sy = float(output_size[1]) / float(source_height)
    return (
        float(intrinsics.fx) * sx,
        float(intrinsics.fy) * sy,
        float(intrinsics.cx) * sx,
        float(intrinsics.cy) * sy,
    )


def intrinsics_to_dict(intrinsics: Intrinsics) -> Dict[str, Optional[float]]:
    return {
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "width": int(intrinsics.width) if intrinsics.width is not None else None,
        "height": int(intrinsics.height) if intrinsics.height is not None else None,
    }


def camera_params_to_dict(params: Tuple[float, float, float, float]) -> Dict[str, float]:
    fl_x, fl_y, cx, cy = params
    return {
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx),
        "cy": float(cy),
    }


def resolve_intrinsics_mode(
    requested_mode: Optional[str],
    gt_dir_name: str,
    output_size: Tuple[int, int],
) -> str:
    width, height = output_size
    if requested_mode == "direct-scale":
        if gt_dir_name == "HR_131072_gaussian":
            raise ValueError(
                "HR_131072_gaussian is a SuperGaussian square-render GT source; "
                "use raw gt_rgb for direct-scale, or explicitly pass "
                "--intrinsics-mode supergaussian-square for square-render experiments."
            )
        return "direct-scale"

    square_allowed = gt_dir_name != "gt_rgb" and width == height
    if requested_mode == "supergaussian-square":
        if not square_allowed:
            raise ValueError(
                "--intrinsics-mode supergaussian-square requires square SuperGaussian render GT "
                f"(got gt_image_dir={gt_dir_name}, output_size={width}x{height})"
            )
        return "supergaussian-square"

    if gt_dir_name == "HR_131072_gaussian":
        raise ValueError(
            "HR_131072_gaussian was selected but --intrinsics-mode was omitted. "
            "This script defaults to raw gt_rgb/direct-scale; pass "
            "--intrinsics-mode supergaussian-square only when square-render GT is intentional."
        )
    return "direct-scale"


def compute_intrinsics(
    intrinsics: Intrinsics,
    source_size: Tuple[int, int],
    output_size: Tuple[int, int],
    mode: str,
) -> Tuple[float, float, float, float]:
    if mode == "supergaussian-square":
        return supergaussian_square_intrinsics(intrinsics, output_size[0], output_size[1])
    return direct_scale_intrinsics(intrinsics, source_size, output_size)


def intrinsics_close(lhs: Tuple[float, float, float, float], rhs: Tuple[float, float, float, float]) -> bool:
    return bool(np.allclose(np.asarray(lhs), np.asarray(rhs), rtol=INTRINSIC_RTOL, atol=INTRINSIC_ATOL))


def write_colmap_cameras(path: Path, size: Tuple[int, int], params: Tuple[float, float, float, float]) -> None:
    width, height = size
    fx, fy, cx, cy = params
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(f"1 PINHOLE {width} {height} {fx:.12g} {fy:.12g} {cx:.12g} {cy:.12g}\n")


def write_colmap_images_text(path: Path, image_records: Sequence[colmap_utils.Image]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(image_records)}, mean observations per image: 0\n")
        for image in image_records:
            q = " ".join(f"{float(value):.12g}" for value in image.qvec)
            t = " ".join(f"{float(value):.12g}" for value in image.tvec)
            f.write(f"{image.id} {q} {t} {image.camera_id} {image.name}\n\n")


def write_splatformer_layout(
    scene_root: Path,
    image_jobs: Sequence[Tuple[Path, str, Tuple[int, int]]],
    image_records: Sequence[colmap_utils.Image],
    eval_size: Tuple[int, int],
    eval_intrinsics: Tuple[float, float, float, float],
    overwrite: bool,
    size_name: str,
) -> None:
    if scene_root.exists():
        if overwrite:
            shutil.rmtree(scene_root)
        else:
            raise FileExistsError(f"SplatFormer output scene exists; pass --overwrite: {scene_root}")
    eval_dir = scene_root / size_name
    images_dir = eval_dir / "images"
    sparse_dir = scene_root / "256" / "sparse" / "0"
    ckpt_dir = scene_root / "256" / "gaussian_splatting" / "ckpts"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for src_image, output_name, output_size in image_jobs:
        if output_size != eval_size:
            raise ValueError(f"inconsistent SplatFormer eval output size for {output_name}: {output_size} vs {eval_size}")
        resize_image_to_size(src_image, images_dir / output_name, output_size)

    write_colmap_cameras(eval_dir / "cameras.txt", eval_size, eval_intrinsics)
    write_colmap_images_text(eval_dir / "images.txt", image_records)
    write_colmap_images_text(sparse_dir / "images.txt", image_records)
    cameras = {
        1: colmap_utils.Camera(
            id=1,
            model="PINHOLE",
            width=int(eval_size[0]),
            height=int(eval_size[1]),
            params=np.asarray(eval_intrinsics, dtype=np.float64),
        )
    }
    images = {image.id: image for image in image_records}
    colmap_utils.write_cameras_binary(cameras, sparse_dir / "cameras.bin")
    colmap_utils.write_images_binary(images, sparse_dir / "images.bin")


def process_scene(scene_dir: Path, category: str, scene: str, args: argparse.Namespace) -> Dict[str, Any]:
    output_scene_name = make_scene_name(category, scene)
    output_scene = Path(args.output_root).expanduser().resolve() / output_scene_name
    splatformer_scene = (
        Path(args.splatformer_output_root).expanduser().resolve() / output_scene_name
        if args.splatformer_output_root
        else None
    )
    result: Dict[str, Any] = {
        "source_scene": str(scene_dir),
        "output_scene": str(output_scene),
        "splatformer_scene": str(splatformer_scene) if splatformer_scene is not None else None,
        "status": "skipped",
        "reasons": [],
    }
    if output_scene.exists() and not args.overwrite:
        result["reasons"] = ["output exists; pass --overwrite"]
        return result
    if splatformer_scene is not None and splatformer_scene.exists() and not args.overwrite:
        result["reasons"] = ["SplatFormer output exists; pass --overwrite"]
        return result

    gt_dir, gt_dir_name = choose_gt_dir(scene_dir, args.gt_image_dirs)
    if gt_dir is None:
        result["reasons"] = [f"missing GT image dir; tried {list(args.gt_image_dirs)}"]
        return result

    extrinsics_data, error = load_pickle(scene_dir / "cam_extrinsics.pkl")
    if error is not None:
        result["reasons"] = [error]
        return result
    intrinsics_data, error = load_pickle(scene_dir / "cam_intrinsics.pkl")
    if error is not None:
        result["reasons"] = [error]
        return result

    camera_poses = extract_camera_poses(extrinsics_data)
    if not camera_poses:
        result["reasons"] = ["cam_extrinsics.pkl does not contain valid poses"]
        return result
    intrinsics_by_name = build_intrinsics_for_camera_poses(intrinsics_data, camera_poses)

    frames = []
    image_jobs = []
    splatformer_image_jobs = []
    splatformer_image_records: List[colmap_utils.Image] = []
    used_output_names: set[str] = set()
    first_size: Optional[Tuple[int, int]] = None
    first_source_size: Optional[Tuple[int, int]] = None
    first_raw_intrinsics: Optional[Intrinsics] = None
    first_intrinsics: Optional[Tuple[float, float, float, float]] = None
    first_splatformer_size: Optional[Tuple[int, int]] = None
    first_splatformer_intrinsics: Optional[Tuple[float, float, float, float]] = None
    intrinsics_mode: Optional[str] = None
    for frame_index, camera_pose in enumerate(camera_poses):
        if camera_pose.camera_name not in intrinsics_by_name:
            result["reasons"] = [f"missing intrinsics for {camera_pose.camera_name}"]
            return result
        raw_intrinsics = intrinsics_by_name[camera_pose.camera_name]
        src_image = find_image_path(gt_dir, camera_pose.camera_name)
        if src_image is None:
            result["reasons"] = [f"missing GT image for camera {camera_pose.camera_name} in {gt_dir_name}"]
            return result
        output_name = unique_output_name(src_image, frame_index, used_output_names, args)

        with Image.open(src_image) as image:
            source_size = (image.width, image.height)
            width, height = output_size_for_long_edge(image.width, image.height, args.target_size)

        if first_size is None:
            first_size = (width, height)
            first_source_size = source_size
            first_raw_intrinsics = raw_intrinsics
            try:
                intrinsics_mode = resolve_intrinsics_mode(args.intrinsics_mode, gt_dir_name, first_size)
            except ValueError as exc:
                result["reasons"] = [str(exc)]
                return result
        assert intrinsics_mode is not None
        frame_intrinsics = compute_intrinsics(raw_intrinsics, source_size, (width, height), intrinsics_mode)
        if first_intrinsics is None:
            first_intrinsics = frame_intrinsics
        elif (width, height) != first_size:
            result["reasons"] = [f"inconsistent resized LR size: first={first_size}, current={(width, height)}"]
            return result
        elif not intrinsics_close(first_intrinsics, frame_intrinsics):
            result["reasons"] = [f"per-frame scaled intrinsics differ; first mismatch={camera_pose.camera_name}"]
            return result

        if splatformer_scene is not None:
            splat_size = output_size_for_long_edge(source_size[0], source_size[1], args.splatformer_image_long_edge)
            splat_intrinsics = direct_scale_intrinsics(raw_intrinsics, source_size, splat_size)
            if first_splatformer_size is None:
                first_splatformer_size = splat_size
                first_splatformer_intrinsics = splat_intrinsics
            elif splat_size != first_splatformer_size:
                result["reasons"] = [
                    f"inconsistent SplatFormer eval image size: first={first_splatformer_size}, current={splat_size}"
                ]
                return result
            elif first_splatformer_intrinsics is None or not intrinsics_close(first_splatformer_intrinsics, splat_intrinsics):
                result["reasons"] = [
                    f"per-frame raw direct-scale SplatFormer intrinsics differ; first mismatch={camera_pose.camera_name}"
                ]
                return result

        frames.append(
            {
                "file_path": f"images/{output_name}",
                "transform_matrix": colmap_pose_to_supergaussian_transform(camera_pose.pose.qvec, camera_pose.pose.tvec),
            }
        )
        image_jobs.append((src_image, output_name))
        if splatformer_scene is not None:
            assert first_splatformer_size is not None
            splatformer_image_jobs.append((src_image, output_name, first_splatformer_size))
            splatformer_image_records.append(
                colmap_utils.Image(
                    id=frame_index + 1,
                    qvec=np.asarray(camera_pose.pose.qvec, dtype=np.float64),
                    tvec=np.asarray(camera_pose.pose.tvec, dtype=np.float64),
                    camera_id=1,
                    name=output_name,
                    xys=np.empty((0, 2), dtype=np.float64),
                    point3D_ids=np.empty((0,), dtype=np.int64),
                )
            )

    xyz_path = scene_dir / "xyz.pkl"
    rgb_path = scene_dir / "rgb.pkl"
    if not xyz_path.exists() or not rgb_path.exists():
        result["reasons"] = ["missing xyz.pkl or rgb.pkl"]
        return result

    xyz_raw = load_pickle_array(xyz_path)
    rgb_raw = load_pickle_array(rgb_path)
    xyz, rgb, replace = sample_xyz_rgb(xyz_raw, rgb_raw, args.point_count, args.sample_seed)

    result.update(
        {
            "status": "processed",
            "intrinsics_mode": intrinsics_mode,
            "gt_image_dir": gt_dir_name,
            "gt_image_dir_path": str(gt_dir),
            "num_frames": len(frames),
            "source_image_size": list(first_source_size or (0, 0)),
            "lr_size": list(first_size or (0, 0)),
            "output_size": list(first_size or (0, 0)),
            "raw_intrinsics": intrinsics_to_dict(first_raw_intrinsics) if first_raw_intrinsics is not None else None,
            "computed_intrinsics": camera_params_to_dict(first_intrinsics) if first_intrinsics is not None else None,
            "splatformer_layout": {
                "scene_root": str(splatformer_scene) if splatformer_scene is not None else None,
                "size_name": args.splatformer_size_name,
                "image_long_edge": int(args.splatformer_image_long_edge),
                "image_size": list(first_splatformer_size) if first_splatformer_size is not None else None,
                "computed_intrinsics": (
                    camera_params_to_dict(first_splatformer_intrinsics)
                    if first_splatformer_intrinsics is not None
                    else None
                ),
                "camera_model": "PINHOLE" if splatformer_scene is not None else None,
                "single_camera": splatformer_scene is not None,
            },
            "point_count": int(args.point_count),
            "raw_point_count": int(len(xyz_raw)),
            "sample_seed": int(args.sample_seed),
            "sample_replace": replace,
            "name_policy": "preserve-names" if args.preserve_names else "generated-train-prefix",
            "train_name_prefix": args.train_name_prefix,
            "no_gt_eval_required": True,
            "recommended_train_args": ["--use_low_res_as_gt", "--no_gt_eval", "1"],
        }
    )
    if image_jobs:
        result["first_output_name"] = image_jobs[0][1]
    if args.dry_run:
        result["dry_run"] = True
        return result

    if output_scene.exists():
        shutil.rmtree(output_scene)
    resolution_low = output_scene / "resolution_low"
    (resolution_low / "images").mkdir(parents=True, exist_ok=True)
    for src_image, output_name in image_jobs:
        resize_image(src_image, resolution_low / "images" / output_name, args.target_size)

    assert first_intrinsics is not None and first_size is not None
    fl_x, fl_y, cx, cy = first_intrinsics
    transforms = {
        "fl_x": fl_x,
        "fl_y": fl_y,
        "cx": cx,
        "cy": cy,
        "w": first_size[0],
        "h": first_size[1],
        "camera_model": "PINHOLE",
        "frames": frames,
    }
    (resolution_low / "transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    write_surface_ply(resolution_low / f"surface_pcd_{args.point_count}_seed_0.ply", xyz, rgb)
    if splatformer_scene is not None:
        assert first_splatformer_size is not None and first_splatformer_intrinsics is not None
        write_splatformer_layout(
            splatformer_scene,
            splatformer_image_jobs,
            splatformer_image_records,
            first_splatformer_size,
            first_splatformer_intrinsics,
            args.overwrite,
            args.splatformer_size_name,
        )
    return result


def main() -> int:
    args = parse_args()
    args.gt_image_dirs = tuple(args.gt_image_dirs or DEFAULT_GT_IMAGE_DIRS)
    source_root = Path(args.source_root).expanduser().resolve()
    scene_list = Path(args.scene_list).expanduser().resolve() if args.scene_list else None
    sg_root = configure_unpickle_imports(source_root, args.supergaussian_root)

    results = []
    if not source_root.exists():
        results.append({"source_scene": str(source_root), "status": "skipped", "reasons": ["source root does not exist"]})
    elif scene_list is not None and not scene_list.exists():
        results.append({"source_scene": str(scene_list), "status": "skipped", "reasons": ["scene list does not exist"]})
    else:
        for idx, (scene_dir, category, scene) in enumerate(iter_scene_entries(source_root, scene_list)):
            if args.limit is not None and idx >= args.limit:
                break
            try:
                results.append(process_scene(scene_dir, category, scene, args))
            except Exception as exc:
                results.append(
                    {
                        "source_scene": str(scene_dir),
                        "output_scene": make_scene_name(category, scene),
                        "status": "skipped",
                        "reasons": [f"unexpected error: {exc}"],
                    }
                )

    summary = {
        "source_root": str(source_root),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
        "supergaussian_root": str(sg_root) if sg_root is not None else None,
        "gt_image_dirs": list(args.gt_image_dirs),
        "requested_intrinsics_mode": args.intrinsics_mode,
        "intrinsics_mode_default": "direct-scale raw MVImgNet gt_rgb",
        "target_size": args.target_size,
        "splatformer_output_root": (
            str(Path(args.splatformer_output_root).expanduser().resolve()) if args.splatformer_output_root else None
        ),
        "splatformer_image_long_edge": int(args.splatformer_image_long_edge),
        "splatformer_size_name": args.splatformer_size_name,
        "point_count": args.point_count,
        "sample_seed": args.sample_seed,
        "name_policy": "preserve-names" if args.preserve_names else "generated-train-prefix",
        "train_name_prefix": args.train_name_prefix,
        "no_gt_eval_required": True,
        "recommended_train_args": ["--use_low_res_as_gt", "--no_gt_eval", "1"],
        "dry_run": bool(args.dry_run),
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
    sys.exit(main())
