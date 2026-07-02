#!/usr/bin/env python3
"""Prepare SuperGaussian MVImgNet scenes for SplatFormer evaluation."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_OUTPUT_ROOT = "data/mvimgnet_splatformer_ready"
DEFAULT_LOW_IMAGE_DIR = "LR_131072_gaussian"
DEFAULT_HIGH_IMAGE_DIR = "HR_131072_gaussian"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")
INTRINSIC_ATOL = 1e-3
INTRINSIC_RTOL = 1e-6

np = None
Image = None


def ensure_numpy() -> Any:
    global np
    if np is None:
        try:
            import numpy as numpy_module
        except ImportError as exc:
            raise RuntimeError("numpy is required to parse MVImgNet camera pickles") from exc
        np = numpy_module
    return np


def ensure_pillow() -> Any:
    global Image
    if Image is None:
        try:
            from PIL import Image as image_module
        except ImportError as exc:
            raise RuntimeError("Pillow is required to read MVImgNet image sizes") from exc
        Image = image_module
    return Image


@dataclass
class Pose:
    qvec: np.ndarray
    tvec: np.ndarray


@dataclass
class CameraPose:
    camera_name: str
    camera_id: Any
    pose: Pose


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: Optional[int] = None
    height: Optional[int] = None


@dataclass
class ImageSet:
    key: str
    image_dir_name: str
    output_dir_name: str
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    scale: float


@dataclass
class Frame:
    camera_name: str
    image_paths: Dict[str, Path]
    output_name: str
    pose: Pose
    camera_id: Any
    camera_params: Dict[str, Tuple[float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a SuperGaussian MVImgNet test set with category/scene/"
            "cam_extrinsics.pkl, cam_intrinsics.pkl, LR_*/HR_* images into a "
            "SplatFormer-ready layout."
        )
    )
    parser.add_argument("--source-root", required=True, help="Root of the SuperGaussian MVImgNet test set.")
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root. Defaults to {DEFAULT_OUTPUT_ROOT}.",
    )
    parser.add_argument(
        "--scene-list",
        help="Optional text file with scenes to process, one category/scene per line.",
    )
    parser.add_argument("--limit", type=int, help="Optional maximum number of candidate scenes to inspect.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing converted scene folders.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report without writing outputs.")
    parser.add_argument("--log-file", help="Optional JSON summary path.")
    parser.add_argument(
        "--low-image-dir",
        default=DEFAULT_LOW_IMAGE_DIR,
        help=f"Low-resolution source image directory inside each scene. Defaults to {DEFAULT_LOW_IMAGE_DIR}.",
    )
    parser.add_argument(
        "--high-image-dir",
        default=DEFAULT_HIGH_IMAGE_DIR,
        help=f"High-resolution source image directory inside each scene. Defaults to {DEFAULT_HIGH_IMAGE_DIR}.",
    )
    parser.add_argument(
        "--low-size-name",
        default="256",
        help="Output directory name for low-resolution images. Defaults to 256 for SplatFormer compatibility.",
    )
    parser.add_argument(
        "--high-size-name",
        default="1024",
        help="Output directory name for high-resolution images. Defaults to 1024 for SplatFormer compatibility.",
    )
    parser.add_argument(
        "--high-output-scale",
        type=float,
        default=1.0,
        help=(
            "Uniform scale applied when writing high-resolution images. "
            "Defaults to 1.0 for the previous copy-through behavior; use 0.25 for SuperGaussian 4x targets."
        ),
    )
    parser.add_argument(
        "--supergaussian-root",
        help="Optional SuperGaussian repo root to add to PYTHONPATH for unpickling sg_utils classes.",
    )
    parser.add_argument(
        "--intrinsics-mode",
        choices=("supergaussian-square", "direct-scale"),
        default="supergaussian-square",
        help=(
            "Camera conversion mode. 'supergaussian-square' matches SuperGaussian's square render camera "
            "at each output image size; 'direct-scale' scales original intrinsics directly to each output image size."
        ),
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    try:
        with path.open("rb") as f:
            return pickle.load(f), None
    except FileNotFoundError:
        return None, f"missing file: {path.name}"
    except Exception as exc:
        return None, f"failed to read {path.name}: {exc}"


def configure_unpickle_imports(source_root: Path, explicit_root: Optional[str]) -> Optional[Path]:
    candidates: List[Path] = []
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


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    return cleaned.strip("._") or "scene"


def make_scene_name(category: str, scene: str) -> str:
    return sanitize_name(f"{category}__{scene}")


def iter_scene_entries(source_root: Path, scene_list: Optional[Path]) -> Iterable[Tuple[Path, str, str]]:
    if scene_list is not None:
        with scene_list.open("r") as f:
            entries = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        for entry in entries:
            scene_path = Path(entry)
            if not scene_path.is_absolute():
                scene_path = source_root / scene_path
            if not scene_path.exists() and "__" in entry:
                category, scene = entry.split("__", 1)
                scene_path = source_root / category / scene
            yield scene_path, scene_path.parent.name, scene_path.name
        return

    if (source_root / "cam_extrinsics.pkl").exists():
        yield source_root, source_root.parent.name, source_root.name
        return

    if not source_root.exists():
        return

    for category_dir in sorted(p for p in source_root.iterdir() if p.is_dir()):
        for scene_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            yield scene_dir, category_dir.name, scene_dir.name


def get_value(record: Any, keys: Sequence[str]) -> Any:
    if isinstance(record, dict):
        for key in keys:
            if key in record:
                return record[key]
        lower_to_key = {str(key).lower(): key for key in record.keys()}
        for key in keys:
            actual = lower_to_key.get(key.lower())
            if actual is not None:
                return record[actual]
    for key in keys:
        if hasattr(record, key):
            return getattr(record, key)
    return None


def to_float_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    numpy = ensure_numpy()
    try:
        return numpy.asarray(value, dtype=numpy.float64)
    except Exception:
        return None


def rotmat_to_qvec(rotation: np.ndarray) -> np.ndarray:
    numpy = ensure_numpy()
    qvec = numpy.empty(4, dtype=numpy.float64)
    trace = numpy.trace(rotation)
    if trace > 0:
        s = numpy.sqrt(trace + 1.0) * 2.0
        qvec[0] = 0.25 * s
        qvec[1] = (rotation[2, 1] - rotation[1, 2]) / s
        qvec[2] = (rotation[0, 2] - rotation[2, 0]) / s
        qvec[3] = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        axis = int(numpy.argmax(numpy.diag(rotation)))
        if axis == 0:
            s = numpy.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qvec[0] = (rotation[2, 1] - rotation[1, 2]) / s
            qvec[1] = 0.25 * s
            qvec[2] = (rotation[0, 1] + rotation[1, 0]) / s
            qvec[3] = (rotation[0, 2] + rotation[2, 0]) / s
        elif axis == 1:
            s = numpy.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qvec[0] = (rotation[0, 2] - rotation[2, 0]) / s
            qvec[1] = (rotation[0, 1] + rotation[1, 0]) / s
            qvec[2] = 0.25 * s
            qvec[3] = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = numpy.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qvec[0] = (rotation[1, 0] - rotation[0, 1]) / s
            qvec[1] = (rotation[0, 2] + rotation[2, 0]) / s
            qvec[2] = (rotation[1, 2] + rotation[2, 1]) / s
            qvec[3] = 0.25 * s
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def normalize_pose(qvec: np.ndarray, tvec: np.ndarray) -> Optional[Pose]:
    numpy = ensure_numpy()
    qvec = numpy.asarray(qvec, dtype=numpy.float64).reshape(-1)
    tvec = numpy.asarray(tvec, dtype=numpy.float64).reshape(-1)
    if qvec.size != 4 or tvec.size != 3:
        return None
    norm = numpy.linalg.norm(qvec)
    if not numpy.isfinite(norm) or norm <= 0:
        return None
    return Pose(qvec=qvec / norm, tvec=tvec)


def parse_pose(record: Any) -> Optional[Pose]:
    if isinstance(record, dict) or any(hasattr(record, key) for key in ("qvec", "tvec", "R", "t")):
        qvec = get_value(record, ("qvec", "q", "quat", "quaternion"))
        tvec = get_value(record, ("tvec", "t", "trans", "translation"))
        if qvec is not None and tvec is not None:
            parsed = normalize_pose(to_float_array(qvec), to_float_array(tvec))
            if parsed is not None:
                return parsed

        rotation = get_value(record, ("R", "rotation", "rot", "rotation_matrix"))
        if rotation is not None and tvec is not None:
            rotation_arr = to_float_array(rotation)
            tvec_arr = to_float_array(tvec)
            if rotation_arr is not None and rotation_arr.shape == (3, 3):
                return normalize_pose(rotmat_to_qvec(rotation_arr), tvec_arr)
            if rotation_arr is not None and rotation_arr.reshape(-1).size == 4:
                return normalize_pose(rotation_arr.reshape(-1), tvec_arr)

        matrix = get_value(record, ("w2c", "world_to_camera", "extrinsic", "extrinsics", "T_cw", "matrix"))
        matrix_arr = to_float_array(matrix)
        if matrix_arr is not None and matrix_arr.shape == (4, 4):
            return normalize_pose(rotmat_to_qvec(matrix_arr[:3, :3]), matrix_arr[:3, 3])

    if isinstance(record, (list, tuple)):
        if len(record) == 2:
            return normalize_pose(to_float_array(record[0]), to_float_array(record[1]))
        arr = to_float_array(record)
        if arr is not None:
            flat = arr.reshape(-1)
            if flat.size == 7:
                return normalize_pose(flat[:4], flat[4:])
            if arr.shape == (4, 4):
                return normalize_pose(rotmat_to_qvec(arr[:3, :3]), arr[:3, 3])

    return None


def unwrap_common_mapping(data: Any, keys: Sequence[str]) -> Any:
    if isinstance(data, dict):
        for key in keys:
            value = get_value(data, (key,))
            if value is not None:
                return value
    return data


def extract_camera_poses(extrinsics: Any) -> List[CameraPose]:
    extrinsics = unwrap_common_mapping(extrinsics, ("cam_extrinsics", "extrinsics", "cameras", "frames"))
    poses: List[CameraPose] = []

    if isinstance(extrinsics, dict):
        qvecs = get_value(extrinsics, ("qvec", "qvecs", "quaternions"))
        tvecs = get_value(extrinsics, ("tvec", "tvecs", "translations"))
        if isinstance(qvecs, dict) and isinstance(tvecs, dict):
            for name in sorted(set(qvecs.keys()) & set(tvecs.keys()), key=str):
                pose = normalize_pose(to_float_array(qvecs[name]), to_float_array(tvecs[name]))
                if pose is not None:
                    poses.append(CameraPose(camera_name=str(name), camera_id=name, pose=pose))
            if poses:
                return sorted(poses, key=lambda item: item.camera_name)

        for key, record in extrinsics.items():
            pose = parse_pose(record)
            if pose is not None:
                record_name = get_value(record, ("name", "camera_name", "image_name", "file_name", "path"))
                camera_name = str(record_name) if record_name is not None else str(key)
                camera_id = get_value(record, ("camera_id", "cameraid", "id"))
                poses.append(CameraPose(camera_name=camera_name, camera_id=camera_id if camera_id is not None else key, pose=pose))
        if poses:
            return sorted(poses, key=lambda item: item.camera_name)

    if isinstance(extrinsics, (list, tuple)):
        for idx, record in enumerate(extrinsics):
            name = get_value(record, ("name", "camera_name", "image_name", "file_name", "path"))
            camera_id = get_value(record, ("camera_id", "cameraid", "id"))
            pose = parse_pose(record)
            if name is not None and pose is not None:
                poses.append(CameraPose(camera_name=str(name), camera_id=camera_id if camera_id is not None else idx, pose=pose))
            elif pose is not None:
                poses.append(CameraPose(camera_name=f"{idx:06d}", camera_id=camera_id if camera_id is not None else idx, pose=pose))

    return sorted(poses, key=lambda item: item.camera_name)


def parse_intrinsics(record: Any) -> Optional[Intrinsics]:
    if isinstance(record, dict) or any(hasattr(record, key) for key in ("K", "f", "cx", "cy", "params", "model")):
        width = get_value(record, ("width", "w", "image_width"))
        height = get_value(record, ("height", "h", "image_height"))
        width = int(width) if width is not None else None
        height = int(height) if height is not None else None
        matrix = get_value(record, ("K", "k", "intrinsic", "intrinsics", "camera_matrix"))
        matrix_arr = to_float_array(matrix)
        if matrix_arr is not None and matrix_arr.shape == (3, 3):
            return Intrinsics(
                fx=float(matrix_arr[0, 0]),
                fy=float(matrix_arr[1, 1]),
                cx=float(matrix_arr[0, 2]),
                cy=float(matrix_arr[1, 2]),
                width=width,
                height=height,
            )

        fx = get_value(record, ("fx", "fl_x", "focal_x"))
        fy = get_value(record, ("fy", "fl_y", "focal_y"))
        f = get_value(record, ("f", "fl", "focal", "focal_length"))
        cx = get_value(record, ("cx", "c_x", "principal_x", "ppx"))
        cy = get_value(record, ("cy", "c_y", "principal_y", "ppy"))
        if f is not None and fx is None:
            fx = f
        if f is not None and fy is None:
            fy = f
        if fx is not None and fy is not None and cx is not None and cy is not None:
            return Intrinsics(float(fx), float(fy), float(cx), float(cy), width=width, height=height)

        params = get_value(record, ("params", "param", "camera_params"))
        params_arr = to_float_array(params)
        if params_arr is not None:
            flat = params_arr.reshape(-1)
            model = str(get_value(record, ("model", "camera_model", "type")) or "").upper()
            if model == "PINHOLE" and flat.size >= 4:
                return Intrinsics(float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3]), width=width, height=height)
            if model == "SIMPLE_RADIAL" and flat.size >= 3:
                return Intrinsics(float(flat[0]), float(flat[0]), float(flat[1]), float(flat[2]), width=width, height=height)
            parsed = parse_intrinsics(params)
            if parsed is not None:
                parsed.width = parsed.width if parsed.width is not None else width
                parsed.height = parsed.height if parsed.height is not None else height
                return parsed

    arr = to_float_array(record)
    if arr is None:
        return None
    if arr.shape == (3, 3):
        return Intrinsics(float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2]))

    flat = arr.reshape(-1)
    if flat.size >= 4:
        # SIMPLE_RADIAL stores f, cx, cy, k. Ignore distortion as requested.
        return Intrinsics(float(flat[0]), float(flat[0]), float(flat[1]), float(flat[2]))
    if flat.size == 3:
        return Intrinsics(float(flat[0]), float(flat[0]), float(flat[1]), float(flat[2]))
    return None


def extract_dict_value_by_name(data: Dict[Any, Any], camera_name: str) -> Any:
    if camera_name in data:
        return data[camera_name]
    stem = Path(camera_name).stem
    if stem in data:
        return data[stem]
    for key, value in data.items():
        key_str = str(key)
        if key_str == camera_name or Path(key_str).stem == stem:
            return value
    return None


def build_intrinsics_mapping(intrinsics_data: Any, camera_names: Sequence[str]) -> Dict[str, Intrinsics]:
    intrinsics_data = unwrap_common_mapping(intrinsics_data, ("cam_intrinsics", "intrinsics", "cameras", "frames"))
    global_intrinsics = parse_intrinsics(intrinsics_data)
    if global_intrinsics is not None:
        return {name: global_intrinsics for name in camera_names}

    mapping: Dict[str, Intrinsics] = {}
    if isinstance(intrinsics_data, dict):
        for name in camera_names:
            record = extract_dict_value_by_name(intrinsics_data, name)
            parsed = parse_intrinsics(record)
            if parsed is not None:
                mapping[name] = parsed
        return mapping

    if isinstance(intrinsics_data, (list, tuple)) and len(intrinsics_data) == len(camera_names):
        for name, record in zip(camera_names, intrinsics_data):
            parsed = parse_intrinsics(record)
            if parsed is not None:
                mapping[name] = parsed
    return mapping


def build_intrinsics_for_camera_poses(intrinsics_data: Any, camera_poses: Sequence[CameraPose]) -> Dict[str, Intrinsics]:
    camera_names = [camera_pose.camera_name for camera_pose in camera_poses]
    by_name = build_intrinsics_mapping(intrinsics_data, camera_names)
    if len(by_name) == len(camera_names):
        return by_name

    intrinsics_data = unwrap_common_mapping(intrinsics_data, ("cam_intrinsics", "intrinsics", "cameras", "frames"))
    mapping: Dict[str, Intrinsics] = {}
    if isinstance(intrinsics_data, dict):
        for camera_pose in camera_poses:
            candidates = [
                camera_pose.camera_id,
                str(camera_pose.camera_id),
                camera_pose.camera_name,
                Path(camera_pose.camera_name).stem,
            ]
            record = None
            for candidate in candidates:
                if candidate in intrinsics_data:
                    record = intrinsics_data[candidate]
                    break
            if record is None:
                record = extract_dict_value_by_name(intrinsics_data, camera_pose.camera_name)
            parsed = parse_intrinsics(record)
            if parsed is not None:
                mapping[camera_pose.camera_name] = parsed
    return mapping


def find_image_path(gt_rgb_dir: Path, camera_name: str) -> Optional[Path]:
    camera_path = gt_rgb_dir / camera_name
    if camera_path.exists() and camera_path.is_file():
        return camera_path

    relative = Path(camera_name)
    stem = relative.stem if relative.suffix else str(relative)
    candidates = []
    if relative.parent != Path("."):
        candidates.extend(gt_rgb_dir / relative.with_suffix(ext) for ext in IMAGE_EXTENSIONS)
    candidates.extend(gt_rgb_dir / f"{Path(stem).name}{ext}" for ext in IMAGE_EXTENSIONS)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    matches = sorted(gt_rgb_dir.glob(f"{Path(stem).name}.*"))
    for match in matches:
        if match.suffix in IMAGE_EXTENSIONS and match.is_file():
            return match
    return None


def supergaussian_square_intrinsics(
    intrinsics: Intrinsics,
    target_width: int,
    target_height: int,
) -> Tuple[float, float, float, float]:
    """Match SuperGaussian's square-camera convention at the requested output size."""
    if intrinsics.width is None or intrinsics.height is None:
        scale_x = float(target_width) / 256.0
        scale_y = float(target_height) / 256.0
        return intrinsics.fx * scale_x, intrinsics.fy * scale_y, intrinsics.cx * scale_x, intrinsics.cy * scale_y

    length = min(intrinsics.width, intrinsics.height)
    base_scale = 256.0 / float(length)
    scaled_width = int(intrinsics.width * base_scale)
    scaled_height = int(intrinsics.height * base_scale)
    left = (scaled_width - 256) // 2
    top = (scaled_height - 256) // 2

    fx = intrinsics.fx * base_scale
    fy = intrinsics.fy * base_scale
    cx = intrinsics.cx * base_scale - left
    cy = intrinsics.cy * base_scale - top

    scale_x = float(target_width) / 256.0
    scale_y = float(target_height) / 256.0
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


def direct_scale_intrinsics(
    intrinsics: Intrinsics,
    target_width: int,
    target_height: int,
) -> Tuple[float, float, float, float]:
    if intrinsics.width is None or intrinsics.height is None:
        sx = float(target_width) / 256.0
        sy = float(target_height) / 256.0
        return intrinsics.fx * sx, intrinsics.fy * sy, intrinsics.cx * sx, intrinsics.cy * sy
    sx = float(target_width) / float(intrinsics.width)
    sy = float(target_height) / float(intrinsics.height)
    return intrinsics.fx * sx, intrinsics.fy * sy, intrinsics.cx * sx, intrinsics.cy * sy


def convert_intrinsics(
    intrinsics: Intrinsics,
    target_width: int,
    target_height: int,
    mode: str,
) -> Tuple[float, float, float, float]:
    if mode == "direct-scale":
        return direct_scale_intrinsics(intrinsics, target_width, target_height)
    return supergaussian_square_intrinsics(intrinsics, target_width, target_height)


def unique_output_names(camera_names: Sequence[str]) -> Dict[str, str]:
    proposed = {name: sanitize_name(Path(name).stem) + ".png" for name in camera_names}
    counts = Counter(proposed.values())
    output_names = {}
    for idx, name in enumerate(camera_names):
        output_names[name] = f"{idx:06d}.png" if counts[proposed[name]] > 1 else proposed[name]
    return output_names


def read_image_size(path: Path) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        image_module = ensure_pillow()
        with image_module.open(path) as image:
            width, height = image.width, image.height
    except Exception as exc:
        return None, None, f"failed to open image {path.name}: {exc}"
    return width, height, None


def infer_image_set(
    scene_dir: Path,
    key: str,
    image_dir_name: str,
    output_dir_name: str,
    output_scale: float,
) -> Tuple[Optional[ImageSet], Optional[str]]:
    image_dir = scene_dir / image_dir_name
    if not image_dir.is_dir():
        return None, f"missing image directory: {image_dir_name}"
    candidates = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS)
    if not candidates:
        return None, f"no images found in {image_dir_name}"
    if not math.isfinite(output_scale) or output_scale <= 0:
        return None, f"invalid output scale for {image_dir_name}: {output_scale}"
    source_width, source_height, error = read_image_size(candidates[0])
    if error is not None:
        return None, error
    output_width = int(round(float(source_width) * output_scale))
    output_height = int(round(float(source_height) * output_scale))
    if output_width <= 0 or output_height <= 0:
        return None, (
            f"scaled image size for {image_dir_name} must be positive: "
            f"{source_width}x{source_height} * {output_scale} -> {output_width}x{output_height}"
        )
    return (
        ImageSet(
            key=key,
            image_dir_name=image_dir_name,
            output_dir_name=output_dir_name,
            source_width=int(source_width),
            source_height=int(source_height),
            output_width=output_width,
            output_height=output_height,
            scale=float(output_scale),
        ),
        None,
    )


def format_float(value: float) -> str:
    return f"{float(value):.12g}"


def write_cameras(path: Path, width: int, height: int, params: Tuple[float, float, float, float]) -> None:
    fx, fy, cx, cy = params
    with path.open("w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write(
            "1 PINHOLE "
            f"{width} {height} {format_float(fx)} {format_float(fy)} {format_float(cx)} {format_float(cy)}\n"
        )


def write_images(path: Path, frames: Sequence[Frame]) -> None:
    with path.open("w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}, mean observations per image: 0\n")
        for image_id, frame in enumerate(frames, start=1):
            q = " ".join(format_float(value) for value in frame.pose.qvec)
            t = " ".join(format_float(value) for value in frame.pose.tvec)
            f.write(f"{image_id} {q} {t} 1 {frame.output_name}\n\n")


def copy_image(src: Path, dst: Path, output_width: int, output_height: int, scale: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if scale == 1.0:
        shutil.copy2(src, dst)
        return

    image_module = ensure_pillow()
    resampling = getattr(image_module, "Resampling", None)
    bilinear = resampling.BILINEAR if resampling is not None else image_module.BILINEAR
    with image_module.open(src) as image:
        resized = image.resize((output_width, output_height), resample=bilinear)
        resized.save(dst)


def validate_scene(
    scene_dir: Path,
    category: str,
    scene: str,
    image_sets: Sequence[ImageSet],
    intrinsics_mode: str,
) -> Tuple[Optional[List[Frame]], List[str]]:
    reasons: List[str] = []
    extrinsics_path = scene_dir / "cam_extrinsics.pkl"
    intrinsics_path = scene_dir / "cam_intrinsics.pkl"

    if not scene_dir.exists():
        return None, [f"missing scene directory: {scene_dir}"]
    for image_set in image_sets:
        if not (scene_dir / image_set.image_dir_name).is_dir():
            reasons.append(f"missing image directory: {image_set.image_dir_name}")

    extrinsics_data, error = load_pickle(extrinsics_path)
    if error is not None:
        reasons.append(error)
    intrinsics_data, error = load_pickle(intrinsics_path)
    if error is not None:
        reasons.append(error)
    if reasons:
        return None, reasons

    camera_poses = extract_camera_poses(extrinsics_data)
    if not camera_poses:
        return None, ["cam_extrinsics.pkl does not contain named qvec/tvec poses"]

    camera_names = [camera_pose.camera_name for camera_pose in camera_poses]
    intrinsics = build_intrinsics_for_camera_poses(intrinsics_data, camera_poses)
    missing_intrinsics = [name for name in camera_names if name not in intrinsics]
    if missing_intrinsics:
        return None, [f"missing intrinsics for {len(missing_intrinsics)} frames, first={missing_intrinsics[0]}"]

    output_names = unique_output_names(camera_names)
    frames: List[Frame] = []
    for camera_pose in camera_poses:
        name = camera_pose.camera_name
        image_paths: Dict[str, Path] = {}
        camera_params: Dict[str, Tuple[float, float, float, float]] = {}
        for image_set in image_sets:
            image_path = find_image_path(scene_dir / image_set.image_dir_name, name)
            if image_path is None:
                return None, [f"missing {image_set.image_dir_name} image for camera {name}"]
            actual_width, actual_height, error = read_image_size(image_path)
            if error is not None:
                return None, [error]
            if actual_width != image_set.source_width or actual_height != image_set.source_height:
                return None, [
                    f"inconsistent image size in {image_set.image_dir_name}: "
                    f"expected {image_set.source_width}x{image_set.source_height}, "
                    f"got {actual_width}x{actual_height} for {image_path.name}"
                ]
            image_paths[image_set.key] = image_path
            camera_params[image_set.key] = convert_intrinsics(
                intrinsics[name],
                image_set.output_width,
                image_set.output_height,
                intrinsics_mode,
            )

        frames.append(
            Frame(
                camera_name=name,
                image_paths=image_paths,
                output_name=output_names[name],
                pose=camera_pose.pose,
                camera_id=camera_pose.camera_id,
                camera_params=camera_params,
            )
        )

    for image_set in image_sets:
        numpy = ensure_numpy()
        reference = numpy.asarray(frames[0].camera_params[image_set.key])
        for frame in frames[1:]:
            current = numpy.asarray(frame.camera_params[image_set.key])
            if not numpy.allclose(reference, current, rtol=INTRINSIC_RTOL, atol=INTRINSIC_ATOL):
                return None, [
                    "per-frame intrinsics are not consistent "
                    f"for {category}/{scene} in {image_set.image_dir_name}, first mismatch={frame.camera_name}"
                ]

    return frames, []


def write_scene(frames: Sequence[Frame], image_sets: Sequence[ImageSet], output_scene_dir: Path) -> None:
    for image_set in image_sets:
        size_dir = output_scene_dir / image_set.output_dir_name
        images_dir = size_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        for frame in frames:
            copy_image(
                frame.image_paths[image_set.key],
                images_dir / frame.output_name,
                image_set.output_width,
                image_set.output_height,
                image_set.scale,
            )

        write_cameras(
            size_dir / "cameras.txt",
            image_set.output_width,
            image_set.output_height,
            frames[0].camera_params[image_set.key],
        )
        write_images(size_dir / "images.txt", frames)


def process_scene(
    scene_dir: Path,
    category: str,
    scene: str,
    output_root: Path,
    low_image_dir: str,
    high_image_dir: str,
    low_size_name: str,
    high_size_name: str,
    high_output_scale: float,
    intrinsics_mode: str,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    output_scene_name = make_scene_name(category, scene)
    output_scene_dir = output_root / output_scene_name
    result = {
        "source": str(scene_dir),
        "category": category,
        "scene": scene,
        "output_scene": output_scene_name,
        "status": "skipped",
        "reasons": [],
        "num_frames": 0,
        "image_sets": {},
        "intrinsics_mode": intrinsics_mode,
    }

    if output_scene_dir.exists() and not overwrite:
        result["reasons"] = ["output scene exists; pass --overwrite to regenerate"]
        return result

    image_sets: List[ImageSet] = []
    for key, image_dir_name, output_dir_name, output_scale in (
        ("low", low_image_dir, low_size_name, 1.0),
        ("high", high_image_dir, high_size_name, high_output_scale),
    ):
        image_set, reason = infer_image_set(scene_dir, key, image_dir_name, output_dir_name, output_scale)
        if image_set is None:
            result["reasons"] = [reason]
            return result
        image_sets.append(image_set)

    result["image_sets"] = {
        image_set.key: {
            "source": image_set.image_dir_name,
            "output": image_set.output_dir_name,
            "source_size": [image_set.source_width, image_set.source_height],
            "output_size": [image_set.output_width, image_set.output_height],
            "scale": image_set.scale,
        }
        for image_set in image_sets
    }

    frames, reasons = validate_scene(scene_dir, category, scene, image_sets, intrinsics_mode)
    if frames is None:
        result["reasons"] = reasons
        return result

    result["num_frames"] = len(frames)
    if dry_run:
        result["status"] = "processed"
        result["dry_run"] = True
        return result

    if output_scene_dir.exists():
        shutil.rmtree(output_scene_dir)
    output_scene_dir.mkdir(parents=True, exist_ok=True)
    write_scene(frames, image_sets, output_scene_dir)

    result["status"] = "processed"
    return result


def build_summary(args: argparse.Namespace) -> Dict[str, Any]:
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    scene_list = Path(args.scene_list).expanduser().resolve() if args.scene_list else None
    supergaussian_root = configure_unpickle_imports(source_root, args.supergaussian_root)

    results: List[Dict[str, Any]] = []
    if not source_root.exists():
        results.append(
            {
                "source": str(source_root),
                "status": "skipped",
                "reasons": ["source root does not exist"],
                "num_frames": 0,
            }
        )
    elif scene_list is not None and not scene_list.exists():
        results.append(
            {
                "source": str(scene_list),
                "status": "skipped",
                "reasons": ["scene list does not exist"],
                "num_frames": 0,
            }
        )
    else:
        candidates = iter_scene_entries(source_root, scene_list)
        for idx, (scene_dir, category, scene) in enumerate(candidates):
            if args.limit is not None and idx >= args.limit:
                break
            try:
                results.append(
                    process_scene(
                        scene_dir,
                        category,
                        scene,
                        output_root,
                        args.low_image_dir,
                        args.high_image_dir,
                        args.low_size_name,
                        args.high_size_name,
                        args.high_output_scale,
                        args.intrinsics_mode,
                        args.overwrite,
                        args.dry_run,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "source": str(scene_dir),
                        "category": category,
                        "scene": scene,
                        "status": "skipped",
                        "reasons": [f"unexpected error: {exc}"],
                        "num_frames": 0,
                    }
                )

    reason_counts = Counter()
    for result in results:
        if result["status"] != "processed":
            for reason in result.get("reasons", []):
                reason_counts[reason] += 1

    return {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "supergaussian_root": str(supergaussian_root) if supergaussian_root is not None else None,
        "intrinsics_mode": args.intrinsics_mode,
        "dry_run": bool(args.dry_run),
        "processed": sum(1 for result in results if result["status"] == "processed"),
        "skipped": sum(1 for result in results if result["status"] != "processed"),
        "reason_counts": dict(sorted(reason_counts.items())),
        "scenes": results,
    }


def main() -> int:
    args = parse_args()
    summary = build_summary(args)
    output = json.dumps(summary, indent=2, sort_keys=True)
    print(output)

    if args.log_file:
        log_path = Path(args.log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
