#!/usr/bin/env python3
"""Convert a 3DGS-format PLY into a SplatFormer gsplat checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from plyfile import PlyData

DEFAULT_CKPT_REL = "256/gaussian_splatting/ckpts/ckpt_14999_rank0.pt"
SUPERGAUSSIAN_DC_SCALE = 1.77245385091


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ply", required=True, help="Input point_cloud.ply from a trained MVImgNet low-res 3DGS scene.")
    parser.add_argument(
        "--output",
        help=(
            "Output ckpt path. If omitted with --scene-dir, writes "
            "256/gaussian_splatting/ckpts/ckpt_<step>_rank0.pt."
        ),
    )
    parser.add_argument("--scene-dir", help="Optional SplatFormer scene dir used to derive the default output path.")
    parser.add_argument(
        "--splatformer-scene-root",
        help="Preferred alias for --scene-dir; destination SplatFormer scene root.",
    )
    parser.add_argument(
        "--copy-layout-from",
        help="Optional existing SplatFormer-ready scene root to copy 256/1024 layout from before writing the ckpt.",
    )
    parser.add_argument(
        "--allow-square-layout-copy",
        action="store_true",
        help=(
            "Allow --copy-layout-from sources that look like SuperGaussian square-render layouts. "
            "By default these are rejected to protect raw MVImgNet evaluation."
        ),
    )
    parser.add_argument("--step", type=int, default=14999)
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help=(
            "Optional expected converted Gaussian count. "
            "Unset allows trainer densification to change the final count."
        ),
    )
    parser.add_argument(
        "--dc-mode",
        choices=("supergaussian-fork", "standard-3dgs"),
        default="supergaussian-fork",
        help=(
            "SuperGaussian fork stores raw f_dc and renders tanh(f_dc) * 1.77245385091; "
            "standard-3dgs keeps f_dc as SH coefficients directly."
        ),
    )
    parser.add_argument("--log-file", help="Optional JSON summary path.")
    return parser.parse_args()


def property_names(ply: PlyData) -> List[str]:
    return [prop.name for prop in ply["vertex"].properties]


def require_fields(names: List[str], required: List[str]) -> None:
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"input PLY is missing required fields: {missing}")


def stack_fields(vertex: np.ndarray, names: List[str]) -> torch.Tensor:
    return torch.as_tensor(np.stack([np.asarray(vertex[name]) for name in names], axis=1), dtype=torch.float32)


def sorted_numbered_fields(names: List[str], prefix: str) -> List[str]:
    fields = [name for name in names if name.startswith(prefix)]
    return sorted(fields, key=lambda name: int(name.rsplit("_", 1)[-1]))


def convert_dc(vertex: np.ndarray, mode: str) -> torch.Tensor:
    f_dc = stack_fields(vertex, ["f_dc_0", "f_dc_1", "f_dc_2"])
    if mode == "supergaussian-fork":
        # The fork renders GaussianModel.get_features as tanh(_features_dc) * 1/C0/2.
        # SplatFormer/gsplat expects SH coefficients, so bake that activation here.
        f_dc = torch.tanh(f_dc) * SUPERGAUSSIAN_DC_SCALE
    return f_dc.reshape(f_dc.shape[0], 1, 3)


def convert_ply(input_ply: Path, dc_mode: str) -> Dict[str, torch.Tensor]:
    ply = PlyData.read(str(input_ply))
    vertex = ply["vertex"]
    names = property_names(ply)
    require_fields(
        names,
        [
            "x",
            "y",
            "z",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
        ],
    )

    means = stack_fields(vertex, ["x", "y", "z"])
    scales = stack_fields(vertex, ["scale_0", "scale_1", "scale_2"])
    opacities = torch.as_tensor(np.asarray(vertex["opacity"]), dtype=torch.float32)
    quats = stack_fields(vertex, ["rot_0", "rot_1", "rot_2", "rot_3"])
    quats = torch.nn.functional.normalize(quats, dim=1)
    sh0 = convert_dc(vertex, dc_mode)

    f_rest_names = sorted_numbered_fields(names, "f_rest_")
    if len(f_rest_names) != 45:
        raise ValueError(f"expected 45 f_rest_* fields for SH degree 3, got {len(f_rest_names)}")
    f_rest = stack_fields(vertex, f_rest_names)
    shN = f_rest.reshape(f_rest.shape[0], 3, 15).transpose(1, 2).contiguous()

    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }


def validate_splats(splats: Dict[str, torch.Tensor], expected_count: Optional[int]) -> Dict[str, object]:
    n = int(splats["means"].shape[0])
    checks: Dict[str, object] = {
        "num_gaussians": n,
        "expected_count": int(expected_count) if expected_count is not None else None,
        "count_matches_expected": None if expected_count is None else n == expected_count,
        "shapes": {key: list(value.shape) for key, value in splats.items()},
    }
    if expected_count is not None and n != expected_count:
        checks["count_warning"] = f"converted Gaussian count differs from expected {expected_count}"
    expected_shapes = {
        "means": (n, 3),
        "scales": (n, 3),
        "quats": (n, 4),
        "opacities": (n,),
        "sh0": (n, 1, 3),
        "shN": (n, 15, 3),
    }
    for key, expected_shape in expected_shapes.items():
        if tuple(splats[key].shape) != expected_shape:
            raise ValueError(f"{key} shape mismatch: expected {expected_shape}, got {tuple(splats[key].shape)}")
        if not torch.isfinite(splats[key]).all():
            raise ValueError(f"{key} contains NaN or Inf")
    quat_norm = torch.linalg.norm(splats["quats"], dim=1)
    checks["quat_norm_min"] = float(quat_norm.min())
    checks["quat_norm_max"] = float(quat_norm.max())
    if not torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-4, rtol=1e-4):
        raise ValueError("quaternions are not normalized after conversion")
    return checks


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def summary_mentions_square_intrinsics(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size > 20 * 1024 * 1024:
        return False
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return False
    return "intrinsics_mode" in text and "supergaussian-square" in text


def reject_square_layout_copy(src_scene: Path, allow_square_layout_copy: bool) -> None:
    if allow_square_layout_copy:
        return
    resolved = src_scene.expanduser().resolve()
    lower_path = str(resolved).lower()
    if "sg4x" in lower_path:
        raise ValueError(
            "--copy-layout-from points to a path containing 'sg4x', which is likely a "
            "SuperGaussian square-render layout. Pass --allow-square-layout-copy only if this is intentional."
        )

    candidate_summaries = list(resolved.glob("*.json")) + list(resolved.parent.glob("*.json"))
    for summary_path in candidate_summaries:
        if summary_mentions_square_intrinsics(summary_path):
            raise ValueError(
                f"--copy-layout-from appears to come from supergaussian-square summary {summary_path}. "
                "Use a raw layout or pass --allow-square-layout-copy only for square-render experiments."
            )


def copy_splatformer_layout(src_scene: Path, dst_scene: Path) -> List[str]:
    src_scene = src_scene.expanduser().resolve()
    dst_scene = dst_scene.expanduser().resolve()
    if src_scene == dst_scene:
        return []

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
    copied = []
    for rel in required:
        src = src_scene / rel
        if not src.exists():
            raise FileNotFoundError(f"Missing required SplatFormer layout path: {src}")
        copy_path(src, dst_scene / rel)
        copied.append(rel)

    for rel in ("256/sparse/0/cameras.txt", "256/sparse/0/images.txt", "1024/sparse/0/images.bin"):
        src = src_scene / rel
        if src.exists():
            copy_path(src, dst_scene / rel)
            copied.append(rel)
    return copied


def resolve_scene_root(args: argparse.Namespace) -> Optional[Path]:
    scene_dir = args.splatformer_scene_root or args.scene_dir
    if args.splatformer_scene_root and args.scene_dir:
        if Path(args.splatformer_scene_root).expanduser().resolve() != Path(args.scene_dir).expanduser().resolve():
            raise ValueError("--scene-dir and --splatformer-scene-root point to different destinations")
    return Path(scene_dir).expanduser().resolve() if scene_dir else None


def resolve_output(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output).expanduser().resolve()
    scene_root = resolve_scene_root(args)
    if scene_root is None:
        raise ValueError("pass --output or --splatformer-scene-root/--scene-dir")
    return scene_root / "256" / "gaussian_splatting" / "ckpts" / f"ckpt_{args.step}_rank0.pt"


def main() -> int:
    args = parse_args()
    input_ply = Path(args.input_ply).expanduser().resolve()
    output = resolve_output(args)
    splats = convert_ply(input_ply, args.dc_mode)
    checks = validate_splats(splats, args.expected_count)

    copied_layout: List[str] = []
    if args.copy_layout_from:
        scene_root = resolve_scene_root(args)
        if scene_root is None:
            raise ValueError("--copy-layout-from requires --splatformer-scene-root/--scene-dir")
        reject_square_layout_copy(Path(args.copy_layout_from), args.allow_square_layout_copy)
        copied_layout = copy_splatformer_layout(Path(args.copy_layout_from), scene_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"step": int(args.step), "splats": splats}, output)

    summary = {
        "input_ply": str(input_ply),
        "output": str(output),
        "step": int(args.step),
        "dc_mode": args.dc_mode,
        "splatformer_scene_root": str(resolve_scene_root(args)) if resolve_scene_root(args) is not None else None,
        "copy_layout_from": str(Path(args.copy_layout_from).expanduser().resolve()) if args.copy_layout_from else None,
        "allow_square_layout_copy": bool(args.allow_square_layout_copy),
        "copied_layout": copied_layout,
        "checks": checks,
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
