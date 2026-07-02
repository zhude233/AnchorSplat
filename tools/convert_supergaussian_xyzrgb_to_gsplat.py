#!/usr/bin/env python3
"""Convert SuperGaussian xyz/rgb inputs into SplatFormer-readable gsplat ckpts."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch


DEFAULT_SOURCE_ROOT = "data/mvimgnet_testset_500"
DEFAULT_LAYOUT_ROOT = "data/mvimgnet_splatformer_ready"
DEFAULT_OUTPUT_ROOT = "data/mvimgnet_sginput_splatformer_ready"
C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--layout-root", default=DEFAULT_LAYOUT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scene-list", help="Optional file with category/scene or category__scene entries.")
    parser.add_argument("--limit", type=int, help="Optional number of scenes to process.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--ckpt-step", type=int, default=14999)
    parser.add_argument("--opacity", type=float, default=0.1)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument(
        "--scale-backend",
        choices=("sklearn", "constant"),
        default="sklearn",
        help="Use sklearn KNN scales by default; constant is an explicit fallback.",
    )
    parser.add_argument("--constant-scale", type=float, default=1e-3)
    parser.add_argument("--log-file", help="Optional JSON summary path.")
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


def copy_layout(layout_scene: Path, output_scene: Path) -> None:
    required = [
        "256/images",
        "256/cameras.txt",
        "256/images.txt",
        "256/sparse",
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


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def normalize_rgb(rgb_np: np.ndarray) -> torch.Tensor:
    rgb = torch.as_tensor(rgb_np, dtype=torch.float32)
    if float(rgb.max()) > 1.5:
        rgb = rgb / 255.0
    return rgb.clamp(0.0, 1.0)


def compute_log_scales(points: torch.Tensor, backend: str, constant_scale: float) -> torch.Tensor:
    if backend == "constant":
        print(f"WARNING: using constant Gaussian scale {constant_scale}")
        return torch.full((points.shape[0], 3), float(np.log(constant_scale)), dtype=torch.float32)

    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:
        raise RuntimeError("scale-backend=sklearn requires scikit-learn; use --scale-backend constant to bypass") from exc

    points_np = points.detach().cpu().numpy()
    nn = NearestNeighbors(n_neighbors=4, metric="euclidean").fit(points_np)
    distances, _ = nn.kneighbors(points_np)
    dist2_avg = torch.from_numpy((distances[:, 1:] ** 2).mean(axis=1)).to(torch.float32)
    dist_avg = torch.sqrt(dist2_avg.clamp_min(1e-14))
    return torch.log(dist_avg).unsqueeze(-1).repeat(1, 3)


def build_splats(xyz_np: np.ndarray, rgb_np: np.ndarray, args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    means = torch.as_tensor(xyz_np, dtype=torch.float32)
    rgb = normalize_rgb(rgb_np)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {tuple(means.shape)}")
    if rgb.shape != means.shape:
        raise ValueError(f"rgb shape {tuple(rgb.shape)} does not match xyz shape {tuple(means.shape)}")

    n = means.shape[0]
    scales = compute_log_scales(means, args.scale_backend, args.constant_scale)
    quats = torch.zeros((n, 4), dtype=torch.float32)
    quats[:, 0] = 1.0
    opacities = torch.full((n,), float(torch.logit(torch.tensor(args.opacity))), dtype=torch.float32)
    sh0 = rgb_to_sh(rgb).reshape(n, 1, 3)
    sh_count = (args.sh_degree + 1) ** 2 - 1
    shN = torch.zeros((n, sh_count, 3), dtype=torch.float32)
    return {
        "means": means,
        "scales": scales,
        "quats": quats,
        "opacities": opacities,
        "sh0": sh0,
        "shN": shN,
    }


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
    splats = build_splats(xyz_np, rgb_np, args)

    result["num_splats"] = int(splats["means"].shape[0])
    result["rgb_max"] = float(np.asarray(rgb_np).max())
    result["xyz_min"] = float(np.asarray(xyz_np).min())
    result["xyz_max"] = float(np.asarray(xyz_np).max())

    if args.dry_run:
        result["status"] = "processed"
        result["dry_run"] = True
        return result

    if output_scene.exists():
        shutil.rmtree(output_scene)
    output_scene.mkdir(parents=True, exist_ok=True)
    copy_layout(layout_scene, output_scene)

    ckpt_dir = output_scene / "256" / "gaussian_splatting" / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ckpt_{args.ckpt_step}_rank0.pt"
    torch.save({"splats": splats}, ckpt_path)
    result["ckpt_path"] = str(ckpt_path)
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
        "scale_backend": args.scale_backend,
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
