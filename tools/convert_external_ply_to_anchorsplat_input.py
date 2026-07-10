#!/usr/bin/env python3
"""Convert external 3DGS PLY files to AnchorSplat's normalized input distribution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.anchorsplat_external_utils import (  # noqa: E402
    clean_gaussians,
    gaussian_state_for_save,
    normalize_gaussians,
    pad_or_truncate_sh_rest,
    read_external_ply,
    subsample_gaussians,
    write_gaussian_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read an Inria/LGM/Trellis Gaussian PLY, normalize coordinates to "
            "AnchorSplat's [0,1]^3 input distribution, shift log-scales by the "
            "same scalar, and write a normalized PLY/PT bundle."
        )
    )
    parser.add_argument("--input-ply", required=True, type=Path, help="External 3DGS PLY file.")
    parser.add_argument(
        "--source-format",
        default="inria",
        choices=("inria", "lgm", "trellis"),
        help="PLY producer. Use inria for COLMAP-trained 3DGS real scenes.",
    )
    parser.add_argument(
        "--normalization",
        default="auto",
        choices=("auto", "bbox", "centered", "unit"),
        help=(
            "bbox is recommended for COLMAP/world-frame real scenes; centered is "
            "recommended for object-centric Trellis assets; auto chooses from bbox center."
        ),
    )
    parser.add_argument(
        "--auto-center-threshold",
        default=0.10,
        type=float,
        help="Auto mode selects centered when bbox-center/max-abs-coordinate is below this value.",
    )
    parser.add_argument("--sh-degree", default=3, type=int, help="Target SH degree expected by AnchorSplat.")
    parser.add_argument(
        "--max-input-gaussians",
        default=0,
        type=int,
        help="Optional random cap before normalization. 0 disables subsampling.",
    )
    parser.add_argument("--random-seed", default=918, type=int, help="Seed used for optional subsampling.")
    parser.add_argument(
        "--allow-missing-features",
        action="store_true",
        help="Fill missing 3DGS fields with defaults. Intended only for debugging non-standard PLYs.",
    )
    parser.add_argument("--output-ply", type=Path, help="Normalized Inria-compatible PLY for inspection.")
    parser.add_argument("--output-pt", type=Path, help="Torch bundle containing normalized tensors and metadata.")
    parser.add_argument("--metadata-json", type=Path, help="JSON sidecar with normalization metadata.")
    return parser.parse_args()


def tensor_shape_dict(bundle: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in bundle.items()}


def main() -> None:
    args = parse_args()
    if not args.output_ply and not args.output_pt and not args.metadata_json:
        raise SystemExit("Please provide at least one of --output-ply, --output-pt, or --metadata-json.")

    print(f"Input PLY: {args.input_ply}")
    print(f"Source format: {args.source_format}")
    print(f"Requested normalization: {args.normalization}")

    gs_params = read_external_ply(
        args.input_ply,
        args.source_format,
        device="cpu",
        allow_missing_features=args.allow_missing_features,
    )
    loaded_count = int(gs_params["means"].shape[0])
    print(f"Loaded Gaussians: {loaded_count}")

    gs_params, removed_invalid = clean_gaussians(gs_params)
    if removed_invalid:
        print(f"Removed invalid Gaussians: {removed_invalid}")

    gs_params, removed_subsample = subsample_gaussians(
        gs_params,
        args.max_input_gaussians,
        args.random_seed,
    )
    if removed_subsample:
        print(f"Subsampled Gaussians: {loaded_count - removed_invalid} -> {args.max_input_gaussians}")

    gs_params, sh_info = pad_or_truncate_sh_rest(gs_params, args.sh_degree)
    normalized, norm_meta, valid_mask = normalize_gaussians(
        gs_params,
        args.normalization,
        args.auto_center_threshold,
    )

    kept_count = int(normalized["means"].shape[0])
    print(f"Selected normalization: {norm_meta['selected']}")
    print(f"Scale: {norm_meta['scale']}")
    print(f"Translation: {norm_meta['translation']}")
    print(f"Scale log shift: {norm_meta['scale_log_shift']:.8f}")
    print(f"Kept Gaussians: {kept_count}")
    print(f"Filtered after normalization: {norm_meta['filtered_after_normalization']}")
    print(f"Normalized bbox min: {norm_meta['normalized_bbox_min']}")
    print(f"Normalized bbox max: {norm_meta['normalized_bbox_max']}")

    metadata = {
        "source_ply": str(args.input_ply),
        "source_format": args.source_format,
        "normalization": norm_meta,
        "sh": sh_info,
        "counts": {
            "loaded": loaded_count,
            "removed_invalid": removed_invalid,
            "removed_subsample": removed_subsample,
            "kept": kept_count,
        },
        "tensor_shapes": tensor_shape_dict(gaussian_state_for_save(normalized)),
        "restore_formula": {
            "means_original": "(means_normalized - translation) / scale",
            "log_scales_original": "log_scales_normalized - log(scale)",
        },
    }

    if args.output_ply:
        write_gaussian_ply(normalized, args.output_ply)
        print(f"Wrote normalized PLY: {args.output_ply}")

    if args.output_pt:
        args.output_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "gaussians": gaussian_state_for_save(normalized),
                "metadata": metadata,
            },
            args.output_pt,
        )
        print(f"Wrote tensor bundle: {args.output_pt}")

    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote metadata: {args.metadata_json}")


if __name__ == "__main__":
    main()
