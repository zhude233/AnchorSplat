"""Utilities for converting external 3DGS PLY files to AnchorSplat inputs."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement


GaussianDict = Dict[str, Optional[torch.Tensor]]


class AnchorSplatScaler:
    """Small scaler matching utils.transform_utils.MinMaxScaler for external inputs."""

    def __init__(
        self,
        feature_range: tuple[float, float] = (0.0, 1.0),
        preserve_ratio: bool = True,
        already_centered: bool = True,
        already_scaled: bool = True,
    ) -> None:
        if not preserve_ratio:
            raise ValueError("AnchorSplat external conversion requires preserve_ratio=True.")
        if already_scaled and not already_centered:
            raise ValueError("already_scaled=True requires already_centered=True.")
        self.feature_range = feature_range
        self.preserve_ratio = preserve_ratio
        self.already_centered = already_centered
        self.already_scaled = already_scaled
        self.scale_: Optional[torch.Tensor] = None
        self.trans_: Optional[torch.Tensor] = None

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.already_centered and not self.already_scaled:
            data_min = torch.min(x, dim=0).values
            data_max = torch.max(x, dim=0).values
            data_range = data_max - data_min
            out_min, out_max = self.feature_range
            center = (out_min + out_max) / 2.0
            scale = (out_max - out_min) / data_range
            self.scale_ = torch.min(scale)
            scaled = x * self.scale_
            scaled_mid = (scaled.min(dim=0).values + scaled.max(dim=0).values) * 0.5
            self.trans_ = torch.as_tensor(center, device=x.device, dtype=x.dtype) - scaled_mid
        else:
            if self.feature_range != (0.0, 1.0):
                raise ValueError("Centered scaling expects feature_range=(0, 1).")
            self.trans_ = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype)
            if not self.already_scaled:
                self.scale_ = 0.5 / torch.abs(x).max()
            else:
                self.scale_ = torch.tensor(0.5, device=x.device, dtype=x.dtype)
            scaled = x * self.scale_
        return scaled + self.trans_


def _property_names(vertex) -> set[str]:
    return {prop.name for prop in vertex.properties}


def _sorted_numbered_properties(names: Iterable[str], prefix: str) -> list[str]:
    selected = [name for name in names if name.startswith(prefix)]
    return sorted(selected, key=lambda name: int(name.split("_")[-1]))


def _require_properties(vertex, names: Iterable[str], context: str, allow_missing: bool) -> None:
    missing = [name for name in names if name not in _property_names(vertex)]
    if missing and not allow_missing:
        raise ValueError(
            f"{context} PLY is missing required fields: {missing}. "
            "Expected Inria-style 3DGS fields: x/y/z, f_dc_0..2, opacity, "
            "scale_0..2, and rot_0..3."
        )


def read_external_ply(
    path: str | Path,
    source_format: str,
    device: str | torch.device = "cpu",
    allow_missing_features: bool = False,
) -> GaussianDict:
    """Read Inria/LGM-style or Trellis-style Gaussian PLY files.

    Trellis PLYs often include nx/ny/nz normal fields. They are ignored. Trellis
    also saves coordinates after its own axis transform; AnchorSplat keeps that
    transformed coordinate frame and later restores outputs to the same frame.
    """
    if source_format not in {"inria", "lgm", "trellis"}:
        raise ValueError(f"Unsupported source_format: {source_format}")

    ply = PlyData.read(str(path))
    vertex = ply["vertex"]
    context = "Trellis" if source_format == "trellis" else "Inria/LGM-style"
    _require_properties(
        vertex,
        [
            "x",
            "y",
            "z",
            "f_dc_0",
            "f_dc_1",
            "f_dc_2",
            "opacity",
            "scale_0",
            "scale_1",
            "scale_2",
            "rot_0",
            "rot_1",
            "rot_2",
            "rot_3",
        ],
        context,
        allow_missing_features,
    )

    names = _property_names(vertex)
    n = len(vertex)

    def get_column(name: str, default: float = 0.0) -> torch.Tensor:
        if name in names:
            return torch.from_numpy(np.asarray(vertex[name], dtype=np.float32))
        return torch.full((n,), float(default), dtype=torch.float32)

    means = torch.stack([get_column("x"), get_column("y"), get_column("z")], dim=1)
    scales = torch.stack(
        [get_column("scale_0"), get_column("scale_1"), get_column("scale_2")],
        dim=1,
    )
    quats = torch.stack(
        [
            get_column("rot_0", 1.0),
            get_column("rot_1", 0.0),
            get_column("rot_2", 0.0),
            get_column("rot_3", 0.0),
        ],
        dim=1,
    )
    opacities = get_column("opacity", 1.0).unsqueeze(1)
    features_dc = torch.stack(
        [get_column("f_dc_0"), get_column("f_dc_1"), get_column("f_dc_2")],
        dim=1,
    )

    f_rest_names = _sorted_numbered_properties(names, "f_rest_")
    features_rest = None
    if f_rest_names:
        features_rest = torch.stack([get_column(name) for name in f_rest_names], dim=1)

    return {
        "means": means.to(device),
        "scales": scales.to(device),
        "quats": quats.to(device),
        "opacities": opacities.to(device),
        "features_dc": features_dc.to(device),
        "features_rest": None if features_rest is None else features_rest.to(device),
    }


def clean_gaussians(gs_params: GaussianDict) -> Tuple[GaussianDict, int]:
    """Remove Gaussians containing NaN/Inf in any available field."""
    total = int(gs_params["means"].shape[0])
    device = gs_params["means"].device
    valid = torch.ones(total, dtype=torch.bool, device=device)

    for value in gs_params.values():
        if value is None:
            continue
        flat = value.reshape(total, -1)
        valid &= torch.isfinite(flat).all(dim=1)

    removed = int((~valid).sum().item())
    if removed:
        for key, value in list(gs_params.items()):
            if value is not None:
                gs_params[key] = value[valid]
    return gs_params, removed


def subsample_gaussians(gs_params: GaussianDict, max_points: int, seed: int) -> Tuple[GaussianDict, int]:
    if max_points <= 0:
        return gs_params, 0
    total = int(gs_params["means"].shape[0])
    if total <= max_points:
        return gs_params, 0
    generator = torch.Generator(device=gs_params["means"].device)
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator, device=gs_params["means"].device)[:max_points]
    for key, value in list(gs_params.items()):
        if value is not None:
            gs_params[key] = value[indices]
    return gs_params, total - max_points


def choose_position_scaler(
    means: torch.Tensor,
    mode: str,
    auto_center_threshold: float,
) -> Tuple[AnchorSplatScaler, str, Dict[str, object]]:
    if mode not in {"auto", "centered", "bbox", "unit"}:
        raise ValueError(f"Unsupported normalization mode: {mode}")

    bbox_min = means.min(dim=0).values
    bbox_max = means.max(dim=0).values
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_range = bbox_max - bbox_min
    max_range = float(bbox_range.max().item())
    max_abs = float(means.abs().max().item())
    center_ratio = float(bbox_center.abs().max().item() / max(max_abs, 1e-8))

    if max_range < 1e-8 and mode in {"auto", "bbox"}:
        raise ValueError("Input coordinates have near-zero bounding-box extent; cannot bbox-normalize.")
    if max_abs < 1e-8 and mode in {"auto", "centered", "unit"}:
        raise ValueError("Input coordinates are all near zero; cannot centered-normalize.")

    selected_mode = mode
    if mode == "auto":
        selected_mode = "centered" if center_ratio <= auto_center_threshold else "bbox"

    if selected_mode == "centered":
        scaler = AnchorSplatScaler(preserve_ratio=True, already_centered=True, already_scaled=False)
    elif selected_mode == "bbox":
        scaler = AnchorSplatScaler(preserve_ratio=True, already_centered=False, already_scaled=False)
    elif selected_mode == "unit":
        scaler = AnchorSplatScaler(preserve_ratio=True, already_centered=True, already_scaled=True)
    else:
        raise ValueError(f"Unsupported selected normalization mode: {selected_mode}")

    stats = {
        "bbox_min": bbox_min.detach().cpu().tolist(),
        "bbox_max": bbox_max.detach().cpu().tolist(),
        "bbox_center": bbox_center.detach().cpu().tolist(),
        "bbox_range": bbox_range.detach().cpu().tolist(),
        "max_range": max_range,
        "max_abs": max_abs,
        "center_ratio": center_ratio,
    }
    return scaler, selected_mode, stats


def reshape_inria_sh_rest(features_rest: torch.Tensor, num_points: int) -> torch.Tensor:
    """Convert flattened Inria f_rest_* fields to (N, coeffs, 3)."""
    if features_rest.dim() == 3:
        return features_rest
    if features_rest.shape[1] % 3 != 0:
        raise ValueError(f"features_rest channel count must be divisible by 3, got {features_rest.shape[1]}")
    return features_rest.reshape(num_points, 3, -1).transpose(1, 2).contiguous()


def pad_or_truncate_sh_rest(gs_params: GaussianDict, sh_degree: int) -> Tuple[GaussianDict, Dict[str, int]]:
    target_channels = ((sh_degree + 1) ** 2 - 1) * 3
    num_points = int(gs_params["means"].shape[0])
    current = gs_params.get("features_rest")

    if current is None:
        current_flat = torch.zeros((num_points, target_channels), device=gs_params["means"].device)
        original_channels = 0
    else:
        current_flat = current.reshape(num_points, -1)
        original_channels = int(current_flat.shape[1])
        if original_channels < target_channels:
            pad = torch.zeros(
                (num_points, target_channels - original_channels),
                device=current_flat.device,
                dtype=current_flat.dtype,
            )
            current_flat = torch.cat([current_flat, pad], dim=1)
        elif original_channels > target_channels:
            current_flat = current_flat[:, :target_channels]

    gs_params["features_rest"] = reshape_inria_sh_rest(current_flat, num_points)
    return gs_params, {
        "original_flat_channels": original_channels,
        "target_flat_channels": target_channels,
        "output_coefficients": int(gs_params["features_rest"].shape[1]),
    }


def normalize_gaussians(
    gs_params: GaussianDict,
    normalization: str,
    auto_center_threshold: float,
) -> Tuple[GaussianDict, Dict[str, object], torch.Tensor]:
    means = gs_params["means"]
    scales = gs_params["scales"]
    scaler, selected_mode, stats = choose_position_scaler(means, normalization, auto_center_threshold)

    normalized_means = scaler.fit_transform(means)
    normalized_scales = scales + torch.log(scaler.scale_)

    valid = torch.isfinite(normalized_scales).all(dim=1)
    valid &= torch.all((normalized_means >= 0) & (normalized_means <= 1), dim=1)

    normalized = {
        "means": normalized_means[valid],
        "scales": normalized_scales[valid],
        "features_dc": gs_params["features_dc"][valid],
        "features_rest": gs_params["features_rest"][valid],
        "opacities": gs_params["opacities"][valid],
        "quats": gs_params["quats"][valid],
    }

    scale_tensor = scaler.scale_.detach().cpu()
    trans_tensor = scaler.trans_.detach().cpu()
    metadata = {
        "requested": normalization,
        "selected": selected_mode,
        "scale": float(scale_tensor.item()) if scale_tensor.numel() == 1 else scale_tensor.tolist(),
        "translation": trans_tensor.tolist(),
        "scale_log_shift": float(torch.log(scaler.scale_).detach().cpu().item()),
        "stats": stats,
        "filtered_after_normalization": int((~valid).sum().item()),
        "normalized_bbox_min": normalized["means"].min(dim=0).values.detach().cpu().tolist(),
        "normalized_bbox_max": normalized["means"].max(dim=0).values.detach().cpu().tolist(),
    }
    return normalized, metadata, valid


def gaussian_state_for_save(gs_params: GaussianDict) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in gs_params.items()
        if isinstance(value, torch.Tensor)
    }


def write_gaussian_ply(gs_params: GaussianDict, path: str | Path) -> None:
    """Write an Inria-compatible Gaussian PLY."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gs_cpu = gaussian_state_for_save(gs_params)
    n = int(gs_cpu["means"].shape[0])
    fields = OrderedDict()
    fields["x"] = gs_cpu["means"][:, 0].numpy()
    fields["y"] = gs_cpu["means"][:, 1].numpy()
    fields["z"] = gs_cpu["means"][:, 2].numpy()
    fields["nx"] = np.zeros(n, dtype=np.float32)
    fields["ny"] = np.zeros(n, dtype=np.float32)
    fields["nz"] = np.zeros(n, dtype=np.float32)

    for i in range(3):
        fields[f"f_dc_{i}"] = gs_cpu["features_dc"][:, i].numpy()

    features_rest = gs_cpu.get("features_rest")
    if features_rest is not None and features_rest.numel() > 0:
        if features_rest.dim() == 3:
            rest = features_rest.transpose(1, 2).contiguous().reshape(n, -1).numpy()
        else:
            rest = features_rest.reshape(n, -1).numpy()
        for i in range(rest.shape[1]):
            fields[f"f_rest_{i}"] = rest[:, i]

    fields["opacity"] = gs_cpu["opacities"].reshape(n).numpy()
    for i in range(3):
        fields[f"scale_{i}"] = gs_cpu["scales"][:, i].numpy()
    for i in range(4):
        fields[f"rot_{i}"] = gs_cpu["quats"][:, i].numpy()

    dtype = [(name, "f4") for name in fields]
    elements = np.empty(n, dtype=dtype)
    columns = np.stack([np.asarray(value, dtype=np.float32).reshape(n) for value in fields.values()], axis=1)
    elements[:] = list(map(tuple, columns))
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
