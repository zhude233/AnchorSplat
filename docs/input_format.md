# Input Format And Normalization

This document describes the input contract used by `inference_external.py`.

AnchorSplat is trained on normalized 3DGS assets, but users should provide PLY files in the original asset coordinate frame. The inference script normalizes coordinates internally, runs the network, and writes the output back in the input coordinate frame.

## Required PLY Fields

The plug-and-play PLY path expects Inria/3DGS-style vertex properties:

```text
x, y, z
f_dc_0, f_dc_1, f_dc_2
opacity
scale_0, scale_1, scale_2
rot_0, rot_1, rot_2, rot_3
```

Optional higher-order SH fields are supported:

```text
f_rest_0, f_rest_1, ...
```

If `f_rest_*` fields are absent, the script pads them with zeros to match `FeaturePredictor.sh_degree = 3`. This is acceptable for sources that only export RGB or SH DC features, but the output quality may differ from inputs with full SH features.

By default, missing required fields raise an error. Use `--allow_missing_features` only for debugging because default-filled scale, rotation, opacity, or color fields are not a faithful 3DGS asset.

## Attribute Conventions

AnchorSplat expects the same parameterization used by common 3D Gaussian Splatting PLY exports:

- `x, y, z`: Gaussian means in the source coordinate frame.
- `scale_0..2`: log-space Gaussian scales. If you have positive scales `sigma`, write `log(sigma)`.
- `opacity`: logit-space opacity. If you have alpha in `[0, 1]`, write `log(alpha / (1 - alpha))` after clamping alpha away from 0 and 1.
- `rot_0..3`: quaternion rotation in the same order as the input 3DGS exporter. The renderer normalizes quaternions at render time.
- `f_dc_0..2`: spherical-harmonic DC coefficients. If you have RGB in `[0, 1]`, convert with `f_dc = (rgb - 0.5) / 0.28209479177387814`.
- `f_rest_*`: flattened non-DC SH coefficients in the Inria 3DGS order. The script converts flattened fields to `(N, coeffs, 3)` internally.

Do not pass sigmoid opacity, positive scales, or RGB colors directly into these fields unless your exporter already follows the same convention.

## Coordinate Normalization

The model operates on normalized means in `[0, 1]^3`. The inference script computes a scalar scale `s` and translation `t`:

```text
means_norm = means * s + t
scales_norm = log_scales + log(s)
```

After prediction, it converts the result back:

```text
means_out = (means_norm_out - t) / s
log_scales_out = scales_norm_out - log(s)
```

Therefore output coordinates and scales are written in the same coordinate frame as the input PLY.

Choose the normalization mode with `--normalization`:

```text
auto      Default. Uses centered mode when the bounding-box center is near the origin, otherwise bbox mode.
centered  For object-centric assets already centered around the origin, but not necessarily scaled.
bbox      For arbitrary translated/scaled assets, COLMAP/world-frame outputs, or real scans not centered at the origin.
unit      For assets already centered and scaled to roughly [-1, 1].
```

For LGM/Trellis/object-centric generated assets, `auto` or `centered` is usually appropriate. For externally reconstructed scenes where coordinates are not centered, use `bbox`.

## Point Count And Memory

The default checkpoint uses `FeaturePredictor.point_multiply_factor = 20`, so the output has approximately `20 * N` Gaussians after input filtering. Large inputs can become expensive quickly.

Use:

```bash
python inference_external.py \
  --weights checkpoints/anchorsplat_20x.pth \
  --input_ply input.ply \
  --output_ply output.ply \
  --model_type lgm \
  --normalization bbox \
  --max_input_gaussians 200000
```

`--max_input_gaussians 0` disables subsampling. Subsampling is random but reproducible with `--random_seed`.

## Output

The output PLY is a standard 3DGS asset with:

```text
x/y/z, normals, f_dc_*, f_rest_*, opacity, scale_*, rot_*
```

It can be loaded by common 3DGS viewers and downstream renderers that support Inria-style Gaussian PLY files.
