"""
Unified inference script for LGM and Trellis generated PLY files.

This script handles the different PLY formats from:
- LGM: No normals, attributes order: x,y,z,f_dc,opacity,scale,rot
- Trellis: Has normals (nx,ny,nz), attributes order: x,y,z,nx,ny,nz,f_dc,opacity,scale,rot
           Also applies coordinate transform during save

Usage:
    python inference_external.py \
        --input_ply /path/to/input.ply \
        --output_ply /path/to/output.ply \
        --model_type lgm|trellis

Example:
    # For LGM generated PLY
    python inference_external.py --input_ply examples/lgm_input.ply \
        --output_ply outputs/lgm_refined.ply --model_type lgm
    
    # For Trellis generated PLY
    python inference_external.py --input_ply examples/trellis_input.ply \
        --output_ply outputs/trellis_refined.ply --model_type trellis
"""

import torch
import numpy as np
import os
import gin
from absl import app, flags
from plyfile import PlyData, PlyElement
from collections import OrderedDict
from models.feature_predictor import FeaturePredictor
from utils.transform_utils import MinMaxScaler
from utils import gs_utils
import dataset.GS
import dataset.Loader  # Register GS_collate_fn # Register SplatfactoDataset

# Default configuration
DEFAULT_WEIGHTS = 'checkpoints/anchorsplat_20x.pth'
DEFAULT_GIN_FILES = [
    'configs/model/ptv3.gin',
]
DEFAULT_GIN_PARAMS = [
    "FeaturePredictor.input_features=['means','scales', 'opacities', 'quats', 'features_dc','features_rest']",
    "FeaturePredictor.output_features=['means','scales', 'opacities', 'quats', 'features_dc','features_rest']",
    "FeaturePredictor.sh_degree=3"
]

flags.DEFINE_string('weights', DEFAULT_WEIGHTS, 'Path to the model weights')
flags.DEFINE_string('input_ply', '', 'Path to the input PLY file (required)')
flags.DEFINE_string('output_ply', '', 'Path to save the output PLY file (required)')
flags.DEFINE_enum('model_type', 'lgm', ['lgm', 'trellis'], 
                  'Type of the source model that generated the PLY file')
flags.DEFINE_enum(
    'normalization',
    'auto',
    ['auto', 'centered', 'bbox', 'unit'],
    'Coordinate normalization for input means. auto chooses centered for object-centric inputs near the origin and bbox otherwise.',
)
flags.DEFINE_float(
    'auto_center_threshold',
    0.10,
    'In auto normalization, treat input as centered when bbox-center/max-abs-coordinate is below this value.',
)
flags.DEFINE_integer(
    'max_input_gaussians',
    0,
    'Optional cap on input Gaussians before 20x expansion. 0 disables subsampling.',
)
flags.DEFINE_integer('random_seed', 918, 'Random seed used when max_input_gaussians subsamples the input.')
flags.DEFINE_boolean('allow_missing_features', False, 'Allow missing scale/rotation/opacity/color fields and fill defaults.')
flags.DEFINE_boolean('strict_weights', True, 'Require checkpoint keys to match the configured model.')
flags.DEFINE_multi_string('gin_file', DEFAULT_GIN_FILES, 'List of paths to the config files.')
flags.DEFINE_multi_string('gin_param', DEFAULT_GIN_PARAMS, 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS

# Note: Trellis applies coordinate transform when saving: [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
# We do NOT apply inverse transform - instead we work in the transformed space
# This avoids complex quaternion transformation and maintains consistency


def vertex_property_set(vertex):
    return {p.name for p in vertex.properties}


def require_vertex_properties(vertex, names, context):
    missing = [name for name in names if name not in vertex_property_set(vertex)]
    if missing and not FLAGS.allow_missing_features:
        raise ValueError(
            f"{context} PLY is missing required fields: {missing}. "
            "AnchorSplat expects Inria-style 3DGS PLY fields: x/y/z, "
            "f_dc_0..2, opacity logits, log scale_0..2, and rot_0..3. "
            "Use --allow_missing_features only for debugging."
        )


def load_checkpoint_state(path):
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    if isinstance(ckpt, dict) and any(k.startswith('module.') for k in ckpt.keys()):
        ckpt = OrderedDict((k[7:] if k.startswith('module.') else k, v) for k, v in ckpt.items())
    return ckpt


def choose_position_scaler(means, mode, auto_center_threshold):
    bbox_min = means.min(dim=0).values
    bbox_max = means.max(dim=0).values
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_range = bbox_max - bbox_min
    max_range = bbox_range.max().item()
    max_abs = means.abs().max().item()

    if max_range < 1e-8 and mode in ['auto', 'bbox']:
        raise ValueError("Input coordinates have near-zero bounding-box extent; cannot bbox-normalize.")
    if max_abs < 1e-8 and mode in ['auto', 'centered', 'unit']:
        raise ValueError("Input coordinates are all near zero; cannot centered-normalize.")

    selected_mode = mode
    center_ratio = bbox_center.abs().max().item() / max(max_abs, 1e-8)
    if mode == 'auto':
        selected_mode = 'centered' if center_ratio <= auto_center_threshold else 'bbox'

    print(f"  BBox min: ({bbox_min[0]:.4f}, {bbox_min[1]:.4f}, {bbox_min[2]:.4f})")
    print(f"  BBox max: ({bbox_max[0]:.4f}, {bbox_max[1]:.4f}, {bbox_max[2]:.4f})")
    print(f"  BBox center/max-abs ratio: {center_ratio:.4f}")
    print(f"  Normalization mode: {selected_mode} (requested: {mode})")

    if selected_mode == 'centered':
        return MinMaxScaler(preserve_ratio=True, already_centered=True, already_scaled=False), selected_mode
    if selected_mode == 'bbox':
        return MinMaxScaler(preserve_ratio=True, already_centered=False, already_scaled=False), selected_mode
    if selected_mode == 'unit':
        return MinMaxScaler(preserve_ratio=True, already_centered=True, already_scaled=True), selected_mode
    raise ValueError(f"Unsupported normalization mode: {mode}")


def subsample_gaussians(gs_params, max_points, seed):
    if max_points <= 0:
        return gs_params
    total = gs_params['means'].shape[0]
    if total <= max_points:
        return gs_params
    generator = torch.Generator(device=gs_params['means'].device)
    generator.manual_seed(seed)
    indices = torch.randperm(total, generator=generator, device=gs_params['means'].device)[:max_points]
    print(f"  Subsampling input Gaussians from {total} to {max_points}")
    for key, value in gs_params.items():
        if value is not None:
            gs_params[key] = value[indices]
    return gs_params


def reshape_inria_sh_rest(features_rest, num_points):
    """Convert flattened 3DGS f_rest_* fields to (N, coeffs, 3)."""
    if features_rest.dim() == 3:
        return features_rest
    if features_rest.shape[1] % 3 != 0:
        raise ValueError(f"features_rest channel count must be divisible by 3, got {features_rest.shape[1]}")
    return features_rest.reshape(num_points, 3, -1).transpose(1, 2).contiguous()


def read_lgm_ply(path, device='cuda'):
    """
    Read PLY file generated by LGM.
    
    LGM PLY format:
    - Attributes: x, y, z, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0, scale_1, scale_2, rot_0-3
    - No normals
    - Scale in log space
    - Opacity in logit space
    - SH DC in (rgb - 0.5) / C0 format
    """
    print(f"Reading LGM PLY from {path}")
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    require_vertex_properties(
        vertex,
        ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
         'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'],
        'LGM',
    )
    
    # Extract positions
    x = torch.tensor(vertex['x'].astype(np.float32))
    y = torch.tensor(vertex['y'].astype(np.float32))
    z = torch.tensor(vertex['z'].astype(np.float32))
    means = torch.stack([x, y, z], dim=1).to(device)
    
    # Extract scales (already in log space)
    try:
        scale_0 = torch.tensor(vertex['scale_0'].astype(np.float32))
        scale_1 = torch.tensor(vertex['scale_1'].astype(np.float32))
        scale_2 = torch.tensor(vertex['scale_2'].astype(np.float32))
        scales = torch.stack([scale_0, scale_1, scale_2], dim=1).to(device)
    except:
        print("Warning: scales not found, initializing with zeros")
        scales = torch.zeros_like(means)

    # Extract rotations (quaternions)
    try:
        rot_0 = torch.tensor(vertex['rot_0'].astype(np.float32))
        rot_1 = torch.tensor(vertex['rot_1'].astype(np.float32))
        rot_2 = torch.tensor(vertex['rot_2'].astype(np.float32))
        rot_3 = torch.tensor(vertex['rot_3'].astype(np.float32))
        quats = torch.stack([rot_0, rot_1, rot_2, rot_3], dim=1).to(device)
    except:
        print("Warning: rotations not found, initializing default")
        quats = torch.zeros((means.shape[0], 4), device=device)
        quats[:, 0] = 1.0

    # Extract opacities (already in logit space)
    try:
        opacities = torch.tensor(vertex['opacity'].astype(np.float32)).unsqueeze(1).to(device)
    except:
        print("Warning: opacity not found")
        opacities = torch.ones((means.shape[0], 1), device=device)

    # Extract SH DC features
    try:
        f_dc_0 = torch.tensor(vertex['f_dc_0'].astype(np.float32))
        f_dc_1 = torch.tensor(vertex['f_dc_1'].astype(np.float32))
        f_dc_2 = torch.tensor(vertex['f_dc_2'].astype(np.float32))
        features_dc = torch.stack([f_dc_0, f_dc_1, f_dc_2], dim=1).to(device)
    except:
        print("Warning: f_dc not found, using zeros")
        features_dc = torch.zeros_like(means)

    # Check for f_rest (higher order SH)
    prop_names = [p.name for p in vertex.properties]
    f_rest_names = [p for p in prop_names if p.startswith('f_rest_')]
    f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(f_rest_names) > 0:
        f_rest_list = []
        for name in f_rest_names:
            f_rest_list.append(torch.tensor(vertex[name].astype(np.float32)))
        features_rest = torch.stack(f_rest_list, dim=1).to(device)
    else:
        features_rest = None

    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'features_dc': features_dc,
        'features_rest': features_rest
    }


def read_trellis_ply(path, device='cuda'):
    """
    Read PLY file generated by Trellis.
    
    Trellis PLY format:
    - Attributes: x, y, z, nx, ny, nz, f_dc_0, f_dc_1, f_dc_2, opacity, scale_0-2, rot_0-3
    - Has normals (all zeros, can be ignored)
    - Coordinate transform was applied during save: [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    - Scale in log space
    - Opacity in logit space
    
    NOTE: We do NOT apply inverse transform here. The model will work in the 
    transformed coordinate space, and the output will be in the same space.
    This avoids complex quaternion transformation issues.
    """
    print(f"Reading Trellis PLY from {path}")
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    require_vertex_properties(
        vertex,
        ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity',
         'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'],
        'Trellis',
    )
    
    # Extract positions directly (no inverse transform)
    # The coordinates are already in the transformed space, we keep them as-is
    x = torch.tensor(vertex['x'].astype(np.float32))
    y = torch.tensor(vertex['y'].astype(np.float32))
    z = torch.tensor(vertex['z'].astype(np.float32))
    means = torch.stack([x, y, z], dim=1).to(device)
    
    # Extract scales (already in log space)
    try:
        scale_0 = torch.tensor(vertex['scale_0'].astype(np.float32))
        scale_1 = torch.tensor(vertex['scale_1'].astype(np.float32))
        scale_2 = torch.tensor(vertex['scale_2'].astype(np.float32))
        scales = torch.stack([scale_0, scale_1, scale_2], dim=1).to(device)
    except:
        print("Warning: scales not found, initializing with zeros")
        scales = torch.zeros_like(means)

    # Extract rotations directly (no inverse transform)
    # Keeping quaternions as-is to maintain consistency with the coordinate space
    try:
        rot_0 = torch.tensor(vertex['rot_0'].astype(np.float32))
        rot_1 = torch.tensor(vertex['rot_1'].astype(np.float32))
        rot_2 = torch.tensor(vertex['rot_2'].astype(np.float32))
        rot_3 = torch.tensor(vertex['rot_3'].astype(np.float32))
        quats = torch.stack([rot_0, rot_1, rot_2, rot_3], dim=1).to(device)
    except:
        print("Warning: rotations not found, initializing default")
        quats = torch.zeros((means.shape[0], 4), device=device)
        quats[:, 0] = 1.0

    # Extract opacities (already in logit space)
    try:
        opacities = torch.tensor(vertex['opacity'].astype(np.float32)).unsqueeze(1).to(device)
    except:
        print("Warning: opacity not found")
        opacities = torch.ones((means.shape[0], 1), device=device)

    # Extract SH DC features
    try:
        f_dc_0 = torch.tensor(vertex['f_dc_0'].astype(np.float32))
        f_dc_1 = torch.tensor(vertex['f_dc_1'].astype(np.float32))
        f_dc_2 = torch.tensor(vertex['f_dc_2'].astype(np.float32))
        features_dc = torch.stack([f_dc_0, f_dc_1, f_dc_2], dim=1).to(device)
    except:
        print("Warning: f_dc not found, using zeros")
        features_dc = torch.zeros_like(means)

    # Check for f_rest (higher order SH) - Trellis typically doesn't have these
    prop_names = [p.name for p in vertex.properties]
    f_rest_names = [p for p in prop_names if p.startswith('f_rest_')]
    f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(f_rest_names) > 0:
        f_rest_list = []
        for name in f_rest_names:
            f_rest_list.append(torch.tensor(vertex[name].astype(np.float32)))
        features_rest = torch.stack(f_rest_list, dim=1).to(device)
    else:
        features_rest = None

    return {
        'means': means,
        'scales': scales,
        'quats': quats,
        'opacities': opacities,
        'features_dc': features_dc,
        'features_rest': features_rest
    }


def clean_gaussian_data(gs_params, device):
    """
    Clean up gaussian parameters by removing invalid points (NaN, Inf).
    """
    print("Cleaning input data...")
    total_points = gs_params['means'].shape[0]
    valid_mask = torch.ones(total_points, dtype=torch.bool, device=device)
    
    field_stats = {}
    for name, tensor in [('means', gs_params['means']), 
                         ('scales', gs_params['scales']), 
                         ('quats', gs_params['quats']), 
                         ('opacities', gs_params['opacities']), 
                         ('features_dc', gs_params['features_dc'])]:
        nan_count = torch.isnan(tensor).any(dim=-1).sum().item()
        inf_count = torch.isinf(tensor).any(dim=-1).sum().item()
        
        if nan_count > 0 or inf_count > 0:
            field_stats[name] = {'nan': nan_count, 'inf': inf_count}
            valid_mask &= ~torch.isnan(tensor).any(dim=-1)
            valid_mask &= ~torch.isinf(tensor).any(dim=-1)
    
    if gs_params['features_rest'] is not None:
        rest_flat = gs_params['features_rest'].reshape(gs_params['features_rest'].shape[0], -1)
        nan_count = torch.isnan(rest_flat).any(dim=-1).sum().item()
        inf_count = torch.isinf(rest_flat).any(dim=-1).sum().item()
        if nan_count > 0 or inf_count > 0:
            field_stats['features_rest'] = {'nan': nan_count, 'inf': inf_count}
            valid_mask &= ~torch.isnan(rest_flat).any(dim=-1)
            valid_mask &= ~torch.isinf(rest_flat).any(dim=-1)
    
    if field_stats:
        print(f"  Found invalid values:")
        for name, stats in field_stats.items():
            print(f"    {name}: {stats['nan']} NaN, {stats['inf']} Inf")
    
    num_invalid = (~valid_mask).sum().item()
    if num_invalid > 0:
        print(f"  Filtering out {num_invalid} invalid points ({num_invalid/total_points*100:.2f}%)")
        for key in gs_params:
            if gs_params[key] is not None:
                gs_params[key] = gs_params[key][valid_mask]
        print(f"  Remaining points: {gs_params['means'].shape[0]}")
    else:
        print(f"  No invalid points found. Total points: {total_points}")
    
    return gs_params


def post_process_output(gs_params, device):
    """Clean up the output gaussian parameters to prevent artifacts."""
    print("Post-processing output data...")
    total = gs_params['means'].shape[0]
    valid_mask = torch.ones(total, dtype=torch.bool, device=device)
    
    # Filter NaN/Inf
    for key in gs_params:
        if isinstance(gs_params[key], torch.Tensor):
            tensor_flat = gs_params[key].view(total, -1)
            is_nan = torch.isnan(tensor_flat).any(dim=-1)
            is_inf = torch.isinf(tensor_flat).any(dim=-1)
            valid_mask &= ~is_nan
            valid_mask &= ~is_inf
    
    # Clamp values to prevent extreme artifacts
    if 'opacities' in gs_params:
        gs_params['opacities'] = torch.clamp(gs_params['opacities'], min=-20.0, max=20.0)
    if 'scales' in gs_params:
        gs_params['scales'] = torch.clamp(gs_params['scales'], max=5.0)
    if 'features_dc' in gs_params:
        gs_params['features_dc'] = torch.clamp(gs_params['features_dc'], min=-10.0, max=10.0)
    if 'features_rest' in gs_params and gs_params['features_rest'] is not None:
        gs_params['features_rest'] = torch.clamp(gs_params['features_rest'], min=-10.0, max=10.0)

    # Apply filter
    if not valid_mask.all():
        num_invalid = (~valid_mask).sum().item()
        print(f"  Filtering out {num_invalid} points with NaN/Inf in output ({num_invalid/total*100:.2f}%)")
        for key in gs_params:
            if isinstance(gs_params[key], torch.Tensor):
                gs_params[key] = gs_params[key][valid_mask]
    else:
        print("  No invalid points found in output.")
        
    return gs_params


def main(argv):
    # Check required arguments
    if not FLAGS.input_ply:
        raise ValueError("--input_ply is required. Please provide the path to the input PLY file.")
    if not FLAGS.output_ply:
        raise ValueError("--output_ply is required. Please provide the path to save the output PLY file.")
    
    print(f"\n{'='*60}")
    print(f"External PLY Inference Script")
    print(f"{'='*60}")
    print(f"Input PLY: {FLAGS.input_ply}")
    print(f"Output PLY: {FLAGS.output_ply}")
    print(f"Model Type: {FLAGS.model_type}")
    print(f"{'='*60}\n")
    
    # Parse gin config
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param, skip_unknown=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Model
    print("\nBuilding model...")
    model = FeaturePredictor()
    model.to(device)
    
    print(f"Loading weights from {FLAGS.weights}")
    ckpt = load_checkpoint_state(FLAGS.weights)
    load_msg = model.load_state_dict(ckpt, strict=FLAGS.strict_weights)
    if not FLAGS.strict_weights:
        print(f"  Missing keys: {load_msg.missing_keys}")
        print(f"  Unexpected keys: {load_msg.unexpected_keys}")
    model.eval()
    
    # 2. Load Input PLY based on model type
    print(f"\nLoading input PLY ({FLAGS.model_type} format)...")
    if FLAGS.model_type == 'lgm':
        gs_params = read_lgm_ply(FLAGS.input_ply, device=device)
    elif FLAGS.model_type == 'trellis':
        gs_params = read_trellis_ply(FLAGS.input_ply, device=device)
    else:
        raise ValueError(f"Unknown model type: {FLAGS.model_type}")
    
    print(f"  Loaded {gs_params['means'].shape[0]} gaussians")
    
    # 3. Clean data
    gs_params = clean_gaussian_data(gs_params, device)
    gs_params = subsample_gaussians(gs_params, FLAGS.max_input_gaussians, FLAGS.random_seed)
    
    # 4. Handle SH padding
    target_sh_degree = model.sh_degree
    target_sh_dim = (target_sh_degree + 1) ** 2 - 1
    target_rest_channels = target_sh_dim * 3
    
    current_rest = gs_params['features_rest']
    
    if current_rest is None:
        print(f"\nInput has no SH rest, padding to {target_rest_channels} channels")
        gs_params['features_rest'] = torch.zeros((gs_params['means'].shape[0], target_rest_channels), device=device)
    else:
        current_rest = current_rest.reshape(current_rest.shape[0], -1)
        current_channels = current_rest.shape[1]
        if current_channels < target_rest_channels:
            print(f"\nInput SH channels {current_channels} < target {target_rest_channels}, padding with zeros")
            padding = torch.zeros((gs_params['means'].shape[0], target_rest_channels - current_channels), device=device)
            gs_params['features_rest'] = torch.cat([current_rest, padding], dim=1)
        elif current_channels > target_rest_channels:
            print(f"\nInput SH channels {current_channels} > target {target_rest_channels}, truncating")
            gs_params['features_rest'] = current_rest[:, :target_rest_channels]
    
    gs_params['features_rest'] = reshape_inria_sh_rest(
        gs_params['features_rest'],
        gs_params['means'].shape[0],
    )
    
    # 5. Normalize (matching training code in GS.py)
    print("\nNormalizing data...")
    means_min = gs_params['means'].min().item()
    means_max = gs_params['means'].max().item()
    means_mean = gs_params['means'].mean(dim=0)
    
    print(f"  Coordinate range: [{means_min:.4f}, {means_max:.4f}]")
    print(f"  Center (mean): ({means_mean[0]:.4f}, {means_mean[1]:.4f}, {means_mean[2]:.4f})")
    
    scaler, selected_norm = choose_position_scaler(
        gs_params['means'],
        FLAGS.normalization,
        FLAGS.auto_center_threshold,
    )

    # Apply normalization
    normalized_means = scaler.fit_transform(gs_params['means'])
    normalized_scales = gs_params['scales'] + torch.log(scaler.scale_)
    
    # Filter points: remove points with inf scales or means outside [0, 1]
    # This matches the training code in GS.py
    inf_mask = torch.isinf(normalized_scales).any(dim=1)
    inrange_mask = torch.all((normalized_means >= 0) & (normalized_means <= 1), dim=1)
    valid_mask = (~inf_mask) & inrange_mask
    
    num_filtered = (~valid_mask).sum().item()
    if num_filtered > 0:
        print(f"  Filtering {num_filtered} points outside [0,1] or with inf scales")
        normalized_means = normalized_means[valid_mask]
        normalized_scales = normalized_scales[valid_mask]
        gs_params['features_dc'] = gs_params['features_dc'][valid_mask]
        gs_params['features_rest'] = gs_params['features_rest'][valid_mask]
        gs_params['opacities'] = gs_params['opacities'][valid_mask]
        gs_params['quats'] = gs_params['quats'][valid_mask]
        print(f"  Remaining points after filtering: {normalized_means.shape[0]}")
    
    print(f"  Normalized range: [{normalized_means.min():.4f}, {normalized_means.max():.4f}]")
    print(f"  Scale shift applied to log-scales: log({float(scaler.scale_):.6g})")
    
    normalized_gs = {}
    normalized_gs['means'] = normalized_means
    normalized_gs['scales'] = normalized_scales
    normalized_gs['features_dc'] = gs_params['features_dc']
    normalized_gs['features_rest'] = gs_params['features_rest']
    normalized_gs['opacities'] = gs_params['opacities']
    normalized_gs['quats'] = gs_params['quats']
    
    # 6. Inference
    print("\nRunning inference...")
    batch_normalized_gs = [normalized_gs]
    batch_scene_idx = [0]
    
    with torch.no_grad():
        out_batch = model(batch_normalized_gs, batch_scene_idx)
    
    out_gs = out_batch[0]
    
    # 7. Un-normalize
    print("\nUn-normalizing output...")
    final_gs = {}
    
    target_device = out_gs['means'].device
    if isinstance(scaler.trans_, torch.Tensor):
        scaler.trans_ = scaler.trans_.to(target_device)
    if isinstance(scaler.scale_, torch.Tensor):
        scaler.scale_ = scaler.scale_.to(target_device)

    final_gs['means'] = scaler.inverse_transform(out_gs['means'])
    final_gs['scales'] = out_gs['scales'] - torch.log(scaler.scale_)
    final_gs['features_dc'] = out_gs['features_dc']
    final_gs['features_rest'] = out_gs['features_rest']
    final_gs['opacities'] = out_gs['opacities']
    final_gs['quats'] = out_gs['quats']
    
    # 8. Post-process
    final_gs = post_process_output(final_gs, target_device)
    
    # 9. Save
    print(f"\nSaving to {FLAGS.output_ply}")
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.output_ply)), exist_ok=True)
    gs_utils.export_ply_forviewer(final_gs, FLAGS.output_ply)
    
    print(f"\n{'='*60}")
    print(f"Done! Output saved to {FLAGS.output_ply}")
    print(f"Output gaussians: {final_gs['means'].shape[0]}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    app.run(main)
