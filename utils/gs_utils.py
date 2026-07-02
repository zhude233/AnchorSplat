import torch
import gin 
import gsplat 
import math
import numpy as np
import os, cv2
from collections import OrderedDict
from plyfile import PlyData, PlyElement
import json
from argparse import Namespace
import torch_scatter

# 尝试导入新版API
try:
    from gsplat import rasterization
    GSPLAT_NEW_API = True
    print("Using new gsplat API (memory efficient)")
except ImportError:
    GSPLAT_NEW_API = False
    print("Using legacy gsplat API")

BLOCK_WIDTH = 16 

C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def rasterize_gaussians_to_multiimgs(gs_params, cameras):
    camera_to_worlds = cameras['camera_to_worlds']
    rgbs, alphas = [], []
    for camera_to_world in camera_to_worlds:
        rgb, alpha = rasterize_gaussians_to_singleimg(gs_params, camera_to_world, **cameras)
        rgbs.append(rgb)
        alphas.append(alpha)
    return rgbs, alphas

def rasterize_gaussians_to_singleimg(gs_params, camera_to_world, cx, cy, fx, fy, width, height, background_color, **kwargs):
    """
    使用最新gsplat API进行渲染，显存效率更高
    对于大规模点云（50w+），新API可以将显存从45G降低到2-5G
    """
    #Turn half to float
    gs_params = {k:v.float() if v.dtype==torch.half else v for k,v in gs_params.items()}
    R = camera_to_world[:3, :3]
    T = camera_to_world[:3, 3:4]
    # flip the z and y axes to align with gsplat conventions (opengl/blender to opencv/colmap)
    R_edit = torch.diag(torch.tensor([1, -1, -1], device='cuda', dtype=R.dtype))
    R = R @ R_edit
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv

    means = gs_params['means']
    scales = torch.exp(gs_params['scales'])
    quats = gs_params['quats']/torch.norm(gs_params['quats'], dim=-1, keepdim=True)
    mask = (quats.norm(dim=-1) - 1)<1e-6
    inv_mask = ~mask
    if inv_mask.sum() > 0:
        # print(f"Warning: {mask.sum()} quaternions are not normalized")
        quats[inv_mask] = torch.tensor([0, 0, 0, 1.], device=quats.device)

    if 'opacities' in gs_params:
        opacities = torch.sigmoid(gs_params['opacities'])
    elif 'opacities_sigmoid' in gs_params:
        opacities = gs_params['opacities_sigmoid']
    else:
        raise ValueError("No opacities found in gs_params")
    
    if 'features_rest' in gs_params:
        colors = torch.cat([gs_params['features_dc'].unsqueeze(1), gs_params['features_rest']], dim=1)
    else:
        colors = gs_params['features_dc'].unsqueeze(1)
    
    H, W = int(height.item()), int(width.item())
    
    if GSPLAT_NEW_API:
        # ========== 使用新版gsplat API (高效显存) ==========
        # 构建相机内参矩阵 K (3x3)
        K = torch.tensor([
            [fx.item(), 0, cx.item()],
            [0, fy.item(), cy.item()],
            [0, 0, 1]
        ], device=means.device, dtype=torch.float32)
        
        # 新版API关键参数说明:
        # - packed=False: 不使用packed模式，对于单图渲染更高效
        # - absgrad=False: 推理时不需要梯度
        # - sparse_grad=False: 不使用稀疏梯度
        # - rasterize_mode="classic": 经典渲染模式，显存效率最高
        background = background_color.to(device=means.device, dtype=torch.float32).reshape(1, 3)
        render_colors, render_alphas, info = rasterization(
            means=means,                    # (N, 3)
            quats=quats,                    # (N, 4) WXYZ格式
            scales=scales,                  # (N, 3)
            opacities=opacities.squeeze(-1) if opacities.dim() > 1 else opacities,  # (N,)
            colors=colors,                  # (N, K, 3) for SH or (N, 1, 3) for RGB
            viewmats=viewmat[None],         # (1, 4, 4) 完整的4x4矩阵
            Ks=K[None],                     # (1, 3, 3)
            width=W,
            height=H,
            packed=False,                   # 单图渲染用False更高效
            absgrad=False,                  # 推理时关闭
            sparse_grad=False,              # 推理时关闭  
            rasterize_mode="classic",       # classic模式显存最优
            sh_degree=None if colors.shape[1] == 1 else int(math.sqrt(colors.shape[1]) - 1),
            backgrounds=background,
        )
        
        rgb = render_colors.squeeze(0)  # (H, W, 3)
        alpha = render_alphas.squeeze(0).unsqueeze(-1)  # (H, W, 1)
        
    else:
        # ========== 回退到旧版API ==========
        n = int(math.sqrt(colors.shape[1])-1)
        if n==0:
            rgbs = torch.sigmoid(colors[:,0,:])
        else:
            viewdirs_ = means.detach() - camera_to_world.detach()[:3, 3]  # (N, 3)
            viewdirs_norm = viewdirs_.norm(dim=-1, keepdim=True)
            viewdirs = viewdirs_ / viewdirs_norm
            ## In some extremely rare case, the gs mean can be the same as the camera position
            if torch.isnan(viewdirs).any():
                mask_ = (viewdirs_norm==0).squeeze() #N,
                newviewdir = torch.randn_like(viewdirs_[mask_]) #N,3
                newviewdir_norm = newviewdir.norm(dim=-1, keepdim=True)
                viewdirs[mask_] = newviewdir/newviewdir_norm
                
            rgbs = gsplat.spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)
       
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = gsplat.project_gaussians(
            means,
            scales,
            1,
            quats,
            viewmat.squeeze()[:3, :].float(),
            fx.item(),
            fy.item(),
            cx.item(),
            cy.item(),
            H,
            W,
            BLOCK_WIDTH,
        ) 
        rgb, alpha = gsplat.rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  
            rgbs,
            opacities,
            H,
            W,
            BLOCK_WIDTH,
            background = background_color,
            return_alpha = True,
        )  
        alpha = alpha.unsqueeze(-1)

    rgb = torch.clamp(rgb, max=1.0)  

    return rgb, alpha

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def prepare_viewer(cameras, dirname, sh_degree):    #1. cfg_args
    cfg_dict = {}
    cfg_dict['source_path'] = '' # It does not matter
    cfg_dict['sh_degree'] = sh_degree
    cfg_dict['white_background'] = False
    with open(dirname+'/cfg_args', 'w') as f:
        f.write(str(Namespace(**cfg_dict)))
    #2. Camera pose
    cameras_towrite= []
    for i, c2w_opengl in enumerate(cameras['camera_to_worlds'].flip(0)):
        c2w_opengl  = cameras['camera_to_worlds'][i]
        cam = {'id':i, 'img_name':f'img_{i}.png',
               'width': cameras['width'].item(),
                'height': cameras['height'].item(),
                'fx': cameras['fx'].item(),
                'fy': cameras['fy'].item(),
                'FovX': None, 'FovY': None,
                'position': None, 'rotation': None}
        cam['FovX'] = focal2fov(cam['fx'], cam['width'])
        cam['FovY'] = focal2fov(cam['fy'], cam['height'])
        c2w_colmap_4x4 = np.eye(4)
        c2w_colmap_4x4[:3,:4] = c2w_opengl.cpu().numpy()
        c2w_colmap_4x4[:3,1:3]*=-1 #flip y and z
        w2c = np.linalg.inv(c2w_colmap_4x4)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        Rt = np.zeros((4, 4))
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = T 
        Rt[3, 3] = 1.0 

        W2C = np.linalg.inv(Rt) 
        pos = W2C[:3, 3] 
        rot = W2C[:3, :3] 
        serializable_array_2d = [x.tolist() for x in rot]
        cam['position'] = pos.tolist()
        cam['rotation'] = serializable_array_2d
        cameras_towrite.append(cam)
    with open(dirname+'/cameras.json', 'w') as f:
        json.dump(cameras_towrite, f)


def export_ply_forviewer(gs_params, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    count = 0
    map_to_tensors = OrderedDict()

    with torch.no_grad():
        positions = gs_params['means'].cpu().numpy()
        count = positions.shape[0]
        n = count
        map_to_tensors["x"] = positions[:, 0]
        map_to_tensors["y"] = positions[:, 1]
        map_to_tensors["z"] = positions[:, 2]
        map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
        map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)


        if 'features_rest' in gs_params and gs_params['features_rest'].shape[1]!=0:
            shs_0 = gs_params['features_dc'].contiguous().cpu().numpy() #N,3
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]
            # transpose(1, 2) was needed to match the sh order in Inria version
            shs_rest = gs_params['features_rest'].transpose(1, 2).contiguous().cpu().numpy()
            shs_rest = shs_rest.reshape((n, -1))
            for i in range(shs_rest.shape[-1]):
                map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
        else:
            #convert logit(color) to features_dc
            color = torch.sigmoid(gs_params['features_dc'])
            shs_0 = RGB2SH(color).cpu().numpy()
            for i in range(shs_0.shape[1]):
                map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

        map_to_tensors["opacity"] = gs_params['opacities'].data.cpu().numpy()
        scales =  gs_params['scales'].data.cpu().numpy()
        for i in range(3):
            map_to_tensors[f"scale_{i}"] = scales[:, i, None]

        quats = gs_params['quats'].data.cpu().numpy()
        for i in range(4):
            map_to_tensors[f"rot_{i}"] = quats[:, i, None]


    write_ply_v2(str(filename), map_to_tensors)

def write_ply_v2(path, map_to_tensors):
    '''
    from Inria's 3DGS implementation
    Save 3DGS for their viewer
    '''
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    all_keys = list(map_to_tensors.keys())
    f_dc = []
    for key in all_keys:
        if key.startswith('f_dc_'):
            l.append(key)
            f_dc.append(map_to_tensors[key])
    f_dc = np.concatenate(f_dc, axis=1) # N, 3


    f_rest = []
    for key in all_keys:
        if key.startswith('f_rest_'):
            l.append(key)
            f_rest.append(map_to_tensors[key]) # (N, 1)
    if len(f_rest) > 0:
        f_rest = np.concatenate(f_rest, axis=1) # (N,D)
    else:
        f_rest = np.zeros((f_dc.shape[0], 0))


    l.append('opacity')
    opacities = map_to_tensors['opacity']


    scale = []
    for key in all_keys:
        if key.startswith('scale_'):
            l.append(key)
            scale.append(map_to_tensors[key])
    scale = np.concatenate(scale, axis=1) # (N, 3)


    rotation = []
    for key in all_keys:
        if key.startswith('rot_'):
            l.append(key)
            rotation.append(map_to_tensors[key])
    rotation = np.concatenate(rotation, axis=1) # (N, 4)


    dtype_full = [(attribute, 'f4') for attribute in l]
    N = map_to_tensors['x'].shape[0]
    elements = np.empty(N, dtype=dtype_full)
    xyz = np.stack([map_to_tensors['x'], map_to_tensors['y'], map_to_tensors['z']], axis=1)
    normals = np.zeros_like(xyz)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
