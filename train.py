import torch, os
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import sys
from torch.optim import SGD
import wandb, json
import argparse
import cv2
from models.feature_predictor import FeaturePredictor

from collections import OrderedDict
import random
import gin 
from absl import app, flags
from dataset.Loader import build_trainloader, build_testloader
from utils import gpu_utils, gs_utils, loss_utils
from utils.optimizers import build_optimizer, build_scheduler
from utils.metrics import MetricComputer
from utils.log_utils import ProcessSafeLogger
from utils.metrics import psnr
from utils.loss_utils import lpips_loss_fn_optimized, lpips_loss_fn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from fused_ssim import fused_ssim
import torch.nn.functional as F
from scipy.ndimage import gaussian_laplace

flags.DEFINE_string('output_dir', 'output', 'Output directory')
flags.DEFINE_string('eval_subdir', 'eval_final', 'Eval subdirectory')
flags.DEFINE_string('wandb_dir', 'outputs/wandb', 'Weights & Biases output directory')
flags.DEFINE_boolean('disable_wandb', True, 'Run W&B in disabled mode.')
flags.DEFINE_boolean('only_eval', False, 'eval or train')
flags.DEFINE_boolean('compare_with_input', False, 'Compare with input') #for evaluation
flags.DEFINE_boolean('save_viewer', False, 'Save viewer')
flags.DEFINE_multi_string(
  'gin_file', None, 'List of paths to the config files.')
flags.DEFINE_multi_string(
  'gin_param', '', 'Newline separated list of Gin parameter bindings.')

FLAGS = flags.FLAGS

@gin.configurable
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# def _gaussian_blur(img: torch.Tensor, ksize: int = 5, sigma: float = 1.0):
#     """
#     简易可分离高斯模糊 (channels groups)
#     img:  (N, C, H, W) tensor in [0,1]
#     """
#     device, dtype, C = img.device, img.dtype, img.shape[1]
#     kernel = torch.arange(ksize, device=device, dtype=dtype) - ksize // 2
#     kernel = torch.exp(-(kernel ** 2) / (2 * sigma ** 2))
#     kernel = (kernel / kernel.sum()).view(1, 1, -1)         # (1,1,K)
#     kernel2d = kernel.transpose(1, 2) @ kernel              # outer product
#     kernel2d = kernel2d.expand(C, 1, ksize, ksize).contiguous()
#     return F.conv2d(img, kernel2d, padding=ksize // 2, groups=C)

# # ---------- 拉普拉斯金字塔 ---------- #
# def _laplacian_pyramid(img: torch.Tensor, num_levels: int = 3):
#     """ 返回 [L0, L1, ..., G_lowest]，L0 = 最高分辨率拉普拉斯层 """
#     G = img
#     pyr = []
#     for _ in range(num_levels):
#         G_blur = _gaussian_blur(G, 5, 1.0)
#         G_down = F.interpolate(G_blur, scale_factor=0.5, mode='bilinear',
#                                align_corners=False, recompute_scale_factor=True)
#         G_up   = F.interpolate(G_down, size=G.shape[-2:], mode='bilinear',
#                                align_corners=False)
#         L      = G - G_up
#         pyr.append(L)
#         G = G_down
#     pyr.append(G)           # 最底层残差
#     return pyr              # list high→low

# ---------- 高频 loss（多尺度 L1） ---------- #
# def high_freq_loss(render: torch.Tensor,
#                    gt: torch.Tensor,
#                    weights = None,
#                    num_levels: int = 3) -> torch.Tensor:
#     """
#     render, gt: (N,C,H,W) / [0,1]
#     weights: list[float] 与层数对应；默认每下一层权重减半
#     """
#     if weights is None:
#         weights = [1.0 / (2 ** i) for i in range(num_levels)] + [0.0]  # 最底层 G 不计入
#     assert len(weights) == num_levels + 1, "weights 长度须等于 num_levels + 1"

#     pyr_r = _laplacian_pyramid(render, num_levels)
#     pyr_g = _laplacian_pyramid(gt,     num_levels)
#     loss  = sum(w * (r - g).abs().mean()
#                 for w, r, g in zip(weights, pyr_r, pyr_g))
#     return loss


# def get_gaussian_laplacian_kernel(device, kernel_size=5, sigma=1.0):
#     """返回一个 2D 高斯拉普拉斯卷积核 (LoG)"""
#     ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.).to(device)
#     xx, yy = torch.meshgrid(ax, ax, indexing='ij')
#     norm = (xx ** 2 + yy ** 2) / (2 * sigma ** 2)
#     log = (norm - 1) * torch.exp(-norm)
#     log = log - log.mean()  # 消除 DC 偏置
#     kernel = log / log.abs().sum()  # 归一化
#     return kernel.unsqueeze(0).unsqueeze(0)  # 形状 (1, 1, H, W)


# def high_freq_loss_gpu(render: torch.Tensor, gt: torch.Tensor, sigma: float = 1.0,
#                        loss_type: str = 'l1', mask_thresh: float = 0.05) -> torch.Tensor:
#     """
#     快速 GPU 版 LoG Loss，避免背景均值稀释。
#     """
#     N, C, H, W = render.shape
#     device = render.device

#     # 灰度转换 (N, 1, H, W)
#     coeffs = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
#     render_gray = (render * coeffs).sum(dim=1, keepdim=True)
#     gt_gray = (gt * coeffs).sum(dim=1, keepdim=True)

#     # 卷积核准备
#     kernel = get_gaussian_laplacian_kernel(device, kernel_size=5, sigma=sigma)

#     # 使用 depthwise 卷积模拟 LoG
#     padding = kernel.shape[-1] // 2
#     render_log = F.conv2d(render_gray, kernel, padding=padding, groups=1)
#     gt_log = F.conv2d(gt_gray, kernel, padding=padding, groups=1)

#     # 归一化至 [0, 1]
#     def norm(x):
#         min_val = x.amin(dim=(2, 3), keepdim=True)
#         max_val = x.amax(dim=(2, 3), keepdim=True)
#         return (x - min_val) / (max_val - min_val + 1e-6)

#     render_log = norm(render_log)
#     gt_log = norm(gt_log)

#     # 仅关注有效高频区域
#     mask = (gt_log.abs() > mask_thresh).float()

#     diff = (render_log - gt_log) * mask
#     if loss_type == 'l1':
#         loss = diff.abs().sum() / (mask.sum() + 1e-6)
#     elif loss_type == 'l2':
#         loss = (diff ** 2).sum() / (mask.sum() + 1e-6)
#     else:
#         raise ValueError("loss_type must be 'l1' or 'l2'")

#     return loss


def get_gaussian_laplacian_kernel(device, kernel_size=5, sigma=1.0):
    """生成 2D LoG 卷积核，用于 depthwise conv2d"""
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    norm = (xx ** 2 + yy ** 2) / (2 * sigma ** 2)
    log = (norm - 1) * torch.exp(-norm)
    log = log - log.mean()  # 消 DC 偏置
    kernel = log / log.abs().sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def high_freq_weighted_rgb_loss_from_gt_gpu(input_img: torch.Tensor, gt_img: torch.Tensor, sigma: float = 1.0,
                                            loss_type: str = 'l1') -> torch.Tensor:
    """
    纯 GPU：用 GT 高频做权重，加权 RGB Loss。
    
    Args:
        input_img (torch.Tensor): (N, 3, H, W)，[0, 1]
        gt_img (torch.Tensor): (N, 3, H, W)，[0, 1]
        sigma (float): LoG 滤波器 sigma
        loss_type (str): 'l1' 或 'l2'
    Returns:
        torch.Tensor: 单个标量 Loss
    """
    device = gt_img.device
    N, C, H, W = gt_img.shape

    # 灰度转换
    coeffs = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)
    gt_gray = (gt_img * coeffs).sum(dim=1, keepdim=True)  # (N, 1, H, W)

    # LoG 滤波器
    kernel = get_gaussian_laplacian_kernel(device, kernel_size=5, sigma=sigma)
    padding = kernel.shape[-1] // 2

    # LoG 滤波，GPU 批量
    weight = F.conv2d(gt_gray, kernel, padding=padding, groups=1)

    # 归一化为 [0, 1]
    min_val = weight.amin(dim=(2, 3), keepdim=True)
    max_val = weight.amax(dim=(2, 3), keepdim=True)
    weight = (weight - min_val) / (max_val - min_val + 1e-6)

    # 广播到 RGB
    weight_rgb = weight.expand_as(input_img)

    # 加权 Loss
    diff = (input_img - gt_img) * weight_rgb
    if loss_type == 'l1':
        loss = diff.abs().sum() / (weight_rgb.sum() + 1e-6)
    elif loss_type == 'l2':
        loss = (diff ** 2).sum() / (weight_rgb.sum() + 1e-6)
    else:
        raise ValueError("loss_type must be 'l1' or 'l2'")

    return loss


def make_grid(imgs, nrow=3, ncols=3):
    img_h, img_w = imgs[0].shape[:2]
    if imgs[0].ndim == 3:
        grid = np.zeros((img_h*nrow, img_w*ncols, 3), dtype=np.uint8)
    elif imgs[0].ndim == 2:
        grid = np.zeros((img_h*nrow, img_w*ncols), dtype=np.uint8)
    for i in range(nrow):
        for j in range(ncols):
            if i*ncols+j >= len(imgs):
                break
            grid[i*img_h:(i+1)*img_h, j*img_w:(j+1)*img_w] = imgs[i*ncols+j]
    return grid

@gin.configurable
def evaluation(model, test_loader, output_dir, output_gt, compare_with_pseudo, 
              compare_with_input=False,
              save_as_single=False,
              save_viewer=False,
              evaluate_input=False):
    model.eval()
    metric_computer = MetricComputer()
    if compare_with_input:
      metric_computer_input = MetricComputer()
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
      cnt = 0
      num_images, num_scenes = 0, 0
      pseudo_loss = {}
      for test_batch in tqdm(test_loader): #A scene at a time
        test_batch_gs = gpu_utils.move_to_device([data['gs_params'] for data in test_batch],model.device)
        test_batch_cameras = gpu_utils.move_to_device([data['cameras'] for data in test_batch],model.device)
        test_batch_images = gpu_utils.move_to_device([data['images'] for data in test_batch],model.device)
        test_batch_idx = [data['scene_idx'] for data in test_batch]
        test_batch_name = [data['scene_name'] for data in test_batch]
        test_batch_imgname = [data['images_name'] for data in test_batch]
        forward_kwargs = {'batch_normalized_gs': test_batch_gs, 'batch_scene_idx': test_batch_idx}

        out_test_batch_gs = model(**forward_kwargs)
        for iii, (out_gs, in_gs, cameras, gt_imgs, scene_idx) in enumerate(zip(out_test_batch_gs, test_batch_gs, test_batch_cameras, test_batch_images, test_batch_idx)):
          if evaluate_input:
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(in_gs, cameras)
          else:
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, cameras) # List of torch.tensor([H,W,3])
          
          # 处理mask并转换为uint8，一次性处理避免多次堆叠
          has_alpha = gt_imgs[0].shape[-1] == 4 if len(gt_imgs) > 0 else False
          
          # 用于保存可视化的图像列表
          pred_imgs_vis = []
          gt_imgs_vis = []
          
          # 用于计算metric的临时列表（分批处理）
          pred_imgs_for_metric = []
          gt_imgs_for_metric = []
          
          background_rgb = cameras['background_color'].to(
              device=model.device,
              dtype=test_batch_images[iii][0].dtype,
          ).view(1, 1, 3)

          # 逐图像处理以节省显存
          for idx, (pred_img, gt_img) in enumerate(zip(pred_imgs, gt_imgs)):
            if has_alpha:
              mask = gt_img[..., 3:4]  # Keep dimension
              pred_img_masked = pred_img * mask + background_rgb * (1.0 - mask)
              gt_img_rgb = gt_img[..., :3]
              # 转换为uint8用于保存
              pred_img_uint8 = (pred_img_masked * 255).to(torch.uint8)
              gt_img_uint8 = (gt_img_rgb * 255).to(torch.uint8)
            else:
              mask = None
              pred_img_uint8 = (pred_img * 255).to(torch.uint8)
              gt_img_uint8 = (gt_img * 255).to(torch.uint8)
            
            pred_imgs_vis.append(pred_img_uint8)
            gt_imgs_vis.append(gt_img_uint8)
            
            # 保存用于metric计算（保持在GPU上）
            pred_imgs_for_metric.append(pred_img_uint8)
            gt_imgs_for_metric.append(gt_img_uint8)
            
            # 每处理8张图片就计算一次metric，避免累积太多
            if len(pred_imgs_for_metric) >= 8 or idx == len(pred_imgs) - 1:
              pred_batch = torch.stack(pred_imgs_for_metric, dim=0)
              gt_batch = torch.stack(gt_imgs_for_metric, dim=0)
              metric_computer.update(pred_batch, gt_batch, name=f'{scene_idx}')
              num_images += len(pred_imgs_for_metric)
              # 立即清理
              del pred_batch, gt_batch
              pred_imgs_for_metric.clear()
              gt_imgs_for_metric.clear()
              torch.cuda.empty_cache()
          
          # 清理原始pred_imgs和gt_imgs
          del pred_imgs, gt_imgs
          torch.cuda.empty_cache()

          # 保存可视化图像（已经是uint8格式）
          imgs = [im.cpu().numpy() for im in pred_imgs_vis]
          grid = make_grid(imgs)
          grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
          cv2.imwrite(os.path.join(output_dir, f'scene{scene_idx}_pred.png'), grid)
          del imgs, grid
          
          if output_gt:
            gt_imgs_ = [im.cpu().numpy() for im in gt_imgs_vis]
            grid = make_grid(gt_imgs_)
            grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f'scene{scene_idx}_gt.png'), grid)
            del gt_imgs_, grid

          if compare_with_input:
            input_imgs_list, _ = gs_utils.rasterize_gaussians_to_multiimgs(in_gs, cameras)
            # 逐图像处理input
            input_imgs_vis = []
            input_imgs_for_metric = []
            
            for idx, input_img in enumerate(input_imgs_list):
              if has_alpha:
                mask = test_batch_images[iii][idx][..., 3:4]
                input_img = input_img * mask + background_rgb * (1.0 - mask)
              input_img_uint8 = (input_img * 255).to(torch.uint8)
              input_imgs_vis.append(input_img_uint8)
              input_imgs_for_metric.append(input_img_uint8)
              
              # 分批计算metric
              if len(input_imgs_for_metric) >= 8 or idx == len(input_imgs_list) - 1:
                input_batch = torch.stack(input_imgs_for_metric, dim=0)
                gt_batch = torch.stack(gt_imgs_vis[idx-len(input_imgs_for_metric)+1:idx+1], dim=0)
                metric_computer_input.update(input_batch, gt_batch, name=f'{scene_idx}')
                del input_batch, gt_batch
                input_imgs_for_metric.clear()
                torch.cuda.empty_cache()
            
            del input_imgs_list
            torch.cuda.empty_cache()
            
            output_dir_thisscene = os.path.join(output_dir, f'compare/{test_batch_name[iii]}')
            os.makedirs(output_dir_thisscene, exist_ok=True)

            # 保存对比图（逐个处理避免大量内存占用）
            for ii, (gt_img, input_img, pred_img) in enumerate(zip(gt_imgs_vis, input_imgs_vis, pred_imgs_vis)):
              gt_img_np = gt_img.cpu().numpy()
              input_img_np = input_img.cpu().numpy()
              pred_img_np = pred_img.cpu().numpy()
              cmp_img = np.concatenate([gt_img_np, input_img_np, pred_img_np], axis=1)
              cv2.imwrite(os.path.join(output_dir_thisscene, f'{ii:02d}.png'), cmp_img[:,:,::-1])
              del gt_img_np, input_img_np, pred_img_np, cmp_img
            
            del input_imgs_vis

          if save_as_single:
            output_dir_thisscene_single = os.path.join(output_dir, f'pred/{test_batch_name[iii]}')
            os.makedirs(output_dir_thisscene_single, exist_ok=True)
            for ii, pred_img in enumerate(pred_imgs_vis):
              pred_img_np = pred_img.cpu().numpy()
              cv2.imwrite(os.path.join(output_dir_thisscene_single, test_batch_imgname[iii][ii]), pred_img_np[:,:,::-1])
              del pred_img_np
              
          if save_viewer:
            viewerdir = os.path.join(output_dir, f'viewer/{test_batch_name[iii]}')
            os.makedirs(viewerdir, exist_ok=True)
            gs_utils.prepare_viewer(cameras, viewerdir, model.module.sh_degree)
            # Save input 3dgs
            gs_utils.export_ply_forviewer(gs_params=in_gs, filename=os.path.join(viewerdir, 'point_cloud/iteration_0/point_cloud.ply'))
            gs_utils.export_ply_forviewer(gs_params=out_gs, filename=os.path.join(viewerdir, 'point_cloud/iteration_1/point_cloud.ply'))
          
          cnt += 1
          
          # 清理本次迭代的所有数据
          del pred_imgs_vis, gt_imgs_vis, out_gs, in_gs, cameras
          torch.cuda.empty_cache()
        
        # 批次处理完后的清理
        del test_batch_gs, test_batch_cameras, test_batch_images, out_test_batch_gs  
        torch.cuda.empty_cache()
      metrics = metric_computer.sum() # We need to sum the metrics
      metric_computer.write_to_file(os.path.join(output_dir, f'metrics.rank{dist.get_rank()}.json'))
      if compare_with_input:
        metric_computer_input.write_to_file(os.path.join(output_dir, f'metrics_input.rank{dist.get_rank()}.json'))

    model.train()

    num_images = torch.tensor([num_images]).to(model.device)
    num_scenes = torch.tensor([num_scenes]).to(model.device)
    torch.distributed.reduce(num_images, dst=0) #Sum
    torch.distributed.reduce(num_scenes, dst=0) #Sum
    for key in metrics:
      torch.distributed.reduce(metrics[key], dst=0) #Sum
      if dist.get_rank() == 0:
        metrics[key] = (metrics[key]/num_images).item()
    if compare_with_pseudo:
      for key in pseudo_loss:
        torch.distributed.reduce(pseudo_loss[key], dst=0)
        if dist.get_rank() == 0:
          metrics[f'pseudo_loss_{key}'] = (pseudo_loss[key]/num_scenes).item()
        
    if compare_with_input:
      metrics_input = metric_computer_input.sum()
      for key in metrics_input:
        torch.distributed.reduce(metrics_input[key], dst=0)
        if dist.get_rank() == 0:
          metrics_input[key] = (metrics_input[key]/num_images).item()
    else:
      metrics_input = {}
    return metrics, metrics_input

       
@gin.configurable
def training(
    model, optimizer_, scheduler_, train_loader, 
    output_dir,
    total_steps: gin.REQUIRED,
    pretrain_steps: gin.REQUIRED,
    eval_interval: gin.REQUIRED,
    eval_step: gin.REQUIRED,
    log_interval: gin.REQUIRED,
    save_interval: gin.REQUIRED,
    log_image_interval: gin.REQUIRED,
    grad_clip_norm: gin.REQUIRED,
    dropout_rate = gin.REQUIRED,
    means_rate = gin.REQUIRED,
    scales_rate = gin.REQUIRED,
    opacities_rate = gin.REQUIRED,
    quats_rate = gin.REQUIRED,
    features_dc_rate = gin.REQUIRED,
    features_rest_rate = gin.REQUIRED,
    offset_rate = gin.REQUIRED,
    image_l1_loss_weight=1.0,
    lpips_loss_weight=0.0,
    laplace_loss_weight=0.0,
    log_loss_weight=0.0,
    means_loss_weight=0.0,
    ssim_loss_weight=0.0,
    resume_from_step=0,
    enable_amp=False,
    empty_cache_fre=-1
):   
    if dist.get_rank() == 0:
      logger = ProcessSafeLogger(os.path.join(output_dir, 'train.log')).get_logger()
    if enable_amp:
      scaler = torch.cuda.amp.GradScaler()
      torch.autograd.set_detect_anomaly(False)
    else:
      torch.autograd.set_detect_anomaly(False)

    if lpips_loss_weight > 0:
      lpips_loss_func = loss_utils.PerceptualLoss().to(model.device)
      # lpips_loss_func = loss_utils.lpips_loss_fn()
      # lpips_loss_func = lpips_loss_fn(
      #     enable_manual_grad=True,
      #     chunk_size=1,  # 根据显存大小调整
      # )
      
    # 创建一个用于记录低PSNR场景的文件
    low_psnr_file = os.path.join(output_dir, 'low_psnr_scenes.txt')
    os.makedirs(os.path.dirname(low_psnr_file), exist_ok=True)
    
    train_iterator = iter(train_loader)
    accumulate_step = gin.query_parameter('build_trainloader.accumulate_step')
    if dist.get_rank() == 0:
      logger.info(f'Accumulate step: {accumulate_step}')
    for step in tqdm(range(resume_from_step*accumulate_step, total_steps*accumulate_step), disable=dist.get_rank()!=0):
      step_consider_accum = step//accumulate_step
      try:
        batch = next(train_iterator)
      except StopIteration:
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

      while batch[0]['gs_params']['means'].shape[0] < 100: # Skip the batch if the number of means is too small
        try:
          batch = next(train_iterator)
        except StopIteration:
          train_iterator = iter(train_loader)
          batch = next(train_iterator)
      
      # ## 数据增强
      # for j, data in enumerate(batch):
      #   gs = data['gs_params']
      #   # 1.丢弃一定比例的Gaussian点
      #   if dropout_rate > 0:
      #     # 以dropout_rate比例创建一个gs的mask
      #     num_gaussians = gs['means'].shape[0]
      #     num_keep = int(num_gaussians * (1 - dropout_rate))
      #     indices = torch.randperm(num_gaussians)[:num_keep]
      #     mask = torch.zeros(num_gaussians, dtype=torch.bool)
      #     mask[indices] = True

      #     # 应用mask到所有gs参数
      #     for key in gs:
      #       if isinstance(gs[key], torch.Tensor) and gs[key].shape[0] == num_gaussians:
      #         gs[key] = gs[key][mask]
        
      #   # 2.对means进行随机偏移
      #   if means_rate > 0:
      #     # 统计每一维的范围
      #     means = gs['means']
      #     means_range = []
      #     for dim in range(3):
      #       dim_min = means[:, dim].min()
      #       dim_max = means[:, dim].max()
      #       dim_range = dim_max - dim_min
      #       means_range.append(dim_range)
          
      #     # 随机选择means_rate比例的高斯点进行偏移
      #     num_gaussians = means.shape[0]
      #     num_select = int(num_gaussians * means_rate)
      #     if num_select > 0:
      #       select_indices = torch.randperm(num_gaussians)[:num_select]
            
      #       # 对每一维添加随机偏移，偏移范围为该维度范围的means_rate比例
      #       noise = torch.randn(num_select, 3)
      #       for i in range(3):
      #         noise[:, i] = (noise[:, i] * 2 - 1) * means_range[i] * offset_rate
            
      #       # 只对选中的高斯点添加偏移
      #       gs['means'][select_indices] = means[select_indices] + noise.to(means.device)
      #     else:
      #       # 如果没有选中任何点，保持原样
      #       gs['means'] = means
        
      #   # 3.对scales进行随机偏移
      #   if scales_rate > 0:
      #     # 统计每一维的范围
      #     scales = gs['scales']
      #     scales_range = []
      #     for dim in range(3):
      #       dim_min = scales[:, dim].min()
      #       dim_max = scales[:, dim].max()
      #       dim_range = dim_max - dim_min
      #       scales_range.append(dim_range)

      #     # 随机选择scales_rate比例的高斯点进行偏移
      #     num_gaussians = scales.shape[0]
      #     num_select = int(num_gaussians * scales_rate)
      #     if num_select > 0:
      #       select_indices = torch.randperm(num_gaussians)[:num_select]

      #       # 对每一维添加随机偏移，偏移范围为该维度范围的scales_rate比例
      #       noise = torch.randn(num_select, 3)
      #       for i in range(3):
      #         noise[:, i] = (noise[:, i] * 2 - 1) * scales_range[i] * offset_rate

      #       # 只对选中的高斯点添加偏移
      #       gs['scales'][select_indices] = scales[select_indices] + noise.to(scales.device)
      #     else:
      #       # 如果没有选中任何点，保持原样
      #       gs['scales'] = scales
            
      #   # 4.对quats进行随机偏移
      #   if quats_rate > 0:
      #     # 统计每一维的范围
      #     quats = gs['quats']
      #     quats_range = []
      #     for dim in range(4):
      #       dim_min = quats[:, dim].min()
      #       dim_max = quats[:, dim].max()
      #       dim_range = dim_max - dim_min
      #       quats_range.append(dim_range)

      #     # 随机选择quats_rate比例的高斯点进行偏移
      #     num_gaussians = quats.shape[0]
      #     num_select = int(num_gaussians * quats_rate)
      #     if num_select > 0:
      #       select_indices = torch.randperm(num_gaussians)[:num_select]

      #       # 对每一维添加随机偏移，偏移范围为该维度范围的quats_rate比例
      #       noise = torch.randn(num_select, 4)
      #       for i in range(4):
      #         noise[:, i] = (noise[:, i] * 2 - 1) * quats_range[i] * offset_rate

      #       # 只对选中的高斯点添加偏移
      #       gs['quats'][select_indices] = quats[select_indices] + noise.to(quats.device)
      #     else:
      #       # 如果没有选中任何点，保持原样
      #       gs['quats'] = quats

      #   # 5.对opacities进行随机偏移
      #   if opacities_rate > 0:
      #     # 统计每一维的范围
      #     opacities = gs['opacities']
      #     opacities_range = []
      #     for dim in range(1):
      #       dim_min = opacities[:, dim].min()
      #       dim_max = opacities[:, dim].max()
      #       dim_range = dim_max - dim_min
      #       opacities_range.append(dim_range)

      #     # 随机选择opacities_rate比例的高斯点进行偏移
      #     num_gaussians = opacities.shape[0]
      #     num_select = int(num_gaussians * opacities_rate)
      #     if num_select > 0:
      #       select_indices = torch.randperm(num_gaussians)[:num_select]

      #       # 对每一维添加随机偏移，偏移范围为该维度范围的opacities_rate比例
      #       noise = torch.randn(num_select, 1)
      #       for i in range(1):
      #         noise[:, i] = (noise[:, i] * 2 - 1) * opacities_range[i] * offset_rate

      #       # 只对选中的高斯点添加偏移
      #       gs['opacities'][select_indices] = opacities[select_indices] + noise.to(opacities.device)
      #     else:
      #       # 如果没有选中任何点，保持原样
      #       gs['opacities'] = opacities
        
      #   # 6.对features_dc进行随机偏移
      #   if features_dc_rate > 0:
      #     # 统计每一维的范围
      #     features_dc = gs['features_dc']
      #     features_dc_range = []
      #     for dim in range(3):
      #       dim_min = features_dc[:, dim].min()
      #       dim_max = features_dc[:, dim].max()
      #       dim_range = dim_max - dim_min
      #       features_dc_range.append(dim_range)

      #     # 随机选择features_dc_rate比例的高斯点进行偏移
      #     num_gaussians = features_dc.shape[0]
      #     num_select = int(num_gaussians * features_dc_rate)
      #     if num_select > 0:
      #       select_indices = torch.randperm(num_gaussians)[:num_select]

      #       # 对每一维添加随机偏移，偏移范围为该维度范围的features_dc_rate比例
      #       noise = torch.randn(num_select, 3)
      #       for i in range(3):
      #         noise[:, i] = (noise[:, i] * 2 - 1) * features_dc_range[i] * offset_rate

      #       # 只对选中的高斯点添加偏移
      #       gs['features_dc'][select_indices] = features_dc[select_indices] + noise.to(features_dc.device)
      #     else:
      #       # 如果没有选中任何点，保持原样
      #       gs['features_dc'] = features_dc
        
      #   batch[j]['gs_params'] = gs


      batch_gs = gpu_utils.move_to_device([data['gs_params'] for data in batch],model.device)
      batch_cameras = gpu_utils.move_to_device([data['cameras'] for data in batch],model.device)
      batch_images = gpu_utils.move_to_device([data['images'] for data in batch],model.device)
      batch_scene_name = [data['scene_name'] for data in batch]
      batch_scene_idx = [data['scene_idx'] for data in batch]
      forward_kwargs = {'batch_normalized_gs': batch_gs, 'batch_scene_idx': batch_scene_idx}

      with torch.cuda.amp.autocast(enabled=enable_amp):
          out_batch_gs = model(**forward_kwargs)

      loss_dict = {}
      metric_dict = {}
      if step_consider_accum < pretrain_steps:
        loss = 0
        for ii, (out_gs, in_gs) in enumerate(zip(out_batch_gs, batch_gs)):
          with torch.no_grad():
            pseudo_target = gs_utils.create_pseudo_target(
              sh_degree=model.module.sh_degree, 
              N=in_gs['means'].shape[0],
              input_gs=in_gs,)

          for key in pseudo_target:
            target = pseudo_target[key].to(model.device)
            pred = out_gs[key]
            value = (pred - target).abs().mean()
            if key == 'features_rest':
              if model.module.sh_degree>0:
                loss += value
            else:
              loss += value
            metric_dict['pretrain/'+key] = value
        loss = loss/len(out_batch_gs)
        loss_dict['pretrain_loss'] = loss/len(out_batch_gs)
        optimizer, scheduler = optimizer_['pretrain'], scheduler_['pretrain']
      else:
          loss_dict['image_l1'], metric_dict['train_psnr'] = 0, 0,
          if lpips_loss_weight > 0:
            loss_dict['lpips'] = 0
          if ssim_loss_weight > 0:
            loss_dict['ssim_loss'] = 0
            metric_dict['ssim'] = 0
          if means_loss_weight > 0:
            loss_dict['means_loss'] = 0
          if laplace_loss_weight > 0:
            loss_dict['laplace_loss'] = 0
          if log_loss_weight > 0:
            loss_dict['log_loss'] = 0
          num_images = 0
          for out_gs, cameras, images, in_gs in zip(out_batch_gs, batch_cameras, batch_images, batch_gs):
            # # 复制一份in_gs并与自身拼接
            # print('in_gs means num:', in_gs['means'].shape[0])
            # print('out_gs means num:', out_gs['means'].shape[0])
            # # 复制一份in_gs并与自身拼接，使数量与out_gs保持一致
            # in_gs_copy = {'means': torch.cat([in_gs['means'], in_gs['means']], dim=0)}
            
            # # 计算means loss
            # means_loss = torch.nn.functional.mse_loss(out_gs['means'], in_gs_copy['means'])
            # loss_dict['means_loss'] += means_loss 
            
            if means_loss_weight > 0:
              num_input = in_gs['means'].shape[0]
              
              # 分别计算前半部分和后半部分的loss
              means_loss_1 = torch.nn.functional.mse_loss(out_gs['means'][:num_input], in_gs['means'])
              means_loss_2 = torch.nn.functional.mse_loss(out_gs['means'][num_input:], in_gs['means'])
              
              # 总的means_loss是两部分的平均
              means_loss = (means_loss_1 + means_loss_2) / 2.0
              loss_dict['means_loss'] += means_loss
            
            pred_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(out_gs, cameras) #a List
            in_gs_imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(in_gs, cameras)
            for pred_img, gt_img, in_img in zip(pred_imgs, images, in_gs_imgs):
                # Threshold gt_img values less than 0.004 to 0
                mask = (gt_img< 0.004).all(dim=-1, keepdim=True)
                gt_img = gt_img * (~mask)  # Set those pixels to 0
                
                gt_mask = (gt_img.sum(dim=-1, keepdim=True) > 0).float()
                pred_mask = (pred_img.sum(dim=-1, keepdim=True) > 0).float()
                combined_mask = torch.maximum(gt_mask, pred_mask)
                # Only compute L1 loss on the masked regions
                if combined_mask.sum() > 0:
                  loss_dict['image_l1'] += (pred_img * combined_mask - gt_img * combined_mask).abs().sum() / combined_mask.sum()
                else:
                  # If no valid pixels, skip this image
                  continue
                
                # loss_dict['image_l1'] += (pred_img - gt_img).abs().mean() #/len(pred_imgs)
                torch.cuda.empty_cache()
                # Save pred_img and gt_img for debugging
                if step_consider_accum % 100 == 0: 
                  pred_img_save = (pred_img * 255).detach().cpu().numpy().astype(np.uint8)
                  gt_img_save = (gt_img * 255).detach().cpu().numpy().astype(np.uint8)
                  debug_dir = os.path.join(output_dir, 'debug_images')
                  os.makedirs(debug_dir, exist_ok=True)
                  cv2.imwrite(os.path.join(debug_dir, f'step_{step_consider_accum}_pred_{num_images}.png'), 
                        cv2.cvtColor(pred_img_save, cv2.COLOR_RGB2BGR))
                  cv2.imwrite(os.path.join(debug_dir, f'step_{step_consider_accum}_gt_{num_images}.png'), 
                        cv2.cvtColor(gt_img_save, cv2.COLOR_RGB2BGR))
                  
                  in_img_save = (in_img * 255).detach().cpu().numpy().astype(np.uint8)
                  cv2.imwrite(os.path.join(debug_dir, f'step_{step_consider_accum}_in_{num_images}.png'), 
                      cv2.cvtColor(in_img_save, cv2.COLOR_RGB2BGR))

                # ...existing code...
                if lpips_loss_weight > 0:
                    # Downsample to 1024x1024 for LPIPS computation
                    pred_img_ds = F.interpolate(pred_img.unsqueeze(0).permute(0, 3, 1, 2), size=(1024, 1024), mode='bilinear', align_corners=False)
                    gt_img_ds = F.interpolate(gt_img.unsqueeze(0).permute(0, 3, 1, 2), size=(1024, 1024), mode='bilinear', align_corners=False)

                    # 智能裁剪函数
                    def get_smart_bbox(pred_img, gt_img, threshold=0.02, min_size=64):
                        """
                        智能裁剪，确保裁剪后的图像足够大
                        Args:
                            pred_img, gt_img: (1, 3, H, W) 张量，值在[0,1]
                            threshold: 阈值
                            min_size: 最小裁剪尺寸
                        """
                        H, W = pred_img.shape[2], pred_img.shape[3]
                        
                        # 转换为[0,1]范围进行处理
                        pred_mask = (pred_img.sum(dim=1, keepdim=True) > threshold).float()  # (1,1,H,W)
                        gt_mask = (gt_img.sum(dim=1, keepdim=True) > threshold).float()
                        
                        # 合并mask
                        combined_mask = torch.maximum(pred_mask, gt_mask)[0, 0]  # (H,W)
                        
                        # 找到非零区域
                        coords = torch.nonzero(combined_mask, as_tuple=False)
                        if coords.shape[0] == 0:
                            # 如果全是黑色，返回中心区域
                            center_h, center_w = H // 2, W // 2
                            half_size = min_size // 2
                            y0 = max(0, center_h - half_size)
                            y1 = min(H, center_h + half_size)
                            x0 = max(0, center_w - half_size)
                            x1 = min(W, center_w + half_size)
                        else:
                            y0, x0 = coords.min(dim=0)[0]
                            y1, x1 = coords.max(dim=0)[0] + 1
                            
                            # 确保裁剪区域足够大
                            crop_h, crop_w = y1 - y0, x1 - x0
                            if crop_h < min_size or crop_w < min_size:
                                # 扩大裁剪区域
                                center_y, center_x = (y0 + y1) // 2, (x0 + x1) // 2
                                half_h, half_w = max(min_size // 2, crop_h // 2), max(min_size // 2, crop_w // 2)
                                y0 = max(0, center_y - half_h)
                                y1 = min(H, center_y + half_h)
                                x0 = max(0, center_x - half_w)
                                x1 = min(W, center_x + half_w)
                        
                        return int(y0), int(y1), int(x0), int(x1)

                    # 获取智能裁剪边界
                    y0, y1, x0, x1 = get_smart_bbox(pred_img_ds, gt_img_ds, threshold=0.02, min_size=64)
                    
                    # 确保裁剪区域有效
                    if (y1 - y0) >= 32 and (x1 - x0) >= 32:  # VGG需要的最小尺寸
                        pred_img_crop = pred_img_ds[:, :, y0:y1, x0:x1]
                        gt_img_crop = gt_img_ds[:, :, y0:y1, x0:x1]
                        
                        # 再次检查尺寸
                        if pred_img_crop.shape[2] >= 32 and pred_img_crop.shape[3] >= 32:
                            try:
                                loss_dict['lpips'] += lpips_loss_func(pred_img_crop, gt_img_crop.detach())
                            except RuntimeError as e:
                                if "Output size is too small" in str(e):
                                    # 如果还是太小，使用原图的中心区域
                                    center_crop_size = 256
                                    H, W = pred_img_ds.shape[2], pred_img_ds.shape[3]
                                    start_h = (H - center_crop_size) // 2
                                    start_w = (W - center_crop_size) // 2
                                    pred_center = pred_img_ds[:, :, start_h:start_h+center_crop_size, start_w:start_w+center_crop_size]
                                    gt_center = gt_img_ds[:, :, start_h:start_h+center_crop_size, start_w:start_w+center_crop_size]
                                    loss_dict['lpips'] += lpips_loss_func(pred_center, gt_center.detach())
                                else:
                                    raise e
                    else:
                        # 裁剪区域太小，使用中心裁剪
                        center_crop_size = 256
                        H, W = pred_img_ds.shape[2], pred_img_ds.shape[3]
                        start_h = (H - center_crop_size) // 2
                        start_w = (W - center_crop_size) // 2
                        pred_center = pred_img_ds[:, :, start_h:start_h+center_crop_size, start_w:start_w+center_crop_size]
                        gt_center = gt_img_ds[:, :, start_h:start_h+center_crop_size, start_w:start_w+center_crop_size]
                        loss_dict['lpips'] += lpips_loss_func(pred_center, gt_center.detach())

                  # loss_dict['lpips'] += lpips_loss_func(pred_img.unsqueeze(0), gt_img.unsqueeze(0), use_checkpoint=True).mean()
                  # loss_dict['lpips'] += lpips_loss_func(pred_img.unsqueeze(0), gt_img.unsqueeze(0),use_manual_grad=True)
                if ssim_loss_weight > 0:
                  loss_dict['ssim_loss'] += 1.0 - fused_ssim(pred_img.unsqueeze(0).permute(0, 3, 1, 2), gt_img.unsqueeze(0).permute(0, 3, 1, 2), padding="valid")

                if laplace_loss_weight > 0:
                  # 计算高频损失
                  laplace_loss = high_freq_loss(pred_img.unsqueeze(0).permute(0, 3, 1, 2), gt_img.unsqueeze(0).permute(0, 3, 1, 2), num_levels=3)
                  loss_dict['laplace_loss'] += laplace_loss
                
                if log_loss_weight > 0:
                  # 计算LoG损失
                  log_loss = high_freq_weighted_rgb_loss_from_gt_gpu(pred_img.unsqueeze(0).permute(0, 3, 1, 2), gt_img.unsqueeze(0).permute(0, 3, 1, 2), sigma=1.0, loss_type='l1')
                  loss_dict['log_loss'] += log_loss
                
                # Compute PSNR
                img_psnr = (psnr(pred_img.unsqueeze(0), gt_img.unsqueeze(0)).mean())
                if img_psnr < 25 and num_images == 0:
                  with open(low_psnr_file, "a") as f:
                    f.write(f'{batch_scene_name[0]}_{img_psnr}_{step_consider_accum}\n')
                  scene_name = batch_scene_name[0].split('/')[-3]
                  pred_img_save = (pred_img * 255).detach().cpu().numpy().astype(np.uint8)
                  gt_img_save = (gt_img * 255).detach().cpu().numpy().astype(np.uint8)
                  low_psnr_dir = os.path.join(output_dir, 'low_psnr_images')
                  os.makedirs(low_psnr_dir, exist_ok=True)
                  cv2.imwrite(os.path.join(low_psnr_dir, f'{scene_name}_{img_psnr}_pred_{num_images}.png'), 
                        cv2.cvtColor(pred_img_save, cv2.COLOR_RGB2BGR))
                  cv2.imwrite(os.path.join(low_psnr_dir, f'{scene_name}_{img_psnr}_gt_{num_images}.png'), 
                        cv2.cvtColor(gt_img_save, cv2.COLOR_RGB2BGR))
                  
                  in_img_save = (in_img * 255).detach().cpu().numpy().astype(np.uint8)
                  cv2.imwrite(os.path.join(low_psnr_dir, f'{scene_name}_{img_psnr}_in_{num_images}.png'), 
                      cv2.cvtColor(in_img_save, cv2.COLOR_RGB2BGR))
                metric_dict['train_psnr'] += img_psnr  #/len(pred_imgs)
                num_images += 1
                
          if means_loss_weight > 0:
            loss_dict['means_loss'] = loss_dict['means_loss'] / len(out_batch_gs) * means_loss_weight
          if laplace_loss_weight > 0:
            loss_dict['laplace_loss'] = loss_dict['laplace_loss'] / num_images / len(out_batch_gs) * laplace_loss_weight
          if log_loss_weight > 0:
            loss_dict['log_loss'] = loss_dict['log_loss'] / num_images / len(out_batch_gs) * log_loss_weight
          loss_dict['image_l1'] = loss_dict['image_l1']/num_images/len(out_batch_gs)*image_l1_loss_weight
          if lpips_loss_weight > 0:
            loss_dict['lpips'] = loss_dict['lpips']/num_images/len(out_batch_gs)*lpips_loss_weight
          if ssim_loss_weight > 0:
            loss_dict['ssim_loss'] = loss_dict['ssim_loss']/num_images/len(out_batch_gs)*ssim_loss_weight
          metric_dict['train_psnr'] = metric_dict['train_psnr']/num_images/len(out_batch_gs)
          
          if ssim_loss_weight > 0:  
            metric_dict['ssim'] = 1.0 - loss_dict['ssim_loss']
                   
          optimizer, scheduler = optimizer_['train2D'], scheduler_['train2D']
      total_loss = sum(loss_dict.values())/accumulate_step
      # # 定期清理LPIPS缓存
      # if step % 1000 == 0:
      #     lpips_loss_func.clear_cache()

      if enable_amp:
        scaler.scale(total_loss).backward()
      else:
        total_loss.backward()
      
      if (step+1) % accumulate_step == 0:
        if grad_clip_norm > 0:
          if enable_amp:
            scaler.unscale_(optimizer)
          torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        if enable_amp:
          scaler.step(optimizer)
          scaler.update()
        else:
          optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

      if empty_cache_fre > 0 and (step+1) % empty_cache_fre == 0:
        torch.cuda.empty_cache()
    
      if (step_consider_accum % log_interval == 0) and step%accumulate_step==0:
        for key, value in list(loss_dict.items()) + list(metric_dict.items()):
            torch.distributed.reduce(value, dst=0)
            if dist.get_rank() == 0:
              value = (value/torch.cuda.device_count()).item()
              wandb.log({key: value}, step=step_consider_accum)
              if step_consider_accum % (log_interval*10)==0:
                logger.info(f'Training-Step {step_consider_accum}: {key}: {value:.3f}')
              wandb.log({'lr': optimizer.param_groups[0]['lr']}, step=step_consider_accum)
      if step_consider_accum % log_image_interval == 0 and step%accumulate_step==0:
        os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
        with torch.no_grad():
          imgs, _ = gs_utils.rasterize_gaussians_to_multiimgs(
            gpu_utils.move_to_device(out_batch_gs[0], device=model.device), batch_cameras[0])
          # gt_ims, _ = gs_utils.rasterize_gaussians_to_multiimgs(
          #   gpu_utils.move_to_device(batch_gs[0], device=model.device), batch_cameras[0])
          imgs = [(im*255).cpu().numpy().astype(np.uint8) for im in imgs]
          # gt_ims = [(im*255).cpu().numpy().astype(np.uint8) for im in gt_ims]
        grid = make_grid(imgs)
        grid = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
        # gt_grid = make_grid(gt_ims)
        # gt_grid = cv2.cvtColor(gt_grid, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f'train/{step_consider_accum:08d}_pred-rank{dist.get_rank()}.png'), grid)
        # cv2.imwrite(os.path.join(output_dir, f'train/{step_consider_accum:08d}_gt_render-rank{dist.get_rank()}.png'), gt_grid)

      if ((step%accumulate_step==0) and ((step_consider_accum % eval_interval == 0) and (eval_step == step_consider_accum) or (step_consider_accum+1)==pretrain_steps)):
        model.eval()
        for test_dataset, test_loader in build_testloader().items():
            metrics, metrics_input = evaluation(model, test_loader = test_loader,
                                output_dir=output_dir+f'/eval/{test_dataset}/{step_consider_accum}', output_gt=(step_consider_accum==0), compare_with_pseudo=step_consider_accum<pretrain_steps,
                                evaluate_input=(step_consider_accum==0)) #when step==0, we evaluate the input
            if dist.get_rank() == 0:
                wandb.log({f'metrics_testscenes/{test_dataset}/{k}_testviews':v for k,v in metrics.items()}, step=step_consider_accum)
                metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics.items()])
                logger.info(f'Test {test_dataset} Step {step_consider_accum}: {metric_str}')
            dist.barrier()

      if (step%accumulate_step==0) and ((step_consider_accum+1) % save_interval == 0 or (step_consider_accum+1)==pretrain_steps): 
        if dist.get_rank()==0:
            os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(output_dir, f'checkpoints/model_{step_consider_accum:08d}.pth'))
            logger.info(f'Save model at step {step_consider_accum}')
        dist.barrier()
      model.train()
        
      if step==resume_from_step and dist.get_rank() == 0:
        with open(os.path.join(FLAGS.output_dir, 'config.gin'),'w') as f:
            f.writelines(gin.operative_config_str())
      
    return step


def main(argv):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    gin.bind_parameter('training.output_dir', FLAGS.output_dir)
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    set_seed()
    # 1. Dataloading
    train_loader = None
    if not FLAGS.only_eval:
      train_loader = build_trainloader()
    # 2. Build Model
    model = FeaturePredictor()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {num_params}')

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    
    # def print_model_info(model, checkpoint_path=None):
    #   """打印模型信息用于调试"""
      # print("Current model structure:")
      # for name, param in model.named_parameters():
      #     print(f"  {name}: {param.shape}")
      
      # if checkpoint_path:
      #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
      #     print(f"\nCheckpoint structure:")
      #     for name, param in checkpoint.items():
      #         print(f"  {name}: {param.shape}")
    if model.resume_ckpt is not None:
      # print_model_info(model, model.resume_ckpt)
      model.load_state_dict(torch.load(model.resume_ckpt,map_location='cpu'))
      print(f'Load model from {model.resume_ckpt}')
    
    # # 替换第432行附近的代码
    # if model.resume_ckpt is not None:
    #     print(f'Loading model from {model.resume_ckpt}')
    #     checkpoint = torch.load(model.resume_ckpt, map_location='cpu')
        
    #     # 获取当前模型的state_dict
    #     model_dict = model.state_dict()
        
    #     # 过滤出兼容的权重
    #     compatible_dict = {}
    #     incompatible_keys = []
        
    #     for key, value in checkpoint.items():
    #         if key in model_dict:
    #             if model_dict[key].shape == value.shape:
    #                 compatible_dict[key] = value
    #             else:
    #                 incompatible_keys.append(f"{key}: checkpoint{value.shape} vs model{model_dict[key].shape}")
    #         else:
    #             incompatible_keys.append(f"{key}: not found in current model")
        
    #     # 打印不兼容的键
    #     if incompatible_keys:
    #         print("Incompatible keys (will be skipped):")
    #         for key in incompatible_keys:
    #             print(f"  {key}")
        
    #     # 加载兼容的权重
    #     model_dict.update(compatible_dict)
    #     model.load_state_dict(model_dict)
        
    #     print(f'Successfully loaded {len(compatible_dict)}/{len(checkpoint)} parameters')

    if FLAGS.only_eval:
      assert model.resume_ckpt is not None, 'Need to specify the model checkpoint for evaluation'

    model = model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    
    if FLAGS.only_eval == False:
      model.train()
      # 3. Optimizer
      optimizer, scheduler = {}, {}
      with gin.config_scope('pretrain'):
        optimizer['pretrain'] = build_optimizer(model.module)
        scheduler['pretrain'] = build_scheduler(optimizer['pretrain'])
      with gin.config_scope('train2D'):
        optimizer['train2D'] = build_optimizer(model.module)
        scheduler['train2D'] = build_scheduler(optimizer['train2D'])

      if rank==0:
        os.makedirs(FLAGS.wandb_dir, exist_ok=True)
        wandb_mode = 'disabled' if FLAGS.disable_wandb else 'online'
        wandb_run = wandb.init(
            project='AnchorSplat',
            dir=FLAGS.wandb_dir,
            mode=wandb_mode,
        )
        if FLAGS.output_dir[-1]=='/':
          FLAGS.output_dir = FLAGS.output_dir[:-1]
        wandb.run.name = '/'.join(FLAGS.output_dir.split('/')[-2:])
      final_step = training(model, optimizer, scheduler, train_loader, output_dir=FLAGS.output_dir)

    model.eval()
    for test_dataset, test_loader in build_testloader().items():
        metrics, metrics_input = evaluation(model, test_loader = test_loader, 
                            output_dir=FLAGS.output_dir+f'/{FLAGS.eval_subdir}/{test_dataset}', 
                            compare_with_input=FLAGS.compare_with_input,
                            save_as_single=True,
                            save_viewer=FLAGS.save_viewer,
                            output_gt=True, compare_with_pseudo=False)
        if dist.get_rank() == 0:
            logger = ProcessSafeLogger(os.path.join(FLAGS.output_dir, FLAGS.eval_subdir, 'eval.log')).get_logger()
            metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics.items()])
            logger.info(f'Test-{test_dataset}: {metric_str}')
            if FLAGS.compare_with_input:
              metric_str = ' '.join([f'{k}: {v:.4f}' for k,v in metrics_input.items()])
              logger.info(f'Input 3DGS: Test-{test_dataset}: {metric_str}')
        dist.barrier()
    
    dist.destroy_process_group()

app.run(main)
