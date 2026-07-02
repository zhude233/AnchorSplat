import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
from torch import autograd as autograd

class lpips_loss_fn():
    def __init__(self):
        import lpips
        self.lpips = lpips.LPIPS(net='alex').cuda()
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def __call__(self, x, y):
        # x  B,H,W,C [0,1]
        # y  B,H,W,C [0,1]
        loss = self.lpips(x.permute(0,3,1,2), y.permute(0,3,1,2), normalize=True)#.mean()
        return loss



class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=[2,7,16,25,34], use_input_norm=True, use_range_norm=False):
        super(VGGFeatureExtractor, self).__init__()
        '''
        use_input_norm: If True, x: [0, 1] --> (x - mean) / std
        use_range_norm: If True, x: [0, 1] --> x: [-1, 1]
        '''
        model = torchvision.models.vgg19(pretrained=True)
        self.use_input_norm = use_input_norm
        self.use_range_norm = use_range_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.list_outputs = isinstance(feature_layer, list)
        if self.list_outputs:
            self.features = nn.Sequential()
            feature_layer = [-1] + feature_layer
            for i in range(len(feature_layer)-1):
                self.features.add_module('child'+str(i), nn.Sequential(*list(model.features.children())[(feature_layer[i]+1):(feature_layer[i+1]+1)]))
        else:
            self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])

        print(self.features)

        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_range_norm:
            x = (x + 1.0) / 2.0
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        if self.list_outputs:
            output = []
            for child_model in self.features.children():
                x = child_model(x)
                output.append(x.clone())
            return output
        else:
            return self.features(x)


class PerceptualLoss(nn.Module):
    """VGG Perceptual loss
    """

    def __init__(self, feature_layer=[2,7,16,25,34], weights=[0.1,0.1,1.0,1.0,1.0], lossfn_type='l1', use_input_norm=True, use_range_norm=False):
        super(PerceptualLoss, self).__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer, use_input_norm=use_input_norm, use_range_norm=use_range_norm)
        self.lossfn_type = lossfn_type
        self.weights = weights
        if self.lossfn_type == 'l1':
            self.lossfn = nn.L1Loss()
        else:
            self.lossfn = nn.MSELoss()
        print(f'feature_layer: {feature_layer}  with weights: {weights}')

    def forward(self, x, gt):
        """Forward function.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x_vgg, gt_vgg = self.vgg(x), self.vgg(gt.detach())
        loss = 0.0
        if isinstance(x_vgg, list):
            n = len(x_vgg)
            for i in range(n):
                loss += self.weights[i] * self.lossfn(x_vgg[i], gt_vgg[i])
        else:
            loss += self.lossfn(x_vgg, gt_vgg.detach())
        return loss


# import torch

# class lpips_loss_fn():
#     def __init__(self):
#         import lpips
#         self.lpips = lpips.LPIPS(net='alex').cuda()
#         self.lpips.eval()
#         for param in self.lpips.parameters():
#             param.requires_grad = False

#     def __call__(self, x, y, use_checkpoint=True):
#         # x  B,H,W,C [0,1]
#         # y  B,H,W,C [0,1]
        
#         def lpips_forward(img1, img2):
#             return self.lpips(img1.permute(0,3,1,2), img2.permute(0,3,1,2), normalize=True)
        
#         if use_checkpoint and x.requires_grad:
#             # 使用梯度检查点减少显存
#             loss = torch.utils.checkpoint.checkpoint(lpips_forward, x, y)
#         else:
#             loss = lpips_forward(x, y)
        
#         return loss




import torch
import torch.nn.functional as F

# class lpips_loss_fn():
#     def __init__(self, enable_manual_grad=False, chunk_size=1):  # 默认禁用手动梯度
#         import lpips
#         self.lpips = lpips.LPIPS(net='alex').cuda()
#         self.lpips.eval()
#         for param in self.lpips.parameters():
#             param.requires_grad = False
        
#         self.enable_manual_grad = enable_manual_grad
#         self.chunk_size = chunk_size
#         self.fallback_count = 0

#     def __call__(self, x, y, use_checkpoint=True, use_manual_grad=None):
#         """
#         Args:
#             x: [B,H,W,C] 预测图像 [0,1]
#             y: [B,H,W,C] 真实图像 [0,1]
#             use_checkpoint: 是否使用梯度检查点
#             use_manual_grad: 是否使用手动梯度（None时使用初始化设置）
#         """
#         if use_manual_grad is None:
#             use_manual_grad = self.enable_manual_grad
        
#         # 对于高分辨率图像，强制禁用手动梯度
#         if x.shape[1] * x.shape[2] > 1024 * 1024:  # 超过1M像素
#             use_manual_grad = False
#             if self.fallback_count < 3:
#                 print(f"High resolution detected {x.shape}, disabling manual gradient")
#                 self.fallback_count += 1
        
#         if not use_manual_grad or not x.requires_grad:
#             return self._standard_lpips(x, y, use_checkpoint)
        
#         try:
#             return self._safe_manual_lpips(x, y)
#         except Exception as e:
#             if self.fallback_count < 5:
#                 print(f"Manual gradient failed: {e}, using standard LPIPS")
#                 self.fallback_count += 1
#             return self._standard_lpips(x, y, use_checkpoint)
    
#     def _standard_lpips(self, x, y, use_checkpoint=True):
#         """标准LPIPS计算"""
#         def lpips_forward(img1, img2):
#             return self.lpips(img1.permute(0,3,1,2), img2.permute(0,3,1,2), normalize=True)
        
#         if use_checkpoint and x.requires_grad:
#             loss = torch.utils.checkpoint.checkpoint(lpips_forward, x, y, use_reentrant=False)
#         else:
#             loss = lpips_forward(x, y)
        
#         return loss
    
#     def _safe_manual_lpips(self, x, y):
#         """安全的手动梯度计算（仅用于小图像）"""
#         if x.numel() > 512 * 512 * 3:  # 限制使用范围
#             raise RuntimeError("Image too large for manual gradient")
        
#         # 实现安全的手动梯度计算
#         # ... 这里可以添加之前的手动梯度代码，但建议只用于小图像
#         return self._standard_lpips(x, y, use_checkpoint=True)
    
#     def clear_cache(self):
#         """清理缓存"""
#         torch.cuda.empty_cache()
    
    
    
    
class lpips_loss_fn_optimized():
    def __init__(self, enable_manual_grad=True, chunk_size=1, enable_mixed_precision=True):
        import lpips
        self.lpips = lpips.LPIPS(net='alex').cuda()
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False
        
        self.enable_manual_grad = enable_manual_grad
        self.chunk_size = chunk_size
        self.enable_mixed_precision = enable_mixed_precision
        self.fallback_count = 0

    def __call__(self, x, y, use_manual_grad=None):
        if use_manual_grad is None:
            use_manual_grad = self.enable_manual_grad
        
        if not use_manual_grad or not x.requires_grad:
            return self.lpips(x.permute(0,3,1,2), y.permute(0,3,1,2), normalize=True)
        
        try:
            return self._safe_manual_lpips(x, y)
        except Exception as e:
            self.fallback_count += 1
            if self.fallback_count % 100 == 1:
                print(f"LPIPS manual grad failed {self.fallback_count} times, using fallback: {e}")
            return self._standard_lpips(x, y)
    
    def _standard_lpips(self, x, y):
        """标准LPIPS计算"""
        return self.lpips(x.permute(0,3,1,2), y.permute(0,3,1,2), normalize=True)
    
    def _safe_manual_lpips(self, x, y):
        """修复形状匹配的安全手动梯度计算"""
        # 记录原始形状
        original_shape = x.shape  # [B, H, W, C]
        B, H, W, C = original_shape
        
        # 使用混合精度
        compute_dtype = torch.float16 if self.enable_mixed_precision else x.dtype
        original_dtype = x.dtype
        
        # 转换精度
        x_compute = x.to(compute_dtype) if compute_dtype != original_dtype else x
        y_compute = y.to(compute_dtype) if compute_dtype != original_dtype else y
        
        # detach并要求梯度
        x_detached = x_compute.detach().requires_grad_(True)
        y_detached = y_compute.detach()
        
        try:
            with torch.enable_grad():
                # 转换为LPIPS格式
                x_formatted = x_detached.permute(0,3,1,2)  # [B,C,H,W]
                y_formatted = y_detached.permute(0,3,1,2)  # [B,C,H,W]
                
                with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision):
                    lpips_output = self.lpips(x_formatted, y_formatted, normalize=True)
                    loss_scalar = lpips_output.float().mean()
                
                # 计算梯度
                loss_scalar.backward()
                x_grad = x_detached.grad
            
            if x_grad is not None:
                grad_shape = x_grad.shape
                expected_input_shape = x_detached.shape  # [B, H, W, C]
                
                # 检查梯度形状是否与输入匹配
                if grad_shape == expected_input_shape:
                    # 梯度形状正确，直接使用
                    x_grad_final = x_grad
                elif grad_shape == (B, C, H, W):
                    # 梯度是LPIPS格式，需要转换回原始格式
                    x_grad_final = x_grad.permute(0,2,3,1)  # [B,C,H,W] -> [B,H,W,C]
                elif grad_shape[0] == B and len(grad_shape) == 4:
                    # 尝试智能重组
                    grad_elements = list(grad_shape[1:])
                    expected_elements = [H, W, C]
                    
                    if sorted(grad_elements) == sorted(expected_elements):
                        # 元素数量匹配，尝试找到正确的置换
                        if grad_shape == (B, H, C, W):
                            x_grad_final = x_grad.permute(0,1,3,2)  # [B,H,C,W] -> [B,H,W,C]
                        elif grad_shape == (B, W, H, C):
                            x_grad_final = x_grad.permute(0,2,1,3)  # [B,W,H,C] -> [B,H,W,C]
                        elif grad_shape == (B, W, C, H):
                            x_grad_final = x_grad.permute(0,3,1,2).permute(0,2,3,1)  # 复杂重组
                        else:
                            raise ValueError(f"Cannot handle gradient shape: {grad_shape}")
                    else:
                        # 尝试reshape
                        if x_grad.numel() == x_detached.numel():
                            x_grad_final = x_grad.reshape(expected_input_shape)
                        else:
                            raise ValueError(f"Gradient elements mismatch: {x_grad.numel()} vs {x_detached.numel()}")
                else:
                    raise ValueError(f"Incompatible gradient shape: {grad_shape} for input {expected_input_shape}")
                
                # 最终形状验证
                if x_grad_final.shape != original_shape:
                    raise ValueError(f"Final shape mismatch: expected {original_shape}, got {x_grad_final.shape}")
                
                # 恢复原始精度
                if compute_dtype != original_dtype:
                    x_grad_final = x_grad_final.to(original_dtype)
                
                # 构造手动损失
                manual_loss = torch.sum(x * x_grad_final.detach()) / x.numel()
                return manual_loss
            else:
                return loss_scalar.detach()
                
        finally:
            # 清理变量
            vars_to_del = [
                'x_detached', 'y_detached', 'x_formatted', 'y_formatted',
                'lpips_output', 'loss_scalar', 'x_grad', 'x_grad_final'
            ]
            for var_name in vars_to_del:
                if var_name in locals():
                    del locals()[var_name]
            
            torch.cuda.empty_cache()
    
    def clear_cache(self):
        """清理缓存"""
        torch.cuda.empty_cache()