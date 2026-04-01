from utils.projection_utils import create_erp_to_perspective_grid
import torch
import numpy as np
import torch.nn.functional as F

def blend_to_erp(multi_view_depths, multi_view_weights, angles_list, fov, erp_hw):
    """
    multi_view_depths: list of (1, 1, H, W)
    multi_view_weights: list of (1, 1, H, W) 包含了 sharpness, locality, symmetry 的相关性权重
    angles_list: list of (yaw, pitch) 包含基础视图和动态生成的增强视图的角度
    """
    erp_h, erp_w = erp_hw
    device = multi_view_depths[0].device
    
    erp_depth_sum = torch.zeros((1, 1, erp_h, erp_w), device=device)
    erp_weight_sum = torch.zeros((1, 1, erp_h, erp_w), device=device)
    
    persp_h, persp_w = multi_view_depths[0].shape[-2:]
    
    for i in range(len(angles_list)):
        yaw, pitch = angles_list[i]
        
        # 1. 获取重投影网格
        grid, valid_mask = create_erp_to_perspective_grid(
            yaw, pitch, fov, persp_h, persp_w, erp_h, erp_w, device
        )
        grid = grid.unsqueeze(0) # (1, erp_h, erp_w, 2)
        
        # 2. 将透视图的深度和权重采样到 ERP 画布上
        mapped_depth = F.grid_sample(multi_view_depths[i], grid, mode='bilinear', align_corners=False)
        mapped_weight = F.grid_sample(multi_view_weights[i], grid, mode='bilinear', align_corners=False)
        
        # 3. 屏蔽掉相机背后的无效区域投影
        valid_mask_tensor = valid_mask.unsqueeze(0).unsqueeze(0).float()
        mapped_depth = mapped_depth * valid_mask_tensor
        mapped_weight = mapped_weight * valid_mask_tensor
        
        # 4. 累加
        erp_depth_sum += mapped_depth * mapped_weight
        erp_weight_sum += mapped_weight
        
    # 5. 加权平均
    final_erp_depth = erp_depth_sum / (erp_weight_sum + 1e-8)
    return final_erp_depth

def get_perspective_coords(h, w, device='cpu'):
    """
    生成透视图像的 2D 空间像素坐标，并归一化到 [0, 1] 区间，
    最后拉平以匹配 Attention Map 的 (N_points, N_points) 维度。
    """
    # 1. 生成网格坐标 (y, x)
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing='ij'
    )
    
    # 2. 组合成 (H, W, 2)
    coords = torch.stack([x, y], dim=-1)
    
    # 3. 归一化到 [0, 1] 以保证计算高斯距离时的尺度一致性
    coords[..., 0] = coords[..., 0] / (w - 1)
    coords[..., 1] = coords[..., 1] / (h - 1)
    
    # 4. 拉平为 (N_points, 2)，其中 N_points = h * w
    coords_flattened = coords.view(-1, 2)
    return coords_flattened

def compute_correlation_weights(attn_map, spatial_coords, sigma=0.1):
    """
    计算相关性 3D 校正权重。
    :param attn_map: (N_points, N_points) VGGT 最后一层帧内注意力图
    :param spatial_coords: (N_points, 2) 由 get_perspective_coords 生成的坐标
    :param sigma: 高斯核的超参数，控制局部性的衰减速度
    :return: (H, W) 形状的权重图
    """
    N_points = attn_map.shape[-1]
    device = attn_map.device
    
    # 为了防止 log(0) 或 sqrt(0) 导致 NaN，设置一个极小值
    eps = 1e-8 
    
    # ---------------------------------------------------------
    # 1. Sharpness (锐度) - 基于香农熵
    # ---------------------------------------------------------
    entropy = -torch.sum(attn_map * torch.log(attn_map + eps), dim=-1)
    max_entropy = np.log(N_points)
    S_sharp = 1.0 - (entropy / max_entropy)
    
    # ---------------------------------------------------------
    # 2. Locality (局部性) - 基于高斯核距离加权
    # ---------------------------------------------------------
    # 计算所有点对之间的欧式距离的平方: (N_points, N_points)
    dist_sq = torch.cdist(spatial_coords, spatial_coords).pow(2)
    gaussian_kernel = torch.exp(-dist_sq / (2 * sigma**2))
    S_loc = torch.sum(attn_map * gaussian_kernel, dim=-1)
    
    # ---------------------------------------------------------
    # 3. Symmetry (对称性) - 基于巴氏系数
    # ---------------------------------------------------------
    attn_map_t = attn_map.transpose(-2, -1)
    S_sym = torch.sum(torch.sqrt(attn_map * attn_map_t + eps), dim=-1)
    
    # ---------------------------------------------------------
    # 4. 融合与归一化
    # ---------------------------------------------------------
    C_p = S_sharp + S_loc + S_sym
    
    # Max-Min 归一化，将权重映射到 [0, 1] 区间
    C_p_min = C_p.min()
    C_p_max = C_p.max()
    C_p_norm = (C_p - C_p_min) / (C_p_max - C_p_min + eps)
    
    # 5. 将拉平的权重 (N_points,) 还原回 2D 图像维度 (H, W)
    h = int(np.sqrt(N_points))
    w = h
    weight_map = C_p_norm.view(h, w)
    
    return weight_map