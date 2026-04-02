import torch
import numpy as np

def create_perspective_to_erp_grid(yaw, pitch, fov_deg, persp_h, persp_w, device='cpu'):
    """
    生成用于 F.grid_sample 的网格，将 ERP 全景图采样为透视图。
    yaw, pitch: 相机的偏航角和俯仰角 (角度制)
    fov_deg: 透视相机的视场角 (角度制)
    """
    # 1. 角度转弧度
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    fov_rad = np.radians(fov_deg)
    
    # 2. 计算相机焦距 (假设相机中心在图像正中央)
    f = 0.5 * persp_w / np.tan(0.5 * fov_rad)
    cx, cy = persp_w / 2.0, persp_h / 2.0
    
    # 3. 生成透视图的 2D 像素坐标网格
    u, v = torch.meshgrid(
        torch.arange(persp_w, device=device),
        torch.arange(persp_h, device=device),
        indexing='xy'
    )
    
    # 4. 将 2D 像素坐标转换为 3D 相机坐标系下的射线
    x_c = (u - cx) / f
    y_c = (v - cy) / f
    z_c = torch.ones_like(x_c)  # 在相机坐标系中，我们假设图像平面位于 Z = 1 的地方
    rays_c = torch.stack([x_c, y_c, z_c], dim=-1) # (H, W, 3)
    
    # 5. 构建旋转矩阵 (绕 X 轴旋转 Pitch，绕 Y 轴旋转 Yaw)
    Rx = torch.tensor([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad),  np.cos(pitch_rad)]
    ], dtype=torch.float32, device=device)
    
    Ry = torch.tensor([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ], dtype=torch.float32, device=device)
    
    R = Ry @ Rx
    
    # 6. 将射线旋转到世界坐标系
    rays_w = torch.einsum('ij,hwj->hwi', R, rays_c)  # 世界坐标系下指向四面八方的射线
    x_w, y_w, z_w = rays_w[..., 0], rays_w[..., 1], rays_w[..., 2]
    
    # 7. 转换为球面坐标 (经度 theta, 纬度 phi)
    theta = torch.atan2(x_w, z_w)                      # 范围 [-pi, pi]
    phi = torch.asin(y_w / torch.norm(rays_w, dim=-1)) # 范围 [-pi/2, pi/2]
    
    # 8. 归一化到 [-1, 1] 区间，供 F.grid_sample 使用
    grid_x = theta / np.pi
    grid_y = 2.0 * phi / np.pi
    
    grid = torch.stack([grid_x, grid_y], dim=-1) # (H, W, 2)   #2代表的是一对ERP全景图的(x, y)坐标值，x,y是grid的像素坐标
    return grid

# 在 utils/projection_utils.py 中补充反向投影函数
def create_erp_to_perspective_grid(yaw, pitch, fov_deg, persp_h, persp_w, erp_h, erp_w, device='cpu'):
    """
    生成用于 F.grid_sample 的网格，将透视图采样/贴回 ERP 全景图。
    """
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)
    fov_rad = np.radians(fov_deg)
    f = 0.5 * persp_w / np.tan(0.5 * fov_rad)
    cx, cy = persp_w / 2.0, persp_h / 2.0

    # 1. 生成 ERP 的球面坐标 (经纬度)
    u_erp, v_erp = torch.meshgrid(
        torch.arange(erp_w, device=device),
        torch.arange(erp_h, device=device),
        indexing='xy'
    )
    theta = (u_erp / erp_w - 0.5) * 2 * np.pi  # [-pi, pi]
    phi = (v_erp / erp_h - 0.5) * np.pi        # [-pi/2, pi/2]

    # 2. 转换为世界坐标系下的 3D 射线
    x_w = torch.sin(theta) * torch.cos(phi)
    y_w = torch.sin(phi)
    z_w = torch.cos(theta) * torch.cos(phi)
    rays_w = torch.stack([x_w, y_w, z_w], dim=-1)

    # 3. 构建反向旋转矩阵 (从世界坐标系转到当前透视相机坐标系)
    Rx = torch.tensor([[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]], device=device)
    Ry = torch.tensor([[np.cos(yaw_rad), 0, np.sin(yaw_rad)], [0, 1, 0], [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]], device=device)
    R_inv = torch.inverse(Ry @ Rx)

    # 4. 旋转到相机坐标系
    rays_c = torch.einsum('ij,hwj->hwi', R_inv.float(), rays_w.float())
    
    # 5. 透视投影到相机的 2D 像素平面
    # 注意：只保留相机前方 (Z > 0) 的点
    z_c = rays_c[..., 2]
    valid_mask = z_c > 0.01 
    
    u_persp = (rays_c[..., 0] / z_c) * f + cx
    v_persp = (rays_c[..., 1] / z_c) * f + cy
    
    # 6. 归一化到 [-1, 1]
    grid_x = (u_persp / persp_w) * 2.0 - 1.0
    grid_y = (v_persp / persp_h) * 2.0 - 1.0
    
    # 对于相机背后的点，将其移出采样区域 (设置一个极大的无效值)
    grid_x[~valid_mask] = -2.0
    grid_y[~valid_mask] = -2.0

    grid = torch.stack([grid_x, grid_y], dim=-1) # (erp_h, erp_w, 2)
    return grid, valid_mask

def get_erp_mapping(view_idx, angles_list, fov, persp_hw, erp_hw, device='cpu'):
    """
    获取将透视图深度贴回 ERP 全景图所需的采样网格。
    
    :param view_idx: 当前处理的是第几个视图 (int)
    :param angles_list: 所有视图的 (yaw, pitch) 列表
    :param fov: 透视相机的视场角 (如 90 度)
    :param persp_hw: 透视图的高宽，如 (256, 256)
    :param erp_hw: 全景图的高宽，如 (512, 1024)
    :return: grid (1, erp_h, erp_w, 2), valid_mask (1, 1, erp_h, erp_w)
    """
    yaw, pitch = angles_list[view_idx]
    persp_h, persp_w = persp_hw
    erp_h, erp_w = erp_hw
    
    # 调用底层球面投射几何数学函数
    grid, valid_mask = create_erp_to_perspective_grid(
        yaw=yaw, 
        pitch=pitch, 
        fov_deg=fov, 
        persp_h=persp_h, 
        persp_w=persp_w, 
        erp_h=erp_h, 
        erp_w=erp_w, 
        device=device
    )
    
    # 增加 Batch 维度，以满足 torch.nn.functional.grid_sample 的输入要求
    # grid: (erp_h, erp_w, 2) -> (1, erp_h, erp_w, 2)
    grid = grid.unsqueeze(0)
    
    # 增加 Batch 和 Channel 维度，方便后续做张量乘法过滤无效区域
    # valid_mask: (erp_h, erp_w) -> (1, 1, erp_h, erp_w)
    valid_mask = valid_mask.unsqueeze(0).unsqueeze(0).float()
    
    return grid, valid_mask