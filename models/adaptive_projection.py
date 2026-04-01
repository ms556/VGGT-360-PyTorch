import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.projection_utils import create_perspective_to_erp_grid

class AdaptiveProjection(nn.Module):
    def __init__(self, num_base_views=8, top_k=2, fov=90, persp_size=256):
        super().__init__()
        self.num_base_views = num_base_views
        self.top_k = top_k
        self.fov = fov
        self.persp_size = persp_size  # 透视图的分辨率 (H, W)
        
        # 预定义 8 个基础视图的 (yaw, pitch)
        # 例如：水平环绕 6 个，加上天顶(上)和天底(下) 2 个
        self.base_angles = [
            (0, 0), (60, 0), (120, 0), (180, 0), (240, 0), (300, 0),
            (0, 90), (0, -90)
        ]

    def get_base_views(self, erp_img):
        B, C, H, W = erp_img.shape
        device = erp_img.device
        base_views = []
        
        for yaw, pitch in self.base_angles:
            # 1. 生成采样网格
            grid = create_perspective_to_erp_grid(
                yaw, pitch, self.fov, self.persp_size, self.persp_size, device
            )
            # 扩展 Batch 维度: (1, H, W, 2) -> (B, H, W, 2)
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
            
            # 2. 从 ERP 中采样出透视图
            view = F.grid_sample(erp_img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
            base_views.append(view)
            
        base_views = torch.stack(base_views, dim=1) # (B, 8, C, H, W) 采样出8张透视图
        
        # 模拟生成 Valid Mask (实际工程中可通过判定 grid 是否越界或全黑区域生成)
        valid_masks = torch.ones(B, self.num_base_views, self.persp_size, self.persp_size, device=device)
        return base_views, valid_masks

    def generate_neighbor_views(self, erp_img, topk_indices):
        # topk_indices: (B, top_k)
        B = erp_img.shape[0]
        device = erp_img.device
        neighbor_views = []
        batch_neighbor_angles = [] # 新增：用于记录每个 batch 生成的邻近角度
        
        # 针对不确定性高的视角，向右上(+yaw, +pitch)和左下(-yaw, -pitch)偏移
        offsets = [(15, 15), (-15, -15)] 
        
        for b in range(B):
            b_neighbors = []
            b_angles = [] # 记录当前 batch 的新增角度
            for k in range(self.top_k):
                base_idx = topk_indices[b, k].item()
                base_yaw, base_pitch = self.base_angles[base_idx]
                
                for dy, dp in offsets:
                    new_yaw = (base_yaw + dy) % 360
                    new_pitch = torch.clamp(torch.tensor(base_pitch + dp), -90, 90).item()
                    
                    # 记录这个新生成的角度
                    b_angles.append((new_yaw, new_pitch))
                    
                    grid = create_perspective_to_erp_grid(new_yaw, new_pitch, self.fov, self.persp_size, self.persp_size, device)
                    view = F.grid_sample(erp_img[b:b+1], grid.unsqueeze(0), align_corners=False)
                    b_neighbors.append(view.squeeze(0))
            neighbor_views.append(torch.stack(b_neighbors))
            batch_neighbor_angles.append(b_angles)
            
        return torch.stack(neighbor_views), batch_neighbor_angles # (B, 2*top_k, C, H, W)
    def forward(self, erp_img):
        base_views, valid_masks = self.get_base_views(erp_img) 
        scores = self.compute_uncertainty(base_views, valid_masks)
        _, topk_indices = torch.topk(scores, self.top_k, dim=1)
        
        # 接收图像和对应的角度列表
        neighbor_views, batch_neighbor_angles = self.generate_neighbor_views(erp_img, topk_indices)
        
        final_views = torch.cat([base_views, neighbor_views], dim=1)
        
        # 假设推理时 Batch Size = 1，我们把基础角度和第一张图生成的新角度拼在一起
        # self.base_angles 是 8个基础角度，batch_neighbor_angles[0] 是 4个新增角度
        final_angles_list = self.base_angles + batch_neighbor_angles[0] 
        
        # 核心修改：同时返回图像和角度列表！
        return final_views, final_angles_list