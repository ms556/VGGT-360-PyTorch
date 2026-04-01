import torch
import cv2
import numpy as np
import argparse
from pathlib import Path

# 导入自定义模块
from datasets.panorama_dataset import PanoramaDataset
from models.adaptive_projection import AdaptiveProjection
from models.enhanced_attention import inject_enhanced_attention
from models.model_correction import compute_correlation_weights, blend_to_erp
from utils.projection_utils import get_perspective_coords, get_erp_mapping

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-360 Zero-Shot Panoramic Depth Estimation")
    parser.add_argument("--img_path", type=str, required=True, help="Path to input ERP panorama")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save depth maps")
    parser.add_argument("--vggt_weights", type=str, default="weights/vggt_pretrained.pth")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模块
    print("Initializing modules...")
    adaptive_proj = AdaptiveProjection(num_base_views=8, top_k=2).to(device)
    
    # 假设这是从官方或第三方加载的原始 VGGT 模型
    # vggt_model = build_vggt_model(checkpoint=args.vggt_weights).to(device)
    # inject_enhanced_attention(vggt_model) # 动态注入结构感知先验
    
    # 2. 加载与预处理图像
    print(f"Processing panorama: {args.img_path}")
    # 这里为了演示，直接使用 cv2 读取。实际工程中可调用 datasets 里的类
    img = cv2.imread(args.img_path) # 注意：cv2 默认读取为 BGR，转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    erp_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    erp_hw = erp_tensor.shape[-2:]
    
    # 3. 阶段一：自适应投影 (Adaptive Projection)
    # views: (1, 12, 3, H, W) -> 8个基础视图 + 4个增强视图
    multi_views , angles_list= adaptive_proj(erp_tensor) 
    num_views = multi_views.shape[1]
    # 4. 阶段二：VGGT 3D 推理与增强注意力 (Enhanced Attention)
    print("Running VGGT-like 3D reasoning...")
    # pred_depths: (1, 12, 1, H, W), attn_maps: 最后一层注意力矩阵
    # pred_depths, attn_maps = vggt_model(multi_views)
    
    # [模拟模型输出，仅作流程跑通用]
    num_views = multi_views.shape[1]
    pred_depths = torch.rand((1, num_views, 1, 256, 256), device=device)
    attn_maps = torch.rand((1, num_views, 1024, 1024), device=device)
    
    # 5. 阶段三：相关性加权 3D 校正 (Correlation-Weighted Correction)
    print("Applying Correlation-Weighted 3D Correction and blending...")
    final_erp_depth = torch.zeros(erp_hw, device=device)
    erp_weight_sum = torch.zeros(erp_hw, device=device)
    
    # for i in range(num_views):
    #     # 计算该视图的相关性权重
    #     spatial_coords = get_perspective_coords(256, 256).to(device)
    #     weight_map = compute_correlation_weights(attn_maps[:, i], spatial_coords)
        
    #     # 将透视图深度与权重映射回 ERP 坐标系
    #     # mapping_grid: 用于 F.grid_sample 的网格
    #     mapping_grid = get_erp_mapping(view_idx=i, erp_hw=erp_hw)
        
    #     mapped_depth = torch.nn.functional.grid_sample(pred_depths[:, i], mapping_grid)
    #     mapped_weight = torch.nn.functional.grid_sample(weight_map.unsqueeze(1), mapping_grid)
        
    #     final_erp_depth += mapped_depth.squeeze() * mapped_weight.squeeze()
    #     erp_weight_sum += mapped_weight.squeeze()
    for i in range(num_views):
        # 1. 获取并生成透视图 2D 坐标
        spatial_coords = get_perspective_coords(256, 256, device=device)
        
        # 2. 计算当前视图的 3D 相关性融合权重
        weight_map = compute_correlation_weights(attn_maps[:, i], spatial_coords)
        
        # 3. 获取向 ERP 贴图的几何映射网格
        mapping_grid, valid_mask = get_erp_mapping(
            view_idx=i, 
            angles_list=angles_list, # 你需要维护一个所有生成视图角度的列表
            fov=90, 
            persp_hw=(256, 256), 
            erp_hw=erp_hw, 
            device=device
        )
        
        # 4. 执行重采样投影 (从 Perspective 映射回 ERP)
        mapped_depth = torch.nn.functional.grid_sample(pred_depths[:, i:i+1], mapping_grid, align_corners=False)
        mapped_weight = torch.nn.functional.grid_sample(weight_map.unsqueeze(0).unsqueeze(0), mapping_grid, align_corners=False)
        
        # 5. 屏蔽由于相机背后映射带来的越界伪影
        mapped_depth = mapped_depth * valid_mask
        mapped_weight = mapped_weight * valid_mask
        
        final_erp_depth += mapped_depth.squeeze() * mapped_weight.squeeze()
        erp_weight_sum += mapped_weight.squeeze()
        
    final_erp_depth = final_erp_depth / (erp_weight_sum + 1e-8)
    
    # 6. 保存结果
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    save_path = output_path / f"{Path(args.img_path).stem}_depth.png"
    
    # 深度图伪彩色可视化归一化
    depth_vis = final_erp_depth.cpu().numpy()
    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    
    cv2.imwrite(str(save_path), depth_vis)
    print(f"Done! Depth map saved to {save_path}")

if __name__ == "__main__":
    main()