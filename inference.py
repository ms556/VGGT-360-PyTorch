import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import torch.nn.functional as F

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
    
    # 假设你已经计算好了置信度矩阵 M_s 并转为 log_Ms
    # inject_enhanced_attention(vggt_model, log_Ms)
    
    # 真实前向传播 (FastVGGT 通常返回字典或只返回深度)
    pred_depths = vggt_model(multi_views)
    raw_attn_maps = vggt_model.blocks[-1].attn.saved_attn_map
    
    # [模拟模型输出，仅作流程跑通用]
    num_views = multi_views.shape[1]
    pred_depths = torch.rand((1, num_views, 1, 256, 256), device=device)
    attn_maps = torch.rand((1, num_views, 1024, 1024), device=device)
    
    # 5. 阶段三：相关性加权 3D 校正 (Correlation-Weighted Correction)
    print("Applying Correlation-Weighted 3D Correction and blending...")
    
    # 假设 FastVGGT 的 patch_size 为 14
    patch_size = 14
    h_feat, w_feat = 256 // patch_size, 256 // patch_size
    spatial_coords = get_perspective_coords(h_feat, w_feat, device=device)
    multi_view_depths_list = []
    multi_view_weights_list = []
    
    # 步骤 A: 提取每个视角的深度，并计算对应的校正权重
    for i in range(num_views):
        # 1. 提取深度并加入列表，形状保持为 (1, 1, 256, 256)
        depth_i = pred_depths[:, i:i+1]
        multi_view_depths_list.append(depth_i)
        
        # 2. 获取当前视角的注意力图 (注意：这里应当是去除 CLS token 后的多头平均结果)
        attn_map_i = attn_maps[:, i]
        
        # 3. 计算 Patch 级别的权重图 (h_feat x w_feat)
        weight_map_feat = compute_correlation_weights(attn_map_i, spatial_coords)
        
        # 4. 【关键】：将 Patch 级别的权重插值放大回完整的像素级别 (256x256)
        weight_map_pixel = torch.nn.functional.interpolate(
            weight_map_feat.unsqueeze(0).unsqueeze(0), 
            size=(256, 256), 
            mode='bilinear', 
            align_corners=False
        )
        multi_view_weights_list.append(weight_map_pixel)
        
    # 步骤 B: 直接调用 blend_to_erp 进行全景融合投影
    final_erp_depth = blend_to_erp(
        multi_view_depths=multi_view_depths_list,
        multi_view_weights=multi_view_weights_list,
        angles_list=angles_list,
        fov=90,
        erp_hw=erp_hw
    )
    
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