import torch
import cv2
import numpy as np
import argparse
from pathlib import Path
import torch.nn.functional as F

# --- 修改 1：修正导入路径 ---
from datasets.panorama_dataset import PanoramicDepthDataset
from models.adaptive_projection import AdaptiveProjection
from models.enhanced_attention import inject_enhanced_attention
from models.model_correction import compute_correlation_weights, blend_to_erp, get_perspective_coords, compute_structure_saliency_bias
from utils.projection_utils import get_erp_mapping
from models.vggt_wrapper import load_fastvggt_model # 新增：导入模型封装

def parse_args():
    parser = argparse.ArgumentParser(description="VGGT-360 Zero-Shot Panoramic Depth Estimation")
    parser.add_argument("--img_path", type=str, default="input_images", required=True, help="Path to input ERP panorama")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save depth maps")
    parser.add_argument("--vggt_weights", type=str, default="/ssd/ms/My_model/VGGT-360-PyTorch/FastVGGT/weights/model_tracker_fixed_e20.pt")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 初始化模块
    print("Initializing modules...")
    adaptive_proj = AdaptiveProjection(num_base_views=8, top_k=2, persp_size=252).to(device)
    
    # --- 修改 2：正确加载模型 ---
    
    vggt_model = load_fastvggt_model(args.vggt_weights, device)
    
    # 2. 加载与预处理图像
    print(f"Processing panorama: {args.img_path}")
    img = cv2.imread(args.img_path) # 注意：cv2 默认读取为 BGR，转换为 RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    erp_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    erp_hw = erp_tensor.shape[-2:]
    
    # 3. 阶段一：自适应投影 (Adaptive Projection)
    # views: (1, 12, 3, H, W) -> 8个基础视图 + 4个增强视图
    multi_views , angles_list= adaptive_proj(erp_tensor) 
    # 移除 Batch 维度以便循环处理每个 view
    multi_views = multi_views.squeeze(0).to(torch.bfloat16) # 变为 (N, C, H, W)
    
    num_views = multi_views.shape[0]
    # 4. # 在 inference.py 的阶段二部分
    print("Running VGGT-like 3D reasoning with SEA...")

    # 1. 计算结构显著性偏置
    log_Ms = compute_structure_saliency_bias(multi_views, patch_size=14)
    log_Ms = log_Ms.to(torch.bfloat16)
    # 2. 注入模型 
    inject_enhanced_attention(vggt_model, log_Ms)
      
    # 真实前向传播 (FastVGGT 通常返回字典或只返回深度)
    pred_depths = vggt_model(multi_views)
    # 注意：如果 FastVGGT 返回的是字典，需要改为 pred_depths = pred_depths['depth'] 等键值
    
    raw_attn_maps = vggt_model.aggregator.frame_blocks[-1].attn.saved_attn_map
    
    # --- 修改 4：处理多头注意力图，将形状 (N, heads, L, L) 平均为 (N, L, L) ---
    attn_maps = raw_attn_maps.mean(dim=1)
    
    # 5. 阶段三：相关性加权 3D 校正 (Correlation-Weighted Correction)
    print("Applying Correlation-Weighted 3D Correction and blending...")
    
    # 假设 FastVGGT 的 patch_size 为 14
    patch_size = 14
    persp_size = 252
    h_feat, w_feat = persp_size // patch_size, persp_size // patch_size
    spatial_coords = get_perspective_coords(h_feat, w_feat, device=device)
    spatial_coords = get_perspective_coords(h_feat, w_feat, device=device)
    multi_view_depths_list = []
    multi_view_weights_list = []
    
    # 步骤 A: 提取每个视角的深度，并计算对应的校正权重
    for i in range(num_views):
        # 1. 提取深度并加入列表，形状保持为 (1, 1, 256, 256)
        depth_i = pred_depths[:, i:i+1]
        multi_view_depths_list.append(depth_i)
        
        # 2. 获取当前视角的注意力图 (注意：这里应当是去除 CLS token 后的多头平均结果)
        attn_map_i = attn_maps[i]
        
        # 3. 计算 Patch 级别的权重图 (h_feat x w_feat)
        weight_map_feat = compute_correlation_weights(attn_map_i, spatial_coords)
        
        # 4. 【关键】：将 Patch 级别的权重插值放大回完整的像素级别 (256x256)
        weight_map_pixel = torch.nn.functional.interpolate(
            weight_map_feat.unsqueeze(0).unsqueeze(0), 
            size=(persp_size, persp_size), 
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
    
    
'''
python inference.py \
    --img_path /ssd/ms/My_model/VGGT-360-PyTorch/input_images/__EPszp5486MewfwSMqmSQ,37.742475,-122.404157,.jpg \
    --vggt_weights FastVGGT/weights/model_tracker_fixed_e20.pt \
    --output_dir ./output
'''
