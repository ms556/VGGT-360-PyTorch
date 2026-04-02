import sys
import os
import torch

# 假设你的目录结构是 VGGT-360-PyTorch/FastVGGT/vggt/...
# 将 FastVGGT 根目录加入路径
current_dir = os.path.dirname(os.path.abspath(__file__))
fast_vggt_path = os.path.join(os.path.dirname(current_dir), 'FastVGGT')
if fast_vggt_path not in sys.path:
    sys.path.append(fast_vggt_path)
# 正确的导入方式：指向 vggt.models.vggt 模块
from vggt.models.vggt import VGGT as build_fastvggt

def load_fastvggt_model(weights_path, device='cuda'):
    print(f"Loading official FastVGGT from {weights_path}...")
    
    # 根据 FastVGGT 的实现，通常使用构建函数实例化
    model = build_fastvggt(
        enable_depth=True,     # 确保深度预测头开启
        enable_camera=False,   # 如果你不需要预测相机位姿，可以关闭以节省显存
        enable_point=False, 
        enable_track=False,
        vis_attn_map=True,      # 【必须为True】为了后续的 Correlation-Weighted Correction
        # --- 核心修复：显式传入 None 彻底关闭全局 Token 融合 ---
        merging=None
    )
    
    # 加载权重
    checkpoint = torch.load(weights_path, map_location='cpu')
    # 获取 state_dict（适应包含额外信息的 checkpoint 字典）
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    # 加载模型权重 (使用 strict=False 以避免未开启的 Head 报错)
    model.load_state_dict(state_dict, strict=False)
    
    model = model.to(torch.bfloat16).to(device)
    model.eval()
    return model