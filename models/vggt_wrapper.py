import sys
import torch
# 将拉下来的 FastVGGT 目录加入系统路径，确保能正常 import 其内部模块
sys.path.append('./FastVGGT')

# 导入 FastVGGT 真实的构建函数 (具体导入路径需参考 eval_scannet.py 内部写法)
from fastvggt.models import build_fastvggt 

def load_fastvggt_model(weights_path, device='cuda'):
    print("Loading official FastVGGT...")
    
    # 核心修改点：必须传入 FastVGGT 特有的控制参数
    model = build_fastvggt(
        pretrained=weights_path, 
        merging=0,              # 核心参数：指定从哪一层开始进行 Token Merging
        vis_attn_map=True       # 刚需参数：强制模型在前向传播时返回注意力图，否则后续无法做 3D 校正！
    )
    
    model = model.to(device)
    model.eval()
    return model