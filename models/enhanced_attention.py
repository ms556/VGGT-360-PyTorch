# 修改 models/enhanced_attention.py
import torch.nn as nn

def inject_enhanced_attention(vggt_model, log_Ms_tensor):
    def new_forward(self, x):
        # 复制原有的逻辑，但在计算 softmax 前加上 self.structure_bias
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # --- 关键注入点 ---
        if hasattr(self, 'structure_bias'):
            # log_Ms_tensor 形状应为 (N, N) 或 (1, 1, N, N)
            attn = attn + self.structure_bias 
        
        attn = attn.softmax(dim=-1)
        # 保存用于后期 3D 校正的 map
        self.saved_attn_map = attn.detach() 
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    for block in vggt_model.blocks:
        # 替换 forward 方法
        block.attn.forward = new_forward.__get__(block.attn, block.attn.__class__)
        block.attn.structure_bias = log_Ms_tensor