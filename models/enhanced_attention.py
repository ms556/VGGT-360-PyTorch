import torch.nn as nn
import torch.nn.functional as F

def inject_enhanced_attention(vggt_model, log_Ms_tensor):
    def new_forward(self, x, **kwargs): # 增加 **kwargs 兼容 FastVGGT 原生的形参
        # 复制原有的逻辑，但在计算 softmax 前加上 self.structure_bias
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # --- 关键注入点 ---
        if hasattr(self, 'structure_bias'):
            # log_Ms_tensor 形状应为 (N, N) 或 (1, 1, N, N)
            bias = self.structure_bias
            L = bias.shape[-1] # L 通常为 324
            
            # 如果进入 Attention 的序列长度 N (329) 大于 结构偏置的长度 L (324)
            if N > L:
                num_special = N - L
                # 在偏置矩阵的左侧和上侧补齐 num_special 个 0
                # 补零(0.0)意味着这 5 个特殊 Token 不受局部几何的偏置影响
                bias = F.pad(bias, (num_special, 0, num_special, 0), value=0.0)
            
            # --- 核心修复：防止广播错误，增加 num_heads 维度 ---
            # 如果 bias 只有 3 维 (Batch, N, N)，则拓展为 (Batch, 1, N, N)
            if bias.dim() == 3:
                bias = bias.unsqueeze(1)
                  
            attn = attn + bias 
        
        attn = attn.softmax(dim=-1)
        # 保存用于后期 3D 校正的 map
        self.saved_attn_map = attn.detach() 
        # --- 核心修复：将 softmax 自动升维的 float32 强转回 bfloat16 以匹配 v ---
        attn = attn.to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

    # 核心修复：修改遍历路径为 aggregator.frame_blocks
    for block in vggt_model.aggregator.frame_blocks:
        # 替换 forward 方法 (Monkey Patch)
        block.attn.forward = new_forward.__get__(block.attn, block.attn.__class__)
        block.attn.structure_bias = log_Ms_tensor