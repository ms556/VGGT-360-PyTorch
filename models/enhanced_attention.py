def inject_enhanced_attention(vggt_model, log_Ms_tensor):
    # 遍历模型的所有 transformer block
    for block in vggt_model.blocks:
        # 强行给注意力对象绑定一个属性
        block.attn.structure_bias = log_Ms_tensor