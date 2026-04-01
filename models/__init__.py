# models/__init__.py 
# 将子文件中的核心类/函数导入到当前层级
from .adaptive_projection import AdaptiveProjection
from .enhanced_attention import inject_enhanced_attention
from .model_correction import compute_correlation_weights, blend_to_erp

# 可选：定义 __all__ 列表，明确指出当别人使用 `from models import *` 时，会导入哪些东西
__all__ = [
    "AdaptiveProjection",
    "inject_enhanced_attention",
    "compute_correlation_weights",
    "blend_to_erp"
]