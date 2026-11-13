"""优化 pass 注册表。

每个 pass 是一个"可应用 / 可回滚 / 会自述原因"的独立步骤，顺序由 `ORDER` 固定：

1. `hybrid_linear` —— 把 denoiser 里的 nn.Linear 换成 fp16 张量核 + fp32 累加（含在线预缩放）
2. `attn_fp16`     —— 把 SDPA 输入转 fp16，落进 sm_70 的 cutlass fmha 内核
3. `decoder_fp16`  —— VAE/decoder 转 fp16
"""

from __future__ import annotations

from .base import Context, Pass, PatchReport, get_pass, order_passes, register

# 导入即注册（顺序 = 应用顺序）
from .hybrid_linear import HybridLinearPass  # noqa: E402,F401
from .attn_fp16 import AttentionFp16Pass  # noqa: E402,F401
from .decoder_fp16 import DecoderFp16Pass  # noqa: E402,F401

__all__ = ["Context", "Pass", "PatchReport", "get_pass", "order_passes", "register",
           "HybridLinearPass", "AttentionFp16Pass", "DecoderFp16Pass"]
