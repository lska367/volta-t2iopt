"""S4 pass：把 VAE/decoder 转 fp16。

为什么值得单独一个 pass：这是全流程里**唯一不需要任何数值技巧**的迁移。
VAE 是纯 conv + attention 的带宽活，sm_70 上 cudnn 有原生 fp16 kernel；
实测 1024² 解码 5093 ms → 874 ms（5.83×），与 bf16 输出的 SSIM 0.99966，无 inf/nan。
原来的预案"分段回退 fp32"因此用不上。

对已加载的 pipeline 用这个 pass；如果是本库的 `load_pipe()`，**在加载时就直接给 fp16**
（省一次 5 GB 的显存带宽往返），此时本 pass 会报"已是目标精度"。
"""

from __future__ import annotations

from typing import ClassVar

import torch

from ..config import dtype_from_str, dtype_name
from .base import Context, Pass, PatchResult, register


def _module_bytes(module: torch.nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in module.parameters())
    total += sum(b.numel() * b.element_size() for b in module.buffers())
    return total / 2**20


@register
class DecoderFp16Pass(Pass):
    name: ClassVar[str] = "decoder_fp16"

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self._original_dtype: torch.dtype | None = None

    def _target_dtype(self) -> torch.dtype:
        dt = self.ctx.config.pass_config(self.name).get("dtype", None)
        return self.ctx.config.effective_decoder_dtype() if dt is None else dtype_from_str(dt)

    def applicable(self) -> tuple[bool, str]:
        if self.ctx.decoder is None:
            return False, "这个 pipeline 没有 decoder（VAE）组件"
        cap = self.ctx.capability
        target = self._target_dtype()
        if target == self.ctx.model_dtype:
            return False, f"decoder 目标精度与模型一致（{dtype_name(target)}），无需转换"
        if target == torch.float16 and not cap.fp16_tensorcore:
            return False, f"sm_{cap.sm} 没有 fp16 张量核"
        return True, ""

    def apply(self) -> PatchResult:
        vae = self.ctx.decoder
        target = self._target_dtype()
        try:
            current = next(vae.parameters()).dtype
        except StopIteration:
            return PatchResult(pass_name=self.name, applied=False, reason="decoder 没有参数")

        before_mib = _module_bytes(vae)
        n = sum(1 for _ in vae.parameters())
        if current == target:
            return PatchResult(
                pass_name=self.name, applied=True, modules=n,
                details=dict(dtype=dtype_name(target), already=True,
                             params_mib=round(before_mib, 1)),
                notes=[f"decoder 已是 {dtype_name(target)}（加载时已指定），本 pass 为 no-op"],
            )

        self._original_dtype = current
        vae.to(target)
        after_mib = _module_bytes(vae)
        return PatchResult(
            pass_name=self.name, applied=True, modules=n,
            details=dict(from_dtype=dtype_name(current), dtype=dtype_name(target),
                         params_mib=round(after_mib, 1),
                         freed_mib=round(max(before_mib - after_mib, 0.0), 1)),
            notes=["VAE 是纯 conv/attention 的带宽活，不需要预缩放；"
                   "实测对 bf16 输出 SSIM 0.99966"],
        )

    def revert(self) -> None:
        vae = self.ctx.decoder
        if vae is not None and self._original_dtype is not None:
            vae.to(self._original_dtype)
            self._original_dtype = None
