"""S3 pass：把 attention 的 q/k/v 转 fp16 再进 SDPA，拿到 sm_70 上的 cutlass fmha 内核。

## 为什么 bf16 慢

`dispatch_attention_fn` → `F.scaled_dot_product_attention`。V100(sm_70) 上 flash/cudnn 都不支持，
而 **mem-efficient 的 cutlass fmha 内核只有 f16 版本**（`fmha_cutlassF_f16_aligned_32x128_rf_sm70`）。
bf16 输入直接 fallback 到 math 后端，它还会把 q/k/v 转 fp32 走 SIMT GEMM。

实测（(1,30,4160,128)，同一台 V100）：

| 输入 | 时间 | 走的 kernel |
|---|---|---|
| bf16 + SDPA | 137.5 ms | math → direct_copy + `volta_sgemm` + `cunn_SoftMaxForwardReg` |
| **fp16 + SDPA** | **30.1 ms** | `fmha_cutlassF_f16_..._sm70`（softmax 已融合） |

## 为什么安全

Z-Image 的 attention 前有 qk-norm（RMSNorm），q/k 元素量级 ~6，logits ~O(10)；softmax 输出 ∈[0,1]；
v 量级 ~400 —— 全在 fp16 范围内，而且比 bf16（8 位尾数）更准。

## 坑

`transformer_z_image.py` 在 import 时就 `from ..attention_dispatch import dispatch_attention_fn`
把函数对象绑进了自己的模块命名空间，所以只改 `attention_dispatch` 模块属性无效 ——
这里会扫一遍 `sys.modules`，把所有持有旧引用的模块都换掉，并在 revert 时精确还原。
apply 之后新 import 的模块拿不到 patch（本库在 pipeline 加载完成后应用，不受影响）。
"""

from __future__ import annotations

import sys
from typing import Any, ClassVar

import torch

from .base import Context, Pass, PatchResult, register


@register
class AttentionFp16Pass(Pass):
    name: ClassVar[str] = "attn_fp16"

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self._refs: list[tuple[Any, str, Any]] = []   # (模块对象, 属性名, 原函数)
        self._module = None

    def applicable(self) -> tuple[bool, str]:
        cap = self.ctx.capability
        force = bool(self.ctx.config.pass_config(self.name).get("force", False))
        if self.ctx.model_dtype != torch.bfloat16:
            return False, f"模型原生精度是 {self.ctx.model_dtype}，不需要 bf16→fp16 重定向"
        if cap.bf16_tensorcore and not force:
            return False, f"sm_{cap.sm} 上 bf16 有原生 fmha/flash 内核，无需重定向"
        return True, ""

    def apply(self) -> PatchResult:
        try:
            from diffusers.models import attention_dispatch
        except Exception as exc:  # pragma: no cover - 没有 diffusers
            return PatchResult(pass_name=self.name, applied=False,
                               reason=f"import diffusers 失败：{exc}")

        orig = attention_dispatch.dispatch_attention_fn
        if getattr(orig, "_vt2_attn_fp16", False):
            return PatchResult(pass_name=self.name, applied=False, reason="已经打过补丁（幂等跳过）")

        def patched(query, key, value, *args, **kwargs):
            if query.dtype == torch.bfloat16:
                out = orig(query.to(torch.float16), key.to(torch.float16),
                           value.to(torch.float16), *args, **kwargs)
                return out.to(query.dtype)
            return orig(query, key, value, *args, **kwargs)

        patched._vt2_attn_fp16 = True          # type: ignore[attr-defined]
        patched._vt2_orig = orig               # type: ignore[attr-defined]

        n = 0
        for name, mod in list(sys.modules.items()):
            if mod is not None and getattr(mod, "dispatch_attention_fn", None) is orig:
                self._refs.append((mod, "dispatch_attention_fn", orig))
                setattr(mod, "dispatch_attention_fn", patched)
                n += 1
        self._refs.append((attention_dispatch, "dispatch_attention_fn", orig))
        attention_dispatch.dispatch_attention_fn = patched
        self._module = attention_dispatch

        return PatchResult(
            pass_name=self.name, applied=True, modules=n,
            details=dict(references_rewritten=n + 1,
                         dispatch_module=getattr(attention_dispatch, "__name__", "?")),
            notes=[f"替换了 {n + 1} 处引用（含 diffusers 自己的模块属性）",
                   "sm_70 的 mem-efficient 后端只有 f16 版 cutlass fmha，"
                   "bf16 会 fallback 到 math 后端并转 fp32 走 SIMT GEMM"],
        )

    def revert(self) -> None:
        patched = getattr(self._module, "dispatch_attention_fn", None) if self._module else None
        for mod, attr, orig in self._refs:
            if getattr(mod, attr, None) is patched or getattr(mod, attr, None) is not orig:
                setattr(mod, attr, orig)
        # 兜底：apply 之后新 import 的模块也扫一遍
        if patched is not None:
            for name, mod in list(sys.modules.items()):
                if mod is not None and getattr(mod, "dispatch_attention_fn", None) is patched:
                    setattr(mod, "dispatch_attention_fn", getattr(patched, "_vt2_orig", None))
        self._refs = []
