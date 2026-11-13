"""S3：把 attention 的 q/k/v 转 fp16 再进 SDPA，拿到 sm_70 上的 cutlass fmha 内核。

## 为什么 bf16 慢

`dispatch_attention_fn` → `F.scaled_dot_product_attention`。V100(sm_70) 上
flash / cudnn 都不支持，而 **mem-efficient 的 cutlass fmha 内核只有 f16 版本**
（`fmha_cutlassF_f16_aligned_32x128_rf_sm70`）。所以 bf16 输入直接掉到 math 后端，
而且 math 后端会把 q/k/v 转成 fp32 走 SIMT GEMM（trace 里的 `volta_sgemm_64x64`）。

实测（(1,30,4160,128)，同一台 V100）：

| 输入 | 时间 | 走的 kernel |
|---|---|---|
| bf16 + SDPA | 137.5 ms | math → 3× direct_copy + `volta_sgemm` + `cunn_SoftMaxForwardReg` |
| **fp16 + SDPA** | **30.1 ms** | `fmha_cutlassF_f16_aligned_32x128_rf_sm70`（softmax 已融合） |
| fp16 手写 math | 42.8 ms | tensor core bmm + 独立 softmax |

## 为什么安全

Z-Image 的 attention 前有 qk-norm（`norm_q`/`norm_k` 是 RMSNorm），q/k 元素量级 ~6，
logits 上界 ~`sqrt(128)·6·6/√128` 量级 → O(10)；softmax 输出 ∈[0,1]；v 量级 ~400。
全在 fp16 范围内，而且比 bf16（8 位尾数）**更准**。

## 用法

    from t2i.fp16_attention import enable_fp16_attention
    enable_fp16_attention()      # 幂等；T2I_FP16_ATTN=0 可关

注意：`transformer_z_image.py` 在 import 时就 `from ..attention_dispatch import dispatch_attention_fn`
把函数对象绑进了自己的模块命名空间，所以只改 `attention_dispatch` 模块属性无效 ——
这里会扫一遍 `sys.modules`，把所有持有旧引用的模块都换掉。
"""

from __future__ import annotations

import os
import sys

import torch


def enable_fp16_attention(verbose: bool = True) -> bool:
    """把 SDPA 的输入转 fp16（仅在输入是 bf16 时）。返回是否新装上了。"""
    from diffusers.models import attention_dispatch

    orig = attention_dispatch.dispatch_attention_fn
    if getattr(orig, "_t2i_fp16", False):
        return False

    def patched(query, key, value, *args, **kwargs):
        if query.dtype == torch.bfloat16:
            out = orig(query.to(torch.float16), key.to(torch.float16), value.to(torch.float16),
                       *args, **kwargs)
            return out.to(query.dtype)
        return orig(query, key, value, *args, **kwargs)

    patched._t2i_fp16 = True
    patched._t2i_orig = orig

    n = 0
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "dispatch_attention_fn", None) is orig:
            setattr(mod, "dispatch_attention_fn", patched)
            n += 1
    attention_dispatch.dispatch_attention_fn = patched
    if verbose:
        print(f"[attn] SDPA 输入转 fp16（替换了 {n} 处引用）→ 期望走 cutlass fmha")
    return True


def disable_fp16_attention() -> None:
    """回滚到 bf16 SDPA（对比实验用）。"""
    from diffusers.models import attention_dispatch

    cur = attention_dispatch.dispatch_attention_fn
    orig = getattr(cur, "_t2i_orig", None)
    if orig is None:
        return
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "dispatch_attention_fn", None) is cur:
            setattr(mod, "dispatch_attention_fn", orig)
    attention_dispatch.dispatch_attention_fn = orig


def maybe_enable(verbose: bool = True) -> bool:
    if os.environ.get("T2I_FP16_ATTN", "1") == "0":
        if verbose:
            print("[attn] T2I_FP16_ATTN=0，保持 bf16 SDPA")
        return False
    return enable_fp16_attention(verbose)
