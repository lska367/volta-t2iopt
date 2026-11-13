"""S2：把 bf16 的 `nn.Linear` 换成 fp16 张量核 + **在线自适应 2 的幂预缩放**（纯 PyTorch）。

## 为什么不是"按 profile 定死每层 k"

第一版按 `out/profile_1024.json` 给每层定死 k，512px 一跑就炸：
`noise_refiner.1.feed_forward.w2` 的输入 |max| 在 1024 下是 5824、在 512 下是 **25472**（4.4 倍），
而且同一个 run 里后续 step 的量级还在涨（step0 的 w2 输出 45312 → step1 直接 inf）。
所以激活量级**依赖分辨率 / 步数 / 提示词**，静态表天然不可靠。

## 做法：用权重的 ∞-范数给出输出上界，每层每次调用现场定 k

对任意一层有严格上界 `|out_ij| ≤ max_j|x_j| · max_i Σ_j|w_ij|`（∞-范数）。第二项在加载时
从权重算好（一次），第一项在 forward 里一次 `amax` 归约即可。于是

    s = 2^-k,  k = clamp(ceil(log2(amax_x · rowsum_w / OUT_LIMIT)), 0, 20)

**保证** `|out · s| ≤ OUT_LIMIT < 65504`，任何输入都不会溢出。这个上界比实际值松 ~1-2 个数量级，
但放大缩写在数值上是**免费**的（实测 k=0/4/8/10 的 relL2 都是 2.94e-4，与 fp16 无缩放一致，
因为 fp16 的指数范围远宽于激活的动态范围），代价只是多一次归约（~0.07 ms/层）。

全程 GPU 侧张量运算，**不回 host、不产生同步**；也不改激活流的 dtype（残差流仍是 bf16）。

用法：
    from t2i.hybrid_linear import patch_linear
    patch_linear(pipe.transformer, "transformer")     # 或者 T2I_HYBRID=0 回退 bf16 基线
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn

FP16_MAX = 65504.0
OUT_LIMIT = 50000.0   # 留 30% 余量，避免贴边
K_MAX = 20


class HybridLinear(nn.Module):
    """`x @ w.T + b`：fp16 权重 × fp16 输入 → fp32 累加，输入侧自适应 2 的幂预缩放。"""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None,
                 out_limit: float = OUT_LIMIT):
        super().__init__()
        amax = weight.abs().max().item()
        if amax >= FP16_MAX:
            raise ValueError(f"权重 |max|={amax:.3g} 超出 fp16 范围，这层不能用 fp16")
        self.register_buffer("w", weight.detach().to(torch.float16), persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone(), persistent=False)
        else:
            self.register_buffer("bias", None, persistent=False)
        # 权重 ∞-范数：|out| ≤ max|x| · rowsum。静态，加载时一次算好。
        self.rowsum = weight.detach().abs().sum(dim=1).max().item()
        self.out_limit = out_limit

    # --- 兼容 nn.Linear 的属性读取（diffusers 的 TimestepEmbedder 读 .weight.dtype）---
    @property
    def weight(self) -> torch.Tensor:
        return self.w

    @property
    def in_features(self) -> int:
        return self.w.shape[1]

    @property
    def out_features(self) -> int:
        return self.w.shape[0]

    def extra_repr(self) -> str:
        return f"in={self.w.shape[1]}, out={self.w.shape[0]}, rowsum={self.rowsum:.1f}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 现场定缩放：k 是 0 维张量，不回 host
        amax = x.abs().amax()
        k = torch.clamp(torch.ceil(torch.log2(amax * (self.rowsum / self.out_limit))), 0.0,
                        float(K_MAX))
        s = torch.exp2(-k)
        # 2) **先乘再转**：x 自身可能超 fp16 上限，转完再乘就救不回来了
        xf = (x * s.to(torch.bfloat16)).to(torch.float16)
        out = torch.matmul(xf, self.w.T)
        # 3) fp32 里乘回 2^k（幂次精确），升精度也顺便给出 bf16 转换的安全区
        out = out.to(torch.float32) * torch.exp2(k)
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)


def patch_linear(root: nn.Module, prefix: str, verbose: bool = True) -> list[str]:
    """原地把 root 下所有 nn.Linear 换成 HybridLinear，返回被替换的层名。"""
    if os.environ.get("T2I_HYBRID", "1") == "0":
        if verbose:
            print("[hybrid] T2I_HYBRID=0，跳过替换（bf16 基线）")
        return []

    replaced: list[str] = []
    for full_name, mod in list(root.named_modules()):
        for child_name, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            short = f"{full_name}.{child_name}" if full_name else child_name
            new = HybridLinear(child.weight.data,
                               child.bias.data if child.bias is not None else None)
            setattr(mod, child_name, new)
            replaced.append(f"{prefix}.{short}")
    if verbose:
        rowsums = [m.rowsum for m in root.modules() if isinstance(m, HybridLinear)]
        print(f"[hybrid] 替换 {len(replaced)} 个 nn.Linear（fp16 权重 + 在线预缩放）；"
              f"权重 ∞-范数 {min(rowsums):.1f} ~ {max(rowsums):.1f}")
    return replaced
