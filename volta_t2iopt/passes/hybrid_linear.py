"""S2 pass：把 bf16 的 nn.Linear 换成 fp16 张量核 + fp32 累加，带在线自适应 2 的幂预缩放。

## 为什么不是"按 profile 定死每层 k"

第一版按 profile 给每层定死 k，512px 一跑就炸：`noise_refiner.1.feed_forward.w2` 的输入
|max| 在 1024 下是 5824、在 512 下是 25472（4.4 倍），同一个 run 里后续 step 的量级还在涨
（step0 的 w2 输出 45312 → step1 直接 inf）。所以激活量级**依赖分辨率 / 步数 / 提示词**，
静态表天然不可靠。

## 做法：用权重的 ∞-范数给出输出上界，每层每次调用现场定 k

对任意一层有严格上界 `|out_ij| ≤ max_j|x_j| · max_i Σ_j|w_ij|`（∞-范数）。第二项在替换时
从权重算好（一次），第一项在 forward 里一次 `amax` 归约即可。于是

    s = 2^-k,  k = clamp(ceil(log2(amax_x · rowsum_w / OUT_LIMIT)), 0, 20)

保证 `|out · s| ≤ OUT_LIMIT < 65504`。这个上界比实际值松 1-2 个数量级，但放大缩写在数值上
是免费的（实测 k=0/4/8/10 的 relL2 都是 2.94e-4）：fp16 的指数范围远宽于激活动态范围。
代价只是每层多一次归约（~0.07 ms）。

全程 GPU 侧张量运算，不回 host、不产生同步；**也不改激活流的 dtype**（残差流仍是 bf16）。

## 为什么不直接 `model.half()`

attention 的输出投影在 fp16 下会溢出（注意力输出量级 ~400 × 3840 维累加 > 65504）。
这不是精度瑕疵而是范围不够 —— 必须先缩放再转 fp16，所以需要逐层、逐调用地定 k。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn as nn

from .base import Context, Pass, PatchResult, register

FP16_MAX = 65504.0
OUT_LIMIT = 50000.0   # 默认留 30% 余量，避免贴边
K_MAX = 20


def required_k(amax: float, rowsum: float, out_limit: float = OUT_LIMIT,
               k_max: int = K_MAX) -> int:
    """纯 Python 版：给定 |x|max 与权重 ∞-范数，算需要缩放 2^-k 的 k（供工具/测试用）。"""
    if amax <= 0 or rowsum <= 0:
        return 0
    need = amax * rowsum / out_limit
    if need <= 1.0:
        return 0
    k = 0
    while (2.0**k) < need:
        k += 1
    return min(k, k_max)


def prescale_k(x: torch.Tensor, rowsum: float, out_limit: float = OUT_LIMIT,
               k_max: int = K_MAX) -> torch.Tensor:
    """张量版：k 是 0 维 GPU 张量，**不产生 host 同步**。"""
    amax = x.abs().amax()
    k = torch.clamp(torch.ceil(torch.log2(amax * (rowsum / out_limit))), 0.0, float(k_max))
    return k


class HybridLinear(nn.Module):
    """`y = (x·s) @ W16^T · 2^k + b`：fp16 权重 × fp16 输入 → fp32 累加。

    参数
    ----
    weight, bias : 原始 nn.Linear 的权重/偏置
    out_limit    : 输出被限制在这个量级以内（要显著小于 65504）
    prescale     : False 表示只做 dtype 转换、不做 2 的幂预缩放（消融用）
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None,
                 out_limit: float = OUT_LIMIT, prescale: bool = True,
                 k_max: int = K_MAX) -> None:
        super().__init__()
        self.src_dtype = weight.dtype      # revert 时按它重建 nn.Linear
        amax_w = weight.abs().max().item()
        if amax_w >= FP16_MAX:
            raise ValueError(f"权重 |max|={amax_w:.3g} 超出 fp16 范围，这层不能用 fp16")
        self.register_buffer("w", weight.detach().to(torch.float16), persistent=False)
        self.register_buffer("bias", None if bias is None else bias.detach().clone(),
                             persistent=False)
        # 权重 ∞-范数：|out| ≤ max|x| · rowsum。静态，替换时一次算好。
        self.rowsum = weight.detach().abs().sum(dim=1).max().item()
        self.out_limit = out_limit
        self.prescale = prescale
        self.k_max = k_max

    # --- 兼容 nn.Linear 的外部读取（diffusers 里有读 .weight.dtype / .in_features 的地方）---
    @property
    def weight(self) -> torch.Tensor:
        return self.w

    @property
    def in_features(self) -> int:
        return int(self.w.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.w.shape[0])

    def extra_repr(self) -> str:
        return (f"in={self.w.shape[1]}, out={self.w.shape[0]}, rowsum={self.rowsum:.1f}, "
                f"prescale={self.prescale}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.bfloat16 and x.dtype != torch.float16:
            # 极少数调用会传 fp32（例如 timestep 相关的小 MLP），退回 fp32 保命
            return nn.functional.linear(x, self.w.float(), None if self.bias is None
                                        else self.bias.float())
        if not self.prescale:
            out = torch.matmul(x.to(torch.float16), self.w.T)
            out = out.to(torch.float32) + (self.bias if self.bias is not None else 0.0)
            return out.to(x.dtype)

        k = prescale_k(x, self.rowsum, self.out_limit, self.k_max)
        s = torch.exp2(-k)
        # **先乘再转**：x 自身可能超 fp16 上限，转完再乘就救不回来了
        xf = (x * s.to(x.dtype)).to(torch.float16)
        out = torch.matmul(xf, self.w.T)
        # fp32 里乘回 2^k（幂次精确），顺带给出 bf16 转换的安全区
        out = out.to(torch.float32) * torch.exp2(k)
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)


def replace_linears(root: nn.Module, *, prefix: str = "", out_limit: float = OUT_LIMIT,
                    prescale: bool = True, k_max: int = K_MAX, skip_unsafe: bool = True,
                    keep_originals: bool = False
                    ) -> tuple[list[tuple[nn.Module, str, nn.Module | None]], list[str]]:
    """原地替换，返回 (回滚表, 被跳过的层名)。回滚表每项是 (父模块, 子名, 原模块或 None)。

    **默认不保留原模块**：bf16 权重占 11.7 GiB，保留它做回滚会让峰值显存翻倍
    （32 GB 卡上 1024² 直接没余量）。`revert()` 这时会用 fp16 权重重建 nn.Linear，
    数值上等价到 bf16 的舍入精度。显存充裕时传 `keep_originals=True` 换精确回滚。
    """
    undo: list[tuple[nn.Module, str, nn.Module | None]] = []
    skipped: list[str] = []
    for full_name, mod in list(root.named_modules()):
        for child_name, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            short = f"{full_name}.{child_name}" if full_name else child_name
            try:
                new = HybridLinear(child.weight.data,
                                   child.bias.data if child.bias is not None else None,
                                   out_limit=out_limit, prescale=prescale, k_max=k_max)
            except ValueError:
                if not skip_unsafe:
                    raise
                skipped.append(f"{prefix}.{short}" if prefix else short)
                continue
            setattr(mod, child_name, new)
            undo.append((mod, child_name, child if keep_originals else None))
    return undo, skipped


@register
class HybridLinearPass(Pass):
    """把 denoiser 的线性层换成 fp16 张量核实现。"""

    name: ClassVar[str] = "hybrid_linear"

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self._undo: list[tuple[nn.Module, str, nn.Module]] = []
        self._matmul_flag: bool | None = None

    def applicable(self) -> tuple[bool, str]:
        cap = self.ctx.capability
        force = bool(self.ctx.config.pass_config(self.name).get("force", False))
        if self.ctx.model_dtype != torch.bfloat16:
            return False, f"模型原生精度是 {self.ctx.model_dtype}，这个 pass 只处理 bf16-native"
        if cap.bf16_tensorcore and not force:
            return False, (f"sm_{cap.sm} 有 bf16 张量核，bf16 GEMM 已在张量核上，"
                           f"迁移到 fp16 只多一次类型转换")
        if not cap.fp16_tensorcore:
            return False, f"sm_{cap.sm} 没有 fp16 张量核，本 pass 无收益"
        return True, ""

    def apply(self) -> PatchResult:
        cfg = self.ctx.config.pass_config(self.name)
        out_limit = float(cfg.get("out_limit", OUT_LIMIT))
        k_max = int(cfg.get("k_max", K_MAX))
        prescale = bool(cfg.get("prescale", True))
        skip_unsafe = bool(cfg.get("skip_unsafe", True))

        denoiser = self.ctx.denoiser
        before_gib = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0

        keep_originals = bool(cfg.get("keep_originals", False))
        undo, skipped = replace_linears(denoiser, out_limit=out_limit, prescale=prescale,
                                        k_max=k_max, skip_unsafe=skip_unsafe,
                                        keep_originals=keep_originals)
        self._undo = undo

        # fp16 split-k 的约减精度：关掉能走 fp32 累加（微基准 20.0 vs 19.9 TF，纯赚）
        if torch.cuda.is_available():
            self._matmul_flag = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            torch.cuda.empty_cache()

        rowsums = [m.rowsum for m in denoiser.modules() if isinstance(m, HybridLinear)]
        after_gib = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        res = PatchResult(
            pass_name=self.name, applied=True, modules=len(undo),
            details=dict(out_limit=out_limit, k_max=k_max, prescale=prescale,
                         weight_rowsum_min=round(min(rowsums), 1) if rowsums else None,
                         weight_rowsum_max=round(max(rowsums), 1) if rowsums else None,
                         weights_gib=round(after_gib, 2),
                         freed_gib=round(max(before_gib - after_gib, 0.0), 2)),
            notes=["预缩放在 GPU 上现算（一次 amax 归约），不回 host、不产生同步"],
        )
        if skipped:
            res.notes.append(f"跳过 {len(skipped)} 层（权重超出 fp16 范围）：{skipped[:3]}")
        if not prescale:
            res.notes.append("prescale=False：只做 dtype 转换，溢出风险由调用方承担（消融用）")
        return res

    def revert(self) -> None:
        """还原成 nn.Linear。没有保留原模块时用 fp16 权重重建（等价到 bf16 舍入精度）。"""
        for parent, name, original in self._undo:
            if original is not None:
                setattr(parent, name, original)
                continue
            cur = getattr(parent, name, None)
            if not isinstance(cur, HybridLinear):
                continue
            lin = nn.Linear(cur.w.shape[1], cur.w.shape[0], bias=cur.bias is not None)
            lin.weight.data = cur.w.detach().to(cur.src_dtype).clone()
            if cur.bias is not None:
                lin.bias.data = cur.bias.detach().clone()
            setattr(parent, name, lin)
        self._undo = []
        if self._matmul_flag is not None and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = self._matmul_flag
            self._matmul_flag = None
