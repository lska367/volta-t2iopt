"""分项微基准：每条结论的"可复现证据"。

这些不是优化本身，而是**为什么这么优化**的证据：

- `gemm`      : bf16 在 sm_70 上没有张量核（2.4 vs 20 TFLOPS）—— 整条优化线的起点
- `prescale`  : 2 的幂预缩放无损、且必须"先乘再转"
- `attention` : bf16 SDPA 掉 math 后端 vs fp16 落 cutlass fmha（137.5 → 30.1 ms）
- `vae`      : VAE 直接 fp16 就够（5.26 s → 1.02 s，SSIM 0.99966）
- `gemm-error`: 真实激活上的单层误差，用来证明"端到端差异是扩散放大的，不是实现错了"
"""

from __future__ import annotations

BENCHMARKS = ("gemm", "prescale", "attention", "vae", "gemm-error")


def run(name: str, argv: list[str] | None = None) -> int:
    name = name.replace("_", "-")
    if name == "gemm":
        from . import gemm
        return gemm.main(argv)
    if name == "prescale":
        from . import prescale
        return prescale.main(argv)
    if name == "attention":
        from . import attention
        return attention.main(argv)
    if name == "vae":
        from . import vae
        return vae.main(argv)
    if name == "gemm-error":
        from . import gemm_error
        return gemm_error.main(argv)
    raise SystemExit(f"未知微基准 {name!r}，可用 {BENCHMARKS}")


__all__ = ["BENCHMARKS", "run"]
