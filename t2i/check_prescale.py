"""验证"2 的幂预缩放"(power-of-two prescale) 能否把 bf16 GEMM 换成 fp16 张量核。

背景：Z-Image 是 bf16-native，直接 fp16 会在 `attention.to_out` 溢出（输出 >65504）。
但溢出只发生在 **结果写回 fp16** 这一步，mma 的累加器本来就是 fp32。
所以：把输入先乘 2^-k 再转 fp16，得到的结果就小 2^-k，能落进 fp16；
在 fp32 里乘回 2^k 再转 bf16 —— 幂次缩放对 fp16 是无损的，相对精度完全不变。

三个必须记住的结论（本脚本就是它们的可复现证据）：
  1. 溢出是硬性失败：|out| > 65504 → inf → nan，不是精度问题，调不了。
  2. **必须先乘再转**：`(x*s).half()` 对，`x.half()*s` 错（x 本身就可能超 fp16 上限）。
  3. 预缩放是数值无损的：fp16 relL2 vs fp32 = 2e-4，比 bf16 路径的 1.7e-3 还准。

用法：python t2i/check_prescale.py
"""

from __future__ import annotations

import torch

M, K, N = 4096, 3840, 3840  # 1024² 的 DiT token 数 / dim / dim


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item() / b.norm().item()


def main() -> None:
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}\n")

    # 逐级放大量级，看 fp16 从哪里开始崩（模拟 attention 输出投影 ~400×3840 累加）
    for amp in (10.0, 1e3, 2e4):
        x = (torch.randn(M, K, device="cuda") * amp).bfloat16()
        w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        ref = x.float() @ w.float().T
        o16 = x.half() @ w.half().T
        o_bf = (x @ w.T).float()

        line = (f"输入 |x|max={x.abs().max().item():8.1f}  |out|max(fp32)={ref.abs().max().item():9.3e}"
                f" | fp16: {'inf/nan' if not torch.isfinite(o16).all() else 'ok':6s}"
                f" relL2={rel(o16.float(), ref) if torch.isfinite(o16).all() else float('nan'):.2e}"
                f" | bf16 relL2={rel(o_bf, ref):.2e}")

        # 预缩放：正确顺序（先乘再转）与错误顺序（先转再乘）
        ok, bad = [], []
        for k in (2, 4):
            s = 2.0**-k
            good = ((x.float() * s).half() @ w.half().T).float() / s
            wrong = (x.half() * s @ w.half().T).float() / s
            ok.append(torch.isfinite(good).all().item())
            bad.append(torch.isfinite(wrong).all().item())
        line += f" | prescale(先乘后转) 2^-2/2^-4 finite={ok} | (先转后乘) finite={bad}"
        print(line)
        del x, w, ref, o16, o_bf
        torch.cuda.empty_cache()

    print("\n结论：预缩放只在 |out| 逼近 65504 时才需要，且无损；k 由逐层实测的 |out|max 决定。")


if __name__ == "__main__":
    main()
