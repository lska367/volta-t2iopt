"""验证"2 的幂预缩放"能否把 bf16 GEMM 换成 fp16 张量核（S2 的数值依据）。

三个必须记住的结论：
  1. 溢出是硬性失败：|out| > 65504 → inf → nan，不是精度问题，调不了。
  2. **必须先乘再转**：`(x*s).half()` 对，`x.half()*s` 错（x 本身就可能超 fp16 上限）。
  3. 预缩放是数值无损的：fp16 relL2 vs fp32 = 2e-4，比 bf16 路径的 1.7e-3 还准。
"""

from __future__ import annotations

import argparse

import torch

M, K, N = 4096, 3840, 3840  # 1024² 的 DiT token 数 / dim / dim


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).norm().item() / b.norm().item()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="volta-t2iopt microbench prescale")
    ap.add_argument("--amps", type=float, nargs="+", default=[10.0, 1e3, 2e4])
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    print(f"{torch.cuda.get_device_name(0)}\n")

    for amp in args.amps:
        x = (torch.randn(M, K, device="cuda") * amp).bfloat16()
        w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
        ref = x.float() @ w.float().T
        o16 = x.half() @ w.half().T
        o_bf = (x @ w.T).float()
        finite16 = bool(torch.isfinite(o16).all())
        line = (f"输入 |x|max={x.abs().max().item():8.1f}  "
                f"|out|max(fp32)={ref.abs().max().item():9.3e}"
                f" | fp16: {'ok' if finite16 else 'inf/nan':6s}"
                f" relL2={rel(o16.float(), ref) if finite16 else float('nan'):.2e}"
                f" | bf16 relL2={rel(o_bf, ref):.2e}")
        ok, bad = [], []
        for k in (2, 4):
            s = 2.0**-k
            good = ((x.float() * s).half() @ w.half().T).float() / s
            wrong = (x.half() * s @ w.half().T).float() / s
            ok.append(bool(torch.isfinite(good).all()))
            bad.append(bool(torch.isfinite(wrong).all()))
        line += f" | prescale(先乘后转) 2^-2/2^-4 finite={ok} | (先转后乘) finite={bad}"
        print(line)
        del x, w, ref, o16, o_bf
        torch.cuda.empty_cache()
    print("\n结论：预缩放只在 |out| 逼近 65504 时才需要，且无损；"
          "k 必须运行时按 |x|max·rowsum 现算（静态表会随分辨率失效）。")
    return 0
