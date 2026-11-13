"""Z-Image-Turbo DiT 真实形状下的 GEMM 微基准（M0 的第 0 步，不用加载模型）。

回答三个问题：
  1. fp16（fp32 累加）在真实形状上到底多少 TFLOPS？
  2. fp32 SGEMM 多少？（若够快，就是一个零数值风险的 fallback）
  3. bf16 是不是真的只有 2.4 TFLOPS（torch 在 sm_70 上无张量核）？

同时记录 SM 时钟：这台机器有 HW Power Brake，算力可能被压。

用法：python t2i/microbench_gemm.py [--reps 5]
"""

from __future__ import annotations

import argparse
import subprocess

import torch

# Z-Image transformer: dim=3840, ffn=10240, n_heads=30, tokens=64*64（1024² 输入）
SHAPES = [
    ("qkv    ", 4096, 3 * 3840, 3840),
    ("to_out ", 4096, 3840, 3840),
    ("ffn_in ", 4096, 2 * 10240, 3840),   # w1/w3
    ("ffn_out", 4096, 3840, 10240),       # w2
]


def clocks() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out or "n/a"
    except Exception as e:  # pragma: no cover
        return f"n/a ({e})"


def bench(fn, reps: int) -> float:
    """返回毫秒（3 次取中位）。"""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"clock before: {clocks()}\n")
    print(f"{'shape':8s} {'M':>5s} {'N':>6s} {'K':>5s} {'TFLOPS':>19s}")

    variants = [
        ("fp16 +fp32acc", torch.float16, "half", True),
        ("fp16 -fp32acc", torch.float16, "half", False),
        ("fp16x1/16scale", torch.float16, "half_scaled", True),
        ("bf16 emul    ", torch.bfloat16, "bf16", None),
        ("fp32 sgemm   ", torch.float32, "fp32", None),
    ]

    rows = {}
    for name, m, n, k in SHAPES:
        flops = 2 * m * n * k
        for label, dtype, kind, acc in variants:
            if kind == "half_scaled":
                w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.02).half()
                x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 1.0).half()
                s = 1.0 / 16.0

                def fn(w=w, x=x, s=s):
                    return (x.mul(s) @ w.mul(s).T).float().mul(s * s * s)

            elif kind == "half":
                w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.02).half()
                x = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 1.0).half()
                fn = lambda w=w, x=x: x @ w.T  # noqa: E731
            elif kind == "bf16":
                w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.02).bfloat16()
                x = torch.randn(m, k, device="cuda", dtype=torch.float32).bfloat16()
                fn = lambda w=w, x=x: x @ w.T  # noqa: E731
            else:
                w = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.02)
                x = torch.randn(m, k, device="cuda", dtype=torch.float32)
                fn = lambda w=w, x=x: x @ w.T  # noqa: E731

            if acc is not None:
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = acc
            ms = bench(fn, args.reps)
            tflops = flops / (ms * 1e-3) / 1e12
            rows.setdefault(label.strip(), []).append(tflops)
            print(f"{name:8s} {m:5d} {n:6d} {k:5d}  {label} {tflops:6.2f} TF"[:72] +
                  f"  ({ms:7.1f} ms)")
            del w, x
            torch.cuda.empty_cache()

    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    print(f"\nclock after: {clocks()}")
    print("\n汇总（各形状 TFLOPS，末列=相对 fp16 的倍数）")
    base = rows["fp16 +fp32acc"]
    for label, vals in rows.items():
        print(f"  {label:15s} " + " ".join(f"{v:6.2f}" for v in vals) +
              f"  | 中位 {sorted(vals)[len(vals)//2]:6.2f}  ×{sorted(vals)[len(vals)//2]/sorted(base)[len(base)//2]:5.2f}")


if __name__ == "__main__":
    main()
