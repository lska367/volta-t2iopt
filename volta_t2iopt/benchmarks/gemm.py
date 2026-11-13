"""DiT 真实形状下的 GEMM 微基准（不加载模型）。

回答三个问题：
  1. fp16（fp32 累加）在真实形状上到底多少 TFLOPS？
  2. bf16 是不是真的只有 ~2.4 TFLOPS（torch 在 sm_70 上没有 bf16 张量核）？
  3. fp32 SGEMM 多少？（若够快，可以作为零数值风险的 fallback）

同时记录 SM 时钟：这台机器有 HW Power Brake，算力可能被压。
"""

from __future__ import annotations

import argparse
import subprocess

import torch

# Z-Image transformer: dim=3840, ffn=10240, n_heads=30, tokens=64*64（1024² 输入）
SHAPES = [
    ("qkv    ", 4096, 3 * 3840, 3840),
    ("to_out ", 4096, 3840, 3840),
    ("ffn_in ", 4096, 2 * 10240, 3840),
    ("ffn_out", 4096, 3840, 10240),
]


def clocks() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
        return out or "n/a"
    except Exception:  # pragma: no cover
        return "n/a"


def bench(fn, reps: int) -> float:
    """返回毫秒（中位数）。"""
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
    return sorted(times)[len(times) // 2]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="volta-t2iopt microbench gemm")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)

    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    print(f"clock before: {clocks()}\n")
    print(f"{'shape':8s} {'M':>5s} {'N':>6s} {'K':>5s}  变体")

    rows: dict[str, list[float]] = {}
    for name, m, n, k in SHAPES:
        flops = 2 * m * n * k
        x32 = torch.randn(m, k, device="cuda", dtype=torch.float32)
        w32 = torch.randn(n, k, device="cuda", dtype=torch.float32) * 0.02
        for label, kind in (("fp16 +fp32acc", "fp16"), ("fp16 -fp32acc", "fp16nacc"),
                            ("bf16 (sm70 模拟)", "bf16"), ("fp32 sgemm", "fp32")):
            if kind in ("fp16", "fp16nacc"):
                x, w = x32.half(), w32.half()
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = kind == "fp16"
                fn = lambda x=x, w=w: x @ w.T  # noqa: E731
            elif kind == "bf16":
                x, w = x32.bfloat16(), w32.bfloat16()
                fn = lambda x=x, w=w: x @ w.T  # noqa: E731
            else:
                x, w = x32, w32
                fn = lambda x=x, w=w: x @ w.T  # noqa: E731
            ms = bench(fn, args.reps)
            tflops = flops / (ms * 1e-3) / 1e12
            rows.setdefault(label, []).append(tflops)
            print(f"{name:8s} {m:5d} {n:6d} {k:5d}  {label:18s} {tflops:6.2f} TF  ({ms:7.1f} ms)")
            del x, w
            torch.cuda.empty_cache()
        del x32, w32
        torch.cuda.empty_cache()
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    print(f"\nclock after: {clocks()}")
    print("\n汇总（各形状 TFLOPS，末列=相对 fp16 的中位倍数）")
    base = sorted(rows.get("fp16 +fp32acc", [1.0]))[0]
    for label, vals in rows.items():
        med = sorted(vals)[len(vals) // 2]
        print(f"  {label:18s} " + " ".join(f"{v:6.2f}" for v in vals) +
              f"  | 中位 {med:6.2f}  ×{med / base:5.2f}")
    return 0
