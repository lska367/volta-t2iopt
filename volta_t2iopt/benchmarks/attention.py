"""S3 的证据：同一形状下 bf16 / fp16 的 SDPA 各走哪个 kernel、差多少。

sm_70 上 mem-efficient 的 cutlass fmha 只有 f16 版本，所以 bf16 会 fallback 到 math 后端，
而 math 后端会把 q/k/v 转成 fp32 走 SIMT GEMM（`volta_sgemm`）—— 实测 137.5 ms vs 30.1 ms。
"""

from __future__ import annotations

import argparse

import torch


def bench(fn, reps: int = 3) -> float:
    for _ in range(1):
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


def kernels_used(dtype: torch.dtype, shape, reps: int = 1) -> list[tuple[str, float]]:
    q, k, v = (torch.randn(*shape, device="cuda", dtype=dtype) for _ in range(3))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    rows = [(e.key[:64], round(e.device_time_total / 1000, 1)) for e in prof.key_averages()
            if e.device_time_total > 0]
    return sorted(rows, key=lambda r: -r[1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="volta-t2iopt microbench attention")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=30)
    ap.add_argument("--tokens", type=int, default=4160)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args(argv)

    shape = (args.batch, args.heads, args.tokens, args.head_dim)
    print(f"{torch.cuda.get_device_name(0)} | shape {shape}\n")
    results = {}
    for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
        q, k, v = (torch.randn(*shape, device="cuda", dtype=dt) for _ in range(3))
        ms = bench(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), args.reps)
        results[name] = ms
        print(f"[{name}] {ms:8.1f} ms")
        for kname, kms in kernels_used(dt, shape, reps=1)[:5]:
            print(f"        {kms:8.1f} ms  {kname}")
        del q, k, v
        torch.cuda.empty_cache()
    print(f"\nfp16 / bf16 = {results['bf16'] / results['fp16']:.2f}× （fp16 快这么多）")
    return 0
