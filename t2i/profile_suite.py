"""批量 profile：4 个优化档 × 3 个分辨率，输出可归档的 JSON 快照。

为什么要"档位"这个维度：这台机器上每一步优化都把瓶颈换人了
（bf16 GEMM → attention 的 fp32 SIMT GEMM → 元素级），只看最终版会丢掉这条演化线。

口径（和 `analyze_trace.py` 一致，别用 `key_averages()`）：
  - 只统计 CUDA kernel 事件（`ph=X, cat=kernel`），排除 aten::*，否则父子事件会重复计；
  - **空转率是被 profiler 放大的**：同一次 run 里第 2 个 forward 不挂 profiler，
    两个时间都记下来用于对照（真实空转 ≈ 0）。

用法（每个组合一个进程，避免显存/allocator 状态互相污染）：
    python t2i/profile_suite.py --variant final --size 1024
    python t2i/profile_suite.py --report        # 把 out/suite_*.json 汇总成表
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t2i.analyze_trace import load_kernels, classify  # noqa: E402
from t2i.loader import load_embeds, load_pipe  # noqa: E402
from t2i.profile_zimage import LinearStats, recommend_k  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"

VARIANTS = {
    # 档位            hybrid  fp16-attn  vae dtype
    "bf16": dict(hybrid=False, attn=False, vae=torch.bfloat16),
    "s2": dict(hybrid=True, attn=False, vae=torch.bfloat16),
    "s3": dict(hybrid=True, attn=True, vae=torch.bfloat16),
    "final": dict(hybrid=True, attn=True, vae=torch.float16),
}


def kernel_stats(trace: Path) -> dict:
    ks = load_kernels(trace)
    total = sum(k["dur"] for k in ks)
    span = max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)
    cat: dict[str, float] = {}
    ncat: dict[str, int] = {}
    per: dict[str, list[int]] = {}
    for k in ks:
        c = classify(k["name"])
        cat[c] = cat.get(c, 0.0) + k["dur"]
        ncat[c] = ncat.get(c, 0) + 1
        slot = per.setdefault(k["name"], [0, 0])
        slot[0] += k["dur"]
        slot[1] += 1
    return dict(
        kernel_ms=round(total / 1e3, 1),
        span_ms=round(span / 1e3, 1),
        idle_pct=round((span - total) / span * 100, 1),
        launches=len(ks), distinct=len(per),
        cat_ms={c: round(v / 1e3, 1) for c, v in sorted(cat.items(), key=lambda kv: -kv[1])},
        cat_launches=ncat,
        top=[dict(name=n[:70], ms=round(d / 1e3, 1), calls=c, us=round(d / c, 1),
                  cat=classify(n)) for n, (d, c) in sorted(per.items(), key=lambda kv: -kv[1][0])[:12]],
    )


def run(args) -> None:
    cfg = VARIANTS[args.variant]
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.variant}_{args.size}"

    t0 = time.time()
    pipe = load_pipe(torch.bfloat16, vae_dtype=cfg["vae"], path=args.model)
    load_s = round(time.time() - t0, 1)

    if cfg["attn"]:
        from t2i.fp16_attention import maybe_enable
        maybe_enable()
    if cfg["hybrid"]:
        from t2i.hybrid_linear import patch_linear
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        patch_linear(pipe.transformer, "transformer")

    stats = LinearStats() if not cfg["hybrid"] else None
    n_lin = stats.attach("transformer", pipe.transformer) if stats else 0
    if stats:
        print(f"[hook] {n_lin} 个 nn.Linear")
    else:
        print("[hook] 跳过逐层统计（HybridLinear 不是 nn.Linear）")

    prompt_embeds = load_embeds(None, args.prompt)
    orig_fwd = pipe.transformer.forward
    seen: list[float] = [0.0, 0.0]

    def wrapped(*a, **kw):
        i = 0 if not seen[0] else 1
        if i == 1 and seen[0]:          # 第 2 次 forward：干净的墙钟，不挂 profiler
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            out = orig_fwd(*a, **kw)
            e.record()
            torch.cuda.synchronize()
            seen[1] = s.elapsed_time(e)
            return out
        seen[0] = 1.0
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = orig_fwd(*a, **kw)
        e.record()
        torch.cuda.synchronize()
        wrapped.prof = prof  # type: ignore[attr-defined]
        wrapped.fwd_ms = s.elapsed_time(e)  # type: ignore[attr-defined]
        return out

    pipe.transformer.forward = wrapped

    vae_t: list[float] = []
    orig_decode = pipe.vae.decode

    def timed_decode(*a, **kw):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = orig_decode(*a, **kw)
        e.record()
        torch.cuda.synchronize()
        vae_t.append(s.elapsed_time(e))
        return out

    pipe.vae.decode = timed_decode

    torch.cuda.reset_peak_memory_stats()
    tt = time.time()
    pipe(prompt=None, prompt_embeds=prompt_embeds, height=args.size, width=args.size,
         num_inference_steps=args.steps, guidance_scale=0.0,
         generator=torch.Generator("cuda").manual_seed(args.seed), output_type="pil",
         return_dict=False)
    torch.cuda.synchronize()
    wall = time.time() - tt
    pipe.transformer.forward = orig_fwd
    pipe.vae.decode = orig_decode

    prof = getattr(wrapped, "prof", None)
    rep: dict = dict(variant=args.variant, size=args.size, steps=args.steps, prompt=args.prompt,
                     load_s=load_s, vae_ms=round(vae_t[0], 1) if vae_t else None,
                     pipe_wall_s=round(wall, 1),
                     peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     fwd_profiled_ms=round(getattr(wrapped, "fwd_ms", 0), 1),
                     fwd_clean_ms=round(seen[1], 1))
    print(f"[{tag}] load {load_s} s | forward(profiler) {rep['fwd_profiled_ms']} ms | "
          f"forward(clean) {rep['fwd_clean_ms']} ms | VAE {rep['vae_ms']} ms | "
          f"峰值 {rep['peak_gib']} GiB")

    if prof is not None:
        trace = OUT / f"trace_{tag}.json"
        prof.export_chrome_trace(str(trace))
        rep["kernels"] = kernel_stats(trace)
        print(f"    kernel {rep['kernels']['kernel_ms']} ms / {rep['kernels']['launches']} 次发射 | "
              f"{rep['kernels']['cat_ms']}")

    if stats is not None and stats.rows:
        recommend_k(stats.rows)
        rep["linear"] = stats.rows

    path = OUT / f"suite_{tag}.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"-> {path}")


def report() -> None:
    files = sorted(OUT.glob("suite_*.json"))
    if not files:
        print("没有 suite_*.json")
        return
    rows = [json.loads(f.read_text()) for f in files]
    rows.sort(key=lambda r: (-r["size"], list(VARIANTS).index(r["variant"])))
    print(f"{'档位':6s} {'分辨率':>6s} {'fwd(prof)':>10s} {'fwd(净)':>8s} {'kernel':>8s} {'空转%':>6s} "
          f"{'发射':>6s} {'VAE':>7s} {'峰值GiB':>7s} | 分类 ms")
    for r in rows:
        k = r.get("kernels", {})
        print(f"{r['variant']:6s} {r['size']:6d} {r['fwd_profiled_ms']:10.1f} {r['fwd_clean_ms']:8.1f} "
              f"{k.get('kernel_ms', 0):8.1f} {k.get('idle_pct', 0):6.1f} {k.get('launches', 0):6d} "
              f"{r['vae_ms']:7.1f} {r['peak_gib']:7.2f} | {k.get('cat_ms', {})}")
    bak = [r for r in rows if r.get("linear")]
    if bak:
        print(f"\n逐层激活统计有 {len(bak)} 份（bf16 档）：")
        for r in bak:
            need = [x for x in r["linear"] if x["k"] > 0]
            worst = max(r["linear"], key=lambda x: x["out_max"])
            print(f"  {r['size']}px: {len(r['linear'])} 个 Linear，需缩放 {len(need)} 个；"
                  f"最坏 {worst['name']} |out|max={worst['out_max']:.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(VARIANTS))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=2, help="2 = 第1次挂 profiler、第2次测干净墙钟")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="portrait")
    ap.add_argument("--model", default="/home/lskam/models/z-image-turbo-bf16")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    if not args.variant:
        ap.error("需要 --variant 或 --report")
    if args.steps < 2:
        ap.error("--steps 至少 2（要对比 profiler 与干净墙钟）")
    run(args)


if __name__ == "__main__":
    main()
