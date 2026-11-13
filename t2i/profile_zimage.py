"""S1：一次 forward 拿三张表，决定混合精度怎么落。

要回答的三个问题（`OPTIMIZATION_PLAN.md` §7 S1）：
  (a) kernel 级时间分布：GEMM / attention / 其余 各占多少 → 判 SDPA 是否掉到 math 后端
  (b) 逐层 Linear 的 |x|max 与 |out|max → 定每层 2 的幂预缩放的 k
  (c) 一次 forward 的 kernel 数 + CPU 侧发射时间 → CUDA Graph 的收益上限

关键点：**不要自己拼 transformer 的输入**（Z-Image 的 latent 是 5 维 [B,C,F,H,W]，
条件是多层特征 list[tensor]）。这里直接把 `pipe.transformer.forward` 包一层，
在第 1 次真实调用外面套 profiler —— 输入由 pipeline 自己造，口径不会错。

用法：
    python t2i/profile_zimage.py                 # 全套（约 1 分钟，不含加载）
    python t2i/profile_zimage.py --steps 2 --size 1024
    python t2i/profile_zimage.py --no-profile    # 只跑逐层统计，跳过 profiler
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t2i.loader import load_embeds, load_pipe  # noqa: E402
from t2i.prompts import BY_NAME  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
CACHE_ROOT = Path("/home/lskam/models")

# kernel 名 → 类别。规则先粗后细：先按 GEMM/attention 的强特征分，剩下的都算 other，
# 之后拿 top-15 的真实名字再回来补规则（见 --dump-raw）。
_GEMM = re.compile(r"gemm|xmma|cutlass|nvjet|volta_|ampere_|sm70|sm80|sgemm|hfma|wgrad|dgrad|884|16816|1688", re.I)
_ATTN = re.compile(r"flash|fmha|attention|scaled_dot|softmax|bmm", re.I)
_ELT = re.compile(r"elementwise|vectorized|unrolled|silu|gelu|mul|add|div|rms|layer_norm|layernorm|"
                  r"reduce|copy|cat|pad|slice|contiguous|fill|cast|to_|convert|index|gather|scatter|transpose", re.I)


def classify(name: str) -> str:
    if _GEMM.search(name):
        return "gemm"
    if _ATTN.search(name):
        return "attn"
    if _ELT.search(name):
        return "elt"
    return "other"


class LinearStats:
    """逐层记录 |x|max / |out|max，只存标量，内存无压力。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._pending: dict[int, tuple[str, torch.Tensor]] = {}
        self.enabled = True

    def _pre(self, name: str, mod: torch.nn.Linear, args: tuple) -> None:
        if not self.enabled or not args:
            return
        x = args[0]
        if not torch.is_tensor(x):
            return
        self._pending[id(mod)] = (name, x)

    def _post(self, name: str, mod: torch.nn.Linear, args: tuple, out: torch.Tensor) -> None:
        if not self.enabled or id(mod) not in self._pending:
            return
        _, x = self._pending.pop(id(mod))
        self.rows.append(dict(
            name=name, m=x.numel() // x.shape[-1], dim_in=x.shape[-1], dim_out=out.shape[-1],
            x_max=round(x.abs().max().item(), 4), out_max=round(out.abs().max().item(), 4),
            x_absmax_f16_ok=bool(x.abs().max().item() < 65504),
            out_absmax_f16_ok=bool(out.abs().max().item() < 65504),
        ))

    def attach(self, base: str, module: torch.nn.Module | None) -> int:
        if module is None:
            return 0
        n = 0
        for name, m in module.named_modules():
            if isinstance(m, torch.nn.Linear):
                full = f"{base}.{name}"
                m.register_forward_pre_hook(lambda mm, a, n=full: self._pre(n, mm, a))
                m.register_forward_hook(lambda mm, a, o, n=full: self._post(n, mm, a, o))
                n += 1
        return n


def recommend_k(stats: list[dict], safety: float = 50000.0) -> None:
    """给每层算 2^-k：让 |x|*2^-k 和 |out|*2^-k 都留在 fp16 安全区。

    注意两条约束：输入在**转 fp16 之前**乘（所以约束 |x|），输出写回 fp16 时（约束 |out|）。
    两者取大者，k 向上取整到 2 的幂。
    """
    for r in stats:
        need = max(r["x_max"], r["out_max"]) / safety
        k = 0
        while (2.0**k) < need:
            k += 1
        r["k"] = k
        r["scale"] = 2.0**-k


def analyze(prof: torch.profiler.profile, topk: int = 15) -> dict:
    """粗粒度的 top-N（只看真 kernel，排除 aten::*，否则会和子 kernel 重复计数）。

    更严格的统计（含 GPU 空转率）用 `t2i/analyze_trace.py` 解析 chrome trace。
    """
    evts = [e for e in prof.key_averages()
            if e.device_time_total > 0 and not e.key.startswith("aten::")] 
    total = sum(e.device_time_total for e in evts) or 1.0
    by_cat: dict[str, float] = {}
    for e in evts:
        by_cat[classify(e.key)] = by_cat.get(classify(e.key), 0.0) + e.device_time_total
    top = sorted(evts, key=lambda e: -e.device_time_total)[:topk]

    def cpu_t(e) -> float:
        return getattr(e, "self_host_time_total", None) or getattr(e, "self_cpu_time_total", 0.0)

    kernel_launches = sum(e.count for e in evts)
    return dict(
        device_total_ms=round(total / 1000, 2),
        cpu_total_ms=round(sum(cpu_t(e) for e in evts) / 1000, 2),
        kernel_launches=kernel_launches,
        distinct_kernels=len(evts),
        cat_ms={k: round(v / 1000, 2) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        cat_pct={k: round(100 * v / total, 1) for k, v in sorted(by_cat.items(), key=lambda kv: -kv[1])},
        top=[dict(name=e.key, ms=round(e.device_time_total / 1000, 2), calls=e.count,
                  pct=round(100 * e.device_time_total / total, 1), cat=classify(e.key)) for e in top],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(CACHE_ROOT / "z-image-turbo-bf16"))
    ap.add_argument("--embeds", default="")
    ap.add_argument("--prompt", default="portrait", choices=list(BY_NAME))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1, help="只要 1 次 forward 就够；>1 会多跑几步但不重复统计")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--hybrid", action="store_true", help="profile 前先换 HybridLinear（fp16 张量核）")
    ap.add_argument("--no-fp16-attn", action="store_true", help="不把 SDPA 输入转 fp16")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")

    import time
    t0 = time.time()
    pipe = load_pipe(torch.bfloat16, path=args.model)
    load_s = time.time() - t0
    torch.cuda.synchronize()

    if not args.no_fp16_attn:
        from t2i.fp16_attention import maybe_enable as maybe_fp16_attn
        maybe_fp16_attn()
    if args.hybrid:
        from t2i.hybrid_linear import patch_linear
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        patch_linear(pipe.transformer, "transformer")
        torch.cuda.synchronize()

    # --- 条件：用缓存，避免每次重跑 TE ---
    prompt_embeds = load_embeds(args.embeds or None, args.prompt)
    print(f"[cond] {args.prompt}: {len(prompt_embeds)} 层 {[tuple(t.shape) for t in prompt_embeds[:2]]}…")

    # --- (b) 逐层激活量级 ---
    stats = LinearStats()
    n_lin = stats.attach("transformer", pipe.transformer)
    n_tel = stats.attach("text_encoder", pipe.text_encoder)
    print(f"[hook] transformer {n_lin} 个 Linear，text_encoder {n_tel} 个（后者本次不跑）")

    # --- (a)(c) 用 profiler 包住第一次 forward ---
    state: dict[str, object] = {"done": bool(args.no_profile), "prof": None, "fwd_ms": None}
    orig_fwd = pipe.transformer.forward
    holder: dict[str, torch.Tensor] = {}

    def wrapped(*a, **kw):
        first = a[0][0] if a and isinstance(a[0], (list, tuple)) else (a[0] if a else None)
        holder["x"] = first.detach() if torch.is_tensor(first) else None
        if state["done"]:
            return orig_fwd(*a, **kw)
        state["done"] = True
        print("[prof] 第一次 transformer forward，开始采集…", flush=True)
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            out = orig_fwd(*a, **kw)
        e.record()
        torch.cuda.synchronize()
        state["prof"] = prof
        state["fwd_ms"] = s.elapsed_time(e)
        print(f"[prof] 该次 forward 墙钟 {state['fwd_ms']:.0f} ms", flush=True)
        return out

    pipe.transformer.forward = wrapped
    gen = torch.Generator("cuda").manual_seed(args.seed)
    t1 = time.time()
    img = pipe(prompt=None, prompt_embeds=prompt_embeds, height=args.size, width=args.size,
               num_inference_steps=args.steps, guidance_scale=0.0, generator=gen,
               output_type="pil", return_dict=False)[0][0]
    torch.cuda.synchronize()
    print(f"[run] {args.size}px / {args.steps} 步 总 {time.time()-t1:.1f} s | "
          f"峰值 {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    pipe.transformer.forward = orig_fwd
    stats.enabled = False

    img.save(OUT / f"profile_{args.size}{args.tag}.png")
    if holder.get("x") is not None:
        print(f"[input] transformer 输入(x) {tuple(holder['x'].shape)} dtype={holder['x'].dtype} "
              f"|max|={holder['x'].abs().max().item():.3f}", flush=True)

    # --- 汇总 ---
    recommend_k(stats.rows)
    bad = [r for r in stats.rows if not r["out_absmax_f16_ok"] or not r["x_absmax_f16_ok"]]
    need_k = [r for r in stats.rows if r["k"] > 0]
    print(f"\n[量级] {len(stats.rows)} 个 Linear；fp16 不安全的 {len(bad)} 个；需要预缩放的 {len(need_k)} 个")
    print(f"{'层':50s} {'M':>5s} {'K':>6s} {'N':>6s} {'|x|max':>10s} {'|out|max':>10s} {'k':>2s}")
    for r in sorted(stats.rows, key=lambda r: -max(r["x_max"], r["out_max"]))[:20]:
        print(f"{r['name'][:50]:50s} {r['m']:5d} {r['dim_in']:6d} {r['dim_out']:6d} "
              f"{r['x_max']:10.1f} {r['out_max']:10.1f} {r['k']:2d}")

    report: dict = dict(size=args.size, steps=args.steps, prompt=args.prompt, load_s=round(load_s, 1),
                        fwd_ms=None, linear=[], profile=None)
    if state["prof"] is not None:
        prof = state["prof"]
        rep = analyze(prof)
        report["fwd_ms"] = round(state["fwd_ms"], 1)
        report["profile"] = rep
        trace = OUT / f"trace_{args.size}{args.tag}.json"
        prof.export_chrome_trace(str(trace))
        print(f"\n[kernel] CUDA {rep['device_total_ms']:.0f} ms | CPU 发射 {rep['cpu_total_ms']:.0f} ms "
              f"| kernel 发射 {rep['kernel_launches']} 次 / {rep['distinct_kernels']} 种")
        print("[三分] " + "  ".join(f"{k} {v:.0f} ms ({rep['cat_pct'][k]}%)" for k, v in rep["cat_ms"].items()))
        print(f"\n{'top kernel':60s} {'ms':>9s} {'次数':>6s} {'占比':>6s}  类别")
        for t in rep["top"]:
            print(f"{t['name'][:60]:60s} {t['ms']:9.1f} {t['calls']:6d} {t['pct']:5.1f}%  {t['cat']}")
        print(f"\nchrome trace -> {trace}")

    report["linear"] = stats.rows
    if state["prof"] is None:
        print("\n[警告] 没有采到 profiler 数据")
    path = OUT / f"profile_{args.size}{args.tag}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {path}")


if __name__ == "__main__":
    main()
