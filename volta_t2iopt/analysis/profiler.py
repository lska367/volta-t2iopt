"""一次 forward 的归因：kernel 分布 +（可选）逐层激活量级 + 干净墙钟。

三条口径（不按这个跑会得出错结论）：

1. **只算 kernel 事件**，不用 `key_averages()`（会把 `aten::*` 重复计入）。
2. **同一档位要同时记"挂 profiler"和"不挂 profiler"两个 forward 时间**。profiler 会伪造空转，
   S3 档曾显示 16.5% 空转（0.9 s），去掉 profiler 后 kernel 4.56 s ≈ 整 forward 4.58 s。
3. **逐层统计只能在未替换的模型上做**（HybridLinear 不是 nn.Linear）。

用法（CLI）：`volta-t2iopt profile --preset final --size 1024`
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from ..models import detect_adapter
from ..paths import results_root
from ..runtime.cache import prepare_conditions
from ..runtime.timing import spans_ms, timed_decode
from .layer_stats import LinearStats, format_layer_table
from .trace import kernel_stats

LINEAR_CLASSES = ("nn.Linear",)


def _has_hybrid(module: torch.nn.Module) -> bool:
    from ..passes.hybrid_linear import HybridLinear
    return any(isinstance(m, HybridLinear) for m in module.modules())


def profile_run(pipe, adapter=None, *, prompt: str = "portrait", size: int = 1024,
                steps: int = 2, seed: int = 0, out_dir: Path | str | None = None,
                tag: str = "", collect_layers: bool = True, verbose: bool = True) -> dict:
    """第一次 forward 挂 profiler，第二次 forward 记干净墙钟；返回可归档的 dict。"""
    import torch.profiler

    adapter = adapter or detect_adapter(pipe)
    out = Path(out_dir) if out_dir is not None else results_root()
    out.mkdir(parents=True, exist_ok=True)
    denoiser = adapter.denoiser(pipe)
    conditions = prepare_conditions(pipe, adapter, prompt)

    stats = LinearStats() if (collect_layers and not _has_hybrid(denoiser)) else None
    n_lin = stats.attach(denoiser, "denoiser") if stats else 0
    if verbose:
        if stats:
            print(f"[hook] {n_lin} 个 nn.Linear", flush=True)
        else:
            print("[hook] 跳过逐层统计（已替换为 HybridLinear 或已关闭）", flush=True)

    orig_forward = denoiser.forward
    state: dict[str, Any] = {"done": 0, "prof": None, "prof_ms": 0.0, "clean_ms": 0.0}

    def wrapped(*args, **kwargs):
        if state["done"] >= 2:
            return orig_forward(*args, **kwargs)
        state["done"] += 1
        if state["done"] == 1:
            m0, m1 = torch.cuda.Event(True), torch.cuda.Event(True)
            m0.record()
            with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA]) as prof:
                result = orig_forward(*args, **kwargs)
            m1.record()
            torch.cuda.synchronize()
            state["prof"], state["prof_ms"] = prof, m0.elapsed_time(m1)
            return result
        m0, m1 = torch.cuda.Event(True), torch.cuda.Event(True)
        m0.record()
        result = orig_forward(*args, **kwargs)
        m1.record()
        torch.cuda.synchronize()
        state["clean_ms"] = m0.elapsed_time(m1)
        return result

    denoiser.forward = wrapped
    torch.cuda.reset_peak_memory_stats()
    with timed_decode(pipe) as decode_spans:
        t0 = time.time()
        adapter.run(pipe, conditions, size=size, steps=steps, seed=seed, output_type="pil",
                    callback=None)
        wall_s = time.time() - t0
    torch.cuda.synchronize()
    denoiser.forward = orig_forward
    if stats:
        stats.enabled = False

    rep: dict = dict(model=adapter.name, preset=tag or "profile", size=size, steps=steps,
                     prompt=prompt, wall_s=round(wall_s, 1),
                     fwd_profiled_ms=round(state["prof_ms"], 1),
                     fwd_clean_ms=round(state["clean_ms"], 1),
                     vae_ms=round(spans_ms(decode_spans), 1),
                     peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2))

    if state["prof"] is not None:
        trace = out / f"trace_{tag or 'profile'}_{adapter.name}_{size}.json"
        state["prof"].export_chrome_trace(str(trace))
        rep["trace"] = str(trace)
        rep["kernels"] = kernel_stats(trace)
        if verbose:
            print(f"[fwd] profiler {rep['fwd_profiled_ms']:.0f} ms | 干净 "
                  f"{rep['fwd_clean_ms']:.0f} ms | VAE {rep['vae_ms']:.0f} ms | "
                  f"kernel {rep['kernels']['kernel_ms']:.0f} ms", flush=True)
            print(str(rep["kernels"]["cat_ms"]), flush=True)

    if stats and stats.rows:
        rep["linear"] = stats.rows
        if verbose:
            print(format_layer_table(stats.rows), flush=True)

    path = out / f"profile_{tag or 'default'}_{adapter.name}_{size}.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    rep["file"] = str(path)
    if verbose:
        print(f"-> {path}", flush=True)
    return rep
