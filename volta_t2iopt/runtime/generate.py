"""一次生成 = 条件 → 计时 → 出图 → 落盘。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..models import detect_adapter, get_adapter
from ..models.base import GenerationConditions, ModelAdapter
from ..paths import results_root
from ..prompts import resolve as resolve_prompt
from .cache import prepare_conditions
from .timing import StageTiming, StepRecorder, spans_ms, timed_decode


@dataclass
class RunResult:
    image: Any = None
    latents: torch.Tensor | None = None
    timing: StageTiming = field(default_factory=StageTiming)
    meta: dict = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    stem: str = ""

    def to_dict(self) -> dict:
        out: dict = dict(stem=self.stem)
        out.update(self.meta)          # 两者都有 forwards，timing 的优先（口径一致）
        out.update(self.timing.to_dict())
        out["files"] = self.files
        return out


def generate(pipe, adapter: ModelAdapter | None = None, *, prompt: str = "portrait",
             conditions: GenerationConditions | None = None, size: int | None = None,
             steps: int | None = None, seed: int = 0, runs: int = 1,
             output_type: str = "pil", capture_latents: bool = False,
             out_dir: Path | str | None = None, tag: str = "", stem: str | None = None,
             use_cache: bool = True, save: bool = True, verbose: bool = True,
             extra_meta: dict | None = None) -> RunResult:
    """跑生成并计时。`runs > 1` 时取最后一次的结果，前面几次只用来预热/取稳定值。"""
    adapter = adapter or detect_adapter(pipe)
    size = int(size or adapter.default_size)
    steps = int(steps or adapter.default_steps)

    if conditions is None:
        conditions = prepare_conditions(pipe, adapter, resolve_prompt(prompt),
                                        use_cache=use_cache)

    result = RunResult()
    images = None
    for run_i in range(runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        recorder = StepRecorder(capture_latents_at=(steps - 1) if capture_latents else None)
        with timed_decode(pipe) as decode_spans:
            recorder.start()
            t_start = time.time()
            images = adapter.run(pipe, conditions, size=size, steps=steps, seed=seed,
                                 output_type=output_type, callback=recorder)
            wall_ms = (time.time() - t_start) * 1e3
        vae_ms = spans_ms(decode_spans)

        # 端到端用 event 量（不含 host 抖动）；拿不到就用墙钟
        try:
            total_ms = recorder.marks[-1].since(recorder.marks[0]) + vae_ms
        except Exception:
            total_ms = wall_ms
        per_step = recorder.intervals()
        timing = StageTiming(
            total_ms=total_ms, denoise_ms=max(total_ms - vae_ms, 0.0), vae_ms=vae_ms,
            per_step_ms=per_step, forwards=recorder.calls,
            peak_gib=(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
        )
        if verbose:
            print(f"[{adapter.name} {size}px/{steps}step run{run_i}] {timing.summary_line()}",
                  flush=True)
        result.timing, result.latents = timing, recorder.latents

    image = images[0] if images else None
    result.image = image
    result.meta = dict(model=adapter.name, size=size, steps=steps, seed=seed,
                       prompt=conditions.prompt, forwards=result.timing.forwards)
    if extra_meta:
        result.meta.update(extra_meta)
    if image is not None:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        result.meta.update(img_mean=round(float(arr.mean()), 4),
                           img_std=round(float(arr.std()), 4),
                           img_nan=bool(np.isnan(arr).any()))

    if save:
        out = Path(out_dir) if out_dir is not None else results_root()
        out.mkdir(parents=True, exist_ok=True)
        stem = stem or f"{adapter.name}_{size}_{steps}step{('_' + tag) if tag else ''}"
        result.stem = stem
        png = out / f"{stem}.png"
        js = out / f"{stem}.json"
        if image is not None:
            image.save(png)
            result.files["png"] = str(png)
        if result.latents is not None:
            pt = out / f"{stem}.latent.pt"
            torch.save(result.latents.float().cpu(), pt)
            result.files["latent"] = str(pt)
        js.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        result.files["json"] = str(js)
        if verbose:
            print(f"-> {png}" if image is not None else "-> (无图)", flush=True)
    return result
