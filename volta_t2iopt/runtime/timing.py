"""分阶段计时（CUDA event，不含 host 抖动）。

两条必须记住的口径（都是 bug 换来的）：

1. **VAE 必须直接包住 `vae.decode` 计时，不能用"总数 − 每步间隔"估算**。
   `callback_on_step_end` 只覆盖 forward *之间*，最后一次 forward 会漏进 VAE 桶 ——
   早期就是因为这个把 bf16 的 VAE 记成 30.5 s（真实 5.26 s），还把"VAE 是第二大项"写进了结论。
2. **带 profiler 的墙钟不能当真实耗时**。同一次 run 里第 2 个 forward 不挂 profiler 才是干净数
   （S3 的 profile 显示空转 16.5%，去掉 profiler 后真实空转几乎为 0）。
   这层只负责"干净"的那一半，profiler 口径在 `analysis` 里。
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import torch


class Mark:
    """一个时间点：有 CUDA 就用 event，没有就退回 perf_counter（便于 CPU 侧测试）。"""

    __slots__ = ("_cuda", "_ev", "_t0")

    def __init__(self) -> None:
        self._cuda = torch.cuda.is_available()
        if self._cuda:
            self._ev = torch.cuda.Event(enable_timing=True)
        else:
            self._t0 = 0.0

    def record(self) -> "Mark":
        if self._cuda:
            self._ev.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def since(self, other: "Mark") -> float:
        """从 `other` 到这个 mark 的毫秒数。"""
        if self._cuda:
            return float(other._ev.elapsed_time(self._ev))
        return (self._t0 - other._t0) * 1e3


@dataclass
class StageTiming:
    total_ms: float = 0.0
    denoise_ms: float = 0.0
    vae_ms: float = 0.0
    te_ms: float = 0.0
    per_step_ms: list[float] = field(default_factory=list)
    forwards: int = 0
    peak_gib: float = 0.0

    @property
    def ms_per_fwd(self) -> float:
        return self.denoise_ms / max(self.forwards, 1)

    @property
    def vae_pct(self) -> float:
        return 100.0 * self.vae_ms / max(self.total_ms, 1e-9)

    def summary_line(self) -> str:
        return (f"总 {self.total_ms / 1000:6.2f} s | 去噪 {self.denoise_ms / 1000:6.2f} s "
                f"({self.ms_per_fwd:6.0f} ms/fwd, {self.forwards} 次) | "
                f"VAE {self.vae_ms / 1000:5.2f} s ({self.vae_pct:4.1f}%) | "
                f"峰值 {self.peak_gib:5.1f} GiB")

    def to_dict(self) -> dict:
        return dict(total_ms=round(self.total_ms, 1), denoise_ms=round(self.denoise_ms, 1),
                    vae_ms=round(self.vae_ms, 1), te_ms=round(self.te_ms, 1),
                    ms_per_fwd=round(self.ms_per_fwd, 1), forwards=self.forwards,
                    peak_gib=round(self.peak_gib, 2),
                    per_step_ms=[round(x, 1) for x in self.per_step_ms])


class StepRecorder:
    """兼容 `callback_on_step_end` 的每步打点器，可选抓最后一步的 latent。"""

    def __init__(self, capture_latents_at: int | None = None) -> None:
        self.marks: list[Mark] = []
        self.latents: torch.Tensor | None = None
        self.capture_at = capture_latents_at
        self.calls = 0

    def start(self) -> "StepRecorder":
        self.marks.append(Mark().record())
        return self

    def __call__(self, pipe=None, step: int = 0, timestep: Any = None, kwargs: dict | None = None):
        kwargs = kwargs if kwargs is not None else {}
        self.calls += 1
        self.marks.append(Mark().record())
        if self.capture_at is not None and step == self.capture_at:
            lat = kwargs.get("latents")
            if isinstance(lat, (list, tuple)):
                lat = lat[0]
            if torch.is_tensor(lat):
                self.latents = lat.detach().clone()
        return kwargs

    def intervals(self) -> list[float]:
        return [self.marks[i + 1].since(self.marks[i]) for i in range(len(self.marks) - 1)]


@contextmanager
def timed_decode(pipe) -> Iterator[list[tuple[Mark, Mark]]]:
    """包住 `pipe.vae.decode`，每次调用记一对 event。"""
    vae = getattr(pipe, "vae", None)
    if vae is None:
        yield []
        return
    orig = vae.decode
    spans: list[tuple[Mark, Mark]] = []

    def wrapped(*args, **kwargs):
        m0 = Mark().record()
        out = orig(*args, **kwargs)
        spans.append((m0, Mark().record()))
        return out

    vae.decode = wrapped
    try:
        yield spans
    finally:
        vae.decode = orig


def spans_ms(spans: list[tuple[Mark, Mark]]) -> float:
    return sum(m1.since(m0) for m0, m1 in spans)
