"""优化会话：把"探测 → 选 pass → 应用 → 报告 / 回滚"收成一个对象。

    with volta_t2iopt.optimize(pipe) as sess:
        ...
        print(sess.summary())

会话是**唯一**知道 pass 顺序、失败降级、回滚的地方。pass 本身不互相调用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .capability import Capability, probe
from .config import OptimizerConfig
from .models import ModelAdapter, detect_adapter
from .passes import Context, get_pass
from .passes.base import PatchResult


@dataclass
class OptimizationSession:
    config: OptimizerConfig
    capability: Capability
    adapter: ModelAdapter
    pipe: Any = None
    device: str = "cuda"
    model_dtype: torch.dtype = torch.bfloat16
    reports: list[PatchResult] = field(default_factory=list)
    _passes: dict[str, Any] = field(default_factory=dict, repr=False)
    _applied: bool = False

    # --- 组装 ---
    @classmethod
    def from_pipe(cls, pipe, *, preset: str = "auto", config: OptimizerConfig | None = None,
                  adapter: ModelAdapter | None = None, capability: Capability | None = None,
                  model_dtype: torch.dtype | None = None, device: str = "cuda",
                  measure: bool = False) -> "OptimizationSession":
        adapter = adapter or detect_adapter(pipe)
        mdt = model_dtype or _denoiser_dtype(adapter.denoiser(pipe), adapter)
        cap = capability or probe(device=device, measure=measure)
        cfg = config or OptimizerConfig.from_preset(preset, capability=cap, model_dtype=mdt,
                                                    device=device)
        return cls(config=cfg, capability=cap, adapter=adapter, pipe=pipe, device=device,
                   model_dtype=mdt)

    # --- 执行 ---
    def apply(self) -> "OptimizationSession":
        if self._applied:
            return self
        ctx = Context(pipe=self.pipe, adapter=self.adapter, capability=self.capability,
                      config=self.config, device=self.device, model_dtype=self.model_dtype)
        for name in self.config.enabled_passes():
            cls = get_pass(name)
            inst = cls(ctx)
            ok, reason = inst.applicable()
            if not ok:
                self.reports.append(PatchResult(pass_name=name, applied=False, reason=reason))
                continue
            try:
                rep = inst.apply()
            except Exception as exc:  # 单点失败不影响其它 pass
                rep = PatchResult(pass_name=name, applied=False,
                                  reason=f"apply 抛异常，已跳过：{type(exc).__name__}: {exc}")
            self._passes[name] = inst
            self.reports.append(rep)
        self._applied = True
        return self

    def revert(self) -> None:
        for name in reversed(list(self._passes)):
            try:
                self._passes[name].revert()
            except Exception as exc:  # pragma: no cover
                print(f"[warn] revert {name} 失败：{exc}")
        self._passes.clear()
        self._applied = False

    # --- 查询 ---
    @property
    def applied(self) -> bool:
        return self._applied

    def summary(self) -> str:
        head = [f"volta-t2iopt session | {self.capability.device_name} (sm_{self.capability.sm})",
                self.config.describe()]
        if not self.reports:
            head.append("(尚未 apply)")
        for r in self.reports:
            head.append(str(r))
        return "\n".join(head)

    def to_dict(self) -> dict:
        return dict(device=self.capability.device_name, sm=self.capability.sm,
                    model=self.adapter.name, config=self.config.to_dict(),
                    passes=[r.to_dict() for r in self.reports],
                    capability=self.capability.summary())

    # --- 上下文管理 ---
    def __enter__(self) -> "OptimizationSession":
        return self.apply()

    def __exit__(self, *exc) -> None:
        self.revert()


def optimize(pipe, *, preset: str = "auto", config: OptimizerConfig | None = None,
             adapter: ModelAdapter | None = None, capability: Capability | None = None,
             model_dtype: torch.dtype | None = None, device: str = "cuda",
             measure: bool = False, apply: bool = True) -> OptimizationSession:
    """对已有 pipeline 应用优化（`apply=False` 只看计划不执行）。"""
    sess = OptimizationSession.from_pipe(pipe, preset=preset, config=config, adapter=adapter,
                                        capability=capability, model_dtype=model_dtype,
                                        device=device, measure=measure)
    return sess.apply() if apply else sess


def _denoiser_dtype(denoiser: torch.nn.Module, adapter: ModelAdapter) -> torch.dtype:
    for p in denoiser.parameters():
        return p.dtype
    return getattr(adapter, "native_dtype", torch.bfloat16)
