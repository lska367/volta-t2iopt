"""pass 协议：appliable / apply / revert，外加一份可打印的报告。

设计约束（都是踩过坑换来的）：

- **幂等**：重复 apply 不能叠加两层 patch（旧脚本里第一次忘加标记就会重复包一层 forward）。
- **可回滚**：`revert()` 必须能把这个 pass 装上的东西全部卸掉，包括全局状态
  （例如 `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`）。
- **会自述**：每个 pass 报告"做了什么 / 跳过了什么 / 为什么"，
  这样同一条命令在不同卡上会产生不同的、可解释的日志，而不是一个静默的加速比。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from ..capability import Capability
from ..config import OptimizerConfig


@dataclass
class PatchResult:
    """一次 pass 的成果。modules 是主指标（替换/转换了多少个子模块）。"""

    pass_name: str
    applied: bool = False
    reason: str = ""
    modules: int = 0
    details: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "applied" if self.applied else "skipped"

    def line(self) -> str:
        head = f"[{self.pass_name}] {self.status}"
        if self.applied and self.modules:
            head += f" ({self.modules} modules)"
        if self.reason:
            head += f" | {self.reason}"
        return head

    def __str__(self) -> str:
        extra = "".join(f"\n    · {n}" for n in self.notes)
        return self.line() + extra

    def to_dict(self) -> dict:
        return dict(pass_name=self.pass_name, applied=self.applied, modules=self.modules,
                    reason=self.reason, details=self.details, notes=self.notes)


# 兼容旧名字（文档里出现过 PatchReport）
PatchReport = PatchResult


@dataclass
class Context:
    """一次优化会话的共享上下文：pipeline + 适配器 + 能力 + 配置。"""

    pipe: Any
    adapter: Any                      # models.ModelAdapter；这里不 import 以避免循环依赖
    capability: Capability
    config: OptimizerConfig
    device: str = "cuda"
    model_dtype: torch.dtype = torch.bfloat16
    state: dict[str, Any] = field(default_factory=dict)   # pass 之间传话用

    @property
    def denoiser(self) -> torch.nn.Module:
        return self.adapter.denoiser(self.pipe)

    @property
    def decoder(self) -> torch.nn.Module | None:
        return self.adapter.decoder(self.pipe)


class Pass(ABC):
    """一个可应用 / 可回滚的优化步骤。"""

    name: ClassVar[str] = ""
    requires: ClassVar[tuple[str, ...]] = ()

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.result: PatchResult | None = None

    # --- 生命周期 ---
    def applicable(self) -> tuple[bool, str]:
        """默认只看配置开关；具体 pass 可以追加前置条件。"""
        return True, ""

    @abstractmethod
    def apply(self) -> PatchResult:
        """执行。失败要自己 try/except 成 skipped 报告，不要抛出去污染整条流水线。"""

    def revert(self) -> None:
        """回滚。默认无事可做。"""

    # --- 状态 ---
    @property
    def applied(self) -> bool:
        return bool(self.result and self.result.applied)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} applied={self.applied}>"


_REGISTRY: dict[str, type[Pass]] = {}
_ORDER: list[str] = []


def register(cls: type[Pass]) -> type[Pass]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} 缺少 name")
    if cls.name in _REGISTRY:
        raise ValueError(f"pass 重名：{cls.name}")
    _REGISTRY[cls.name] = cls
    _ORDER.append(cls.name)
    return cls


def get_pass(name: str) -> type[Pass]:
    if name not in _REGISTRY:
        raise KeyError(f"未知 pass {name!r}，已注册：{sorted(_REGISTRY)}")
    return _REGISTRY[name]


def order_passes(names: list[str]) -> list[str]:
    """按注册顺序排序（保证 hybrid 先于 attn 之类的隐含依赖）。"""
    rank = {n: i for i, n in enumerate(_ORDER)}
    return sorted(names, key=lambda n: rank.get(n, len(rank)))
