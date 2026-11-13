"""适配器协议：pipeline 的"哪部分是 denoiser / 怎么出图"由模型自己说。

pass 层只关心三件事：denoiser 在哪、decoder 在哪、模型原生什么精度。
所以新模型接入 = 实现 5 个方法，不需要改任何 pass。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from ..paths import cache_root


@dataclass
class WeightCacheSpec:
    """离线权重缓存的规格（哪些组件要转精度另存、哪些小件直接拷）。"""

    components: dict[str, str] = field(default_factory=dict)   # 组件名 -> "module:Class"
    copy_files: tuple[str, ...] = ("model_index.json",)
    copy_dirs: tuple[str, ...] = ("scheduler", "tokenizer", "text_encoder")


@dataclass
class GenerationConditions:
    """文本条件。层数/形状归一化到 `list[torch.Tensor]`，方便缓存与对比。"""

    prompt: str = ""
    tensors: list[torch.Tensor] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)

    def to(self, device: str | torch.device, dtype: torch.dtype | None = None) -> "GenerationConditions":
        self.tensors = [t.to(device=device, dtype=dtype or t.dtype) for t in self.tensors]
        self.extras = {k: (v.to(device=device, dtype=dtype or v.dtype) if torch.is_tensor(v) else v)
                       for k, v in self.extras.items()}
        return self

    def clone_cpu(self) -> "GenerationConditions":
        return GenerationConditions(
            prompt=self.prompt,
            tensors=[t.detach().to("cpu").clone() for t in self.tensors],
            extras={k: (v.detach().to("cpu").clone() if torch.is_tensor(v) else v)
                    for k, v in self.extras.items()},
        )


class ModelAdapter(ABC):
    """模型适配器基类。"""

    name: ClassVar[str] = "generic"
    native_dtype: ClassVar[torch.dtype] = torch.bfloat16   # 模型原生/磁盘精度
    source_dirname: ClassVar[str] = ""        # 原始权重目录名（在 cache_root 下）
    cache_dirname: ClassVar[str] = ""         # 转换后的权重目录名
    default_size: ClassVar[int] = 1024
    default_steps: ClassVar[int] = 9
    default_guidance: ClassVar[float] = 0.0
    cache_spec: ClassVar[WeightCacheSpec | None] = None

    # --- 加载 ---
    @abstractmethod
    def load(self, *, path: Any = None, model_dtype: torch.dtype = torch.bfloat16,
             decoder_dtype: torch.dtype | None = None, device: str = "cuda",
             with_text_encoder: bool = False, verbose: bool = True):
        """返回 pipeline。`with_text_encoder=False` 时 TE 为 None，调用必须传预计算的条件。"""

    # --- 结构 ---
    @abstractmethod
    def denoiser(self, pipe) -> torch.nn.Module:
        """被 pass 优化的主干（DiT transformer 或 UNet）。"""

    @abstractmethod
    def decoder(self, pipe) -> torch.nn.Module | None:
        """VAE / decoder。"""

    # --- 条件与生成 ---
    @abstractmethod
    def encode(self, pipe, prompt: str, device: str = "cuda") -> GenerationConditions:
        """跑文本编码器（需要 TE 已加载）。"""

    @abstractmethod
    def run(self, pipe, conditions: GenerationConditions, *, size: int, steps: int,
            seed: int = 0, output_type: str = "pil", callback=None) -> list:
        """跑一次生成，返回图片列表。"""

    # --- 展示 ---
    def default_path(self, root=None) -> Any:
        return (root or cache_root()) / (self.cache_dirname or self.source_dirname)
