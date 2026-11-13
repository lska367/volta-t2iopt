"""加载入口：精度在**加载时**就定好，避免"先按 bf16 读进来再全量转 fp16"的浪费。"""

from __future__ import annotations

from typing import Any

import torch

from ..config import OptimizerConfig, dtype_from_str
from ..models import ModelAdapter, get_adapter


def _native_dtype(adapter: ModelAdapter) -> torch.dtype:
    return getattr(adapter, "native_dtype", torch.bfloat16)


def load_pipe(model: str | ModelAdapter = "z-image-turbo", *, path: Any = None,
              model_dtype: torch.dtype | str | None = None,
              decoder_dtype: torch.dtype | str | None = None,
              preset: str | None = None, device: str = "cuda",
              with_text_encoder: bool = False, verbose: bool = True):
    """按模型名/路径加载 pipeline；`preset` 决定 decoder 精度（省一次全量类型转换）。"""
    adapter = get_adapter(model)
    cfg: OptimizerConfig | None = None
    if preset is not None:
        cfg = OptimizerConfig.from_preset(preset, model_dtype=model_dtype, device=device)

    mdt = (dtype_from_str(model_dtype) if model_dtype is not None
           else (cfg.model_dtype if cfg else _native_dtype(adapter)))
    ddt = (dtype_from_str(decoder_dtype) if decoder_dtype is not None
           else (cfg.effective_decoder_dtype() if cfg else mdt))
    return adapter.load(path=path, model_dtype=mdt, decoder_dtype=ddt, device=device,
                        with_text_encoder=with_text_encoder, verbose=verbose)


def load_optimized(model: str | ModelAdapter = "z-image-turbo", *, preset: str = "auto",
                   path: Any = None, model_dtype: torch.dtype | str | None = None,
                   device: str = "cuda", with_text_encoder: bool = False,
                   measure: bool = False, verbose: bool = True):
    """加载 + 探测 + 打补丁，一步到位。返回 `(pipe, session)`；session 可 revert。"""
    from ..session import optimize

    adapter = get_adapter(model)
    cfg = OptimizerConfig.from_preset(preset, model_dtype=model_dtype or _native_dtype(adapter),
                                      device=device)
    pipe = adapter.load(path=path, model_dtype=cfg.model_dtype,
                        decoder_dtype=cfg.effective_decoder_dtype(), device=device,
                        with_text_encoder=with_text_encoder, verbose=verbose)
    session = optimize(pipe, config=cfg, adapter=adapter, measure=measure)
    return pipe, session
