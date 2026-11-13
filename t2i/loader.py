"""统一的模型加载入口：**默认不加载文本编码器**。

为什么：Qwen3-4B 的 TE 在磁盘上 7.5 GB，而它只跑一次（0.33 s）且产物能缓存成 .pt
（`t2i/build_cache.py`）。每轮实验为它多读 7.5 GB 纯属浪费 —— 这台机器顺序读约 100 MB/s，
2 分钟的加载里有一分半是 TE。

`ZImagePipeline.__init__` 的 text_encoder / tokenizer 允许为 None，只要调用时传
`prompt_embeds` 而不是 `prompt`。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

CACHE_ROOT = Path(os.environ.get("ZIMAGE_CACHE_ROOT", "/home/lskam/models"))
DEFAULT_DIR = CACHE_ROOT / "z-image-turbo-bf16"


def load_pipe(dtype: torch.dtype = torch.bfloat16,
              vae_dtype: torch.dtype | None = None,
              path: str | Path | None = None,
              with_te: bool = False,
              device: str = "cuda",
              verbose: bool = True):
    """按组件加载 ZImagePipeline，跳过 TE，并把 transformer / VAE 放到目标精度与设备上。"""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
    from diffusers.models.transformers import ZImageTransformer2DModel

    root = Path(path or DEFAULT_DIR)
    vae_dtype = vae_dtype or dtype
    t0 = time.time()

    if with_te:
        pipe = ZImagePipeline.from_pretrained(root, dtype=dtype)
        if verbose:
            print(f"[load] 整条 pipeline（含 TE）{time.time()-t0:.0f} s", flush=True)
        return pipe.to(device)

    transformer = ZImageTransformer2DModel.from_pretrained(root / "transformer", dtype=dtype)
    vae = AutoencoderKL.from_pretrained(root / "vae", dtype=vae_dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
    pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                          transformer=transformer)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if verbose:
        print(f"[load] transformer+VAE（跳过 TE）{time.time()-t0:.1f} s | "
              f"显存 {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)
    return pipe


def load_embeds(path: str | Path | None = None, prompt: str = "portrait",
                device: str = "cuda") -> list[torch.Tensor]:
    """读缓存的文本条件（多层特征 list[tensor]）。"""
    p = Path(path or (CACHE_ROOT / "z-image-turbo-cache" / "prompt_embeds_bf16.pt"))
    blob = torch.load(p, map_location="cpu", weights_only=False)
    return [t.to(device) for t in blob["embeds"][prompt]]
