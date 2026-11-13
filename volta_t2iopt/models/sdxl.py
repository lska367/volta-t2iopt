"""SDXL-turbo 适配器（fp16-native UNet）—— 对照模型。

它在 V100 上是**已经最优**的形态：fp16-native，什么都不用迁移。
把它接进框架有两个目的：① 证明 pass 层不会误伤（auto 会全关）；
② 给"现代 bf16-native DiT vs 老一代 fp16-native"提供同机对照（227 s vs 6.4 s）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import time

import torch

from ..config import dtype_name
from .base import GenerationConditions, ModelAdapter


class SDXLAdapter(ModelAdapter):
    name: ClassVar[str] = "sdxl-turbo"
    native_dtype: ClassVar[torch.dtype] = torch.float16
    source_dirname: ClassVar[str] = "sdxl-turbo"
    cache_dirname: ClassVar[str] = ""          # 原生 fp16，不需要权重缓存
    default_size: ClassVar[int] = 1024
    default_steps: ClassVar[int] = 4
    default_guidance: ClassVar[float] = 0.0

    def load(self, *, path: Any = None, model_dtype: torch.dtype = torch.float16,
             decoder_dtype: torch.dtype | None = None, device: str = "cuda",
             with_text_encoder: bool = True, verbose: bool = True):
        from diffusers import StableDiffusionXLPipeline

        root = Path(path) if path is not None else self.default_path()
        t0 = time.time()
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                root, torch_dtype=model_dtype, variant="fp16", use_safetensors=True)
        except Exception:
            # 目录里没有 fp16 variant 时退回默认权重
            pipe = StableDiffusionXLPipeline.from_pretrained(root, torch_dtype=model_dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        if decoder_dtype is not None:
            pipe.vae.to(decoder_dtype)
        if verbose:
            alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print(f"[load] {root.name} {dtype_name(model_dtype)} {time.time() - t0:.0f} s | "
                  f"显存 {alloc:.1f} GiB", flush=True)
        return pipe

    def denoiser(self, pipe) -> torch.nn.Module:
        return pipe.unet

    def decoder(self, pipe) -> torch.nn.Module | None:
        return pipe.vae

    def encode(self, pipe, prompt: str, device: str = "cuda") -> GenerationConditions:
        out = pipe.encode_prompt(prompt, prompt_2=None, device=device,
                                 num_images_per_prompt=1,
                                 do_classifier_free_guidance=False)
        # diffusers 不同版本返回 2~4 个张量：prompt_embeds, [neg], [pooled], [neg_pooled]
        names = ["prompt_embeds", "negative_prompt_embeds", "pooled_prompt_embeds",
                 "negative_pooled_prompt_embeds"]
        tensors, extras = [], {}
        for i, t in enumerate(out):
            if not torch.is_tensor(t):
                continue
            if i < 2:
                tensors.append(t)
            else:
                extras[names[i] if i < len(names) else f"extra_{i}"] = t
        return GenerationConditions(prompt=prompt, tensors=tensors, extras=extras)

    def run(self, pipe, conditions: GenerationConditions, *, size: int, steps: int,
            seed: int = 0, output_type: str = "pil", callback=None) -> list:
        kwargs: dict[str, Any] = dict(
            height=size, width=size, num_inference_steps=steps,
            guidance_scale=self.default_guidance,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_type=output_type, callback_on_step_end=callback, return_dict=False,
        )
        kwargs["prompt_embeds"] = conditions.tensors[0].to("cuda")
        if len(conditions.tensors) > 1:
            kwargs["negative_prompt_embeds"] = conditions.tensors[1].to("cuda")
        for k, v in conditions.extras.items():
            if torch.is_tensor(v):
                kwargs[k] = v.to("cuda")
        try:
            out = pipe(**kwargs)
        except TypeError:
            # 老版本不接受 negative_prompt_embeds 与 pooled 一起传，退回只传正向条件
            for k in ("negative_prompt_embeds", "negative_pooled_prompt_embeds"):
                kwargs.pop(k, None)
            out = pipe(**kwargs)
        images = out[0] if isinstance(out, (tuple, list)) else out
        return list(images) if isinstance(images, (list, tuple)) else [images]
