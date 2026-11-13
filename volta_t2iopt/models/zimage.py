"""Z-Image-Turbo 适配器（6B DiT，bf16-native，Apache-2.0）。

几个只有踩过才知道的接口事实：

- `encode_prompt` **不接受** `num_images_per_prompt`，返回 `(list[Tensor], list[Tensor])`，
  文本条件是**多层特征列表**而不是单个 embedding；
- transformer 的输入 latent 是 5 维 `[B, C, F, H, W]`（F=帧数）；
- 9 步实际是 8–9 次 DiT forward，靠 `callback_on_step_end` 数，不要硬编码；
- `ZImagePipeline.__init__` 允许 text_encoder/tokenizer 为 None，只要调用时传 `prompt_embeds`。
  这是"跳过 7.5 GB TE 加载"的依据。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import torch

from .base import GenerationConditions, ModelAdapter, WeightCacheSpec


class ZImageAdapter(ModelAdapter):
    name: ClassVar[str] = "z-image-turbo"
    native_dtype: ClassVar[torch.dtype] = torch.bfloat16
    source_dirname: ClassVar[str] = "z-image-turbo"
    cache_dirname: ClassVar[str] = "z-image-turbo-bf16"
    default_size: ClassVar[int] = 1024
    default_steps: ClassVar[int] = 9
    default_guidance: ClassVar[float] = 0.0
    cache_spec: ClassVar[WeightCacheSpec] = WeightCacheSpec(
        components={
            "transformer": "diffusers.models.transformers:ZImageTransformer2DModel",
            "vae": "diffusers:AutoencoderKL",
        },
        copy_files=("model_index.json",),
        copy_dirs=("scheduler", "tokenizer", "text_encoder"),
    )

    def load(self, *, path: Any = None, model_dtype: torch.dtype = torch.bfloat16,
             decoder_dtype: torch.dtype | None = None, device: str = "cuda",
             with_text_encoder: bool = False, verbose: bool = True):
        import time

        from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, ZImagePipeline
        from diffusers.models.transformers import ZImageTransformer2DModel

        root = Path(path) if path is not None else self.default_path()
        decoder_dtype = decoder_dtype or model_dtype
        t0 = time.time()

        if with_text_encoder:
            pipe = ZImagePipeline.from_pretrained(root, dtype=model_dtype)
            pipe = pipe.to(device)
            pipe.set_progress_bar_config(disable=True)
            if verbose:
                print(f"[load] {root.name} 整条 pipeline（含 TE）{time.time() - t0:.0f} s", flush=True)
            return pipe

        transformer = ZImageTransformer2DModel.from_pretrained(root / "transformer", dtype=model_dtype)
        vae = AutoencoderKL.from_pretrained(root / "vae", dtype=decoder_dtype)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(root / "scheduler")
        pipe = ZImagePipeline(scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None,
                              transformer=transformer)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        if verbose:
            alloc = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
            print(f"[load] {root.name} transformer+VAE（跳过 7.5 GB 的 TE）"
                  f"{time.time() - t0:.1f} s | 显存 {alloc:.1f} GiB", flush=True)
        return pipe

    def denoiser(self, pipe) -> torch.nn.Module:
        return pipe.transformer

    def decoder(self, pipe) -> torch.nn.Module | None:
        return pipe.vae

    def encode(self, pipe, prompt: str, device: str = "cuda") -> GenerationConditions:
        pe = pipe.encode_prompt(prompt, device=device, do_classifier_free_guidance=False)
        feats = pe[0] if isinstance(pe, (tuple, list)) else pe
        assert isinstance(feats, (list, tuple)), f"预期 list[tensor]，拿到 {type(feats)}"
        return GenerationConditions(prompt=prompt, tensors=list(feats))

    def run(self, pipe, conditions: GenerationConditions, *, size: int, steps: int,
            seed: int = 0, output_type: str = "pil", callback=None) -> list:
        tensors = [t.to("cuda") if t.device.type == "cpu" else t for t in conditions.tensors]
        out = pipe(
            prompt=None,
            prompt_embeds=tensors,
            height=size,
            width=size,
            num_inference_steps=steps,
            guidance_scale=self.default_guidance,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_type=output_type,
            callback_on_step_end=callback,
            return_dict=False,
        )
        images = out[0] if isinstance(out, (tuple, list)) else out
        return list(images) if isinstance(images, (list, tuple)) else [images]
