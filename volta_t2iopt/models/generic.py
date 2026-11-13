"""兜底适配器：任何 `vae + (transformer|unet)` 结构的 diffusers pipeline。

它只保证"能跑、能被 pass 优化"，不保证输出语义与官方 pipeline 一致
（不同模型的文本条件格式差异太大）。接入新模型时优先写一个专用适配器。
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch

from .base import GenerationConditions, ModelAdapter


class GenericAdapter(ModelAdapter):
    name: ClassVar[str] = "generic"
    native_dtype: ClassVar[torch.dtype] = torch.bfloat16

    def load(self, *, path: Any = None, model_dtype: torch.dtype = torch.bfloat16,
             decoder_dtype: torch.dtype | None = None, device: str = "cuda",
             with_text_encoder: bool = True, verbose: bool = True):
        from diffusers import DiffusionPipeline

        if path is None:
            raise ValueError("GenericAdapter 需要显式传 path")
        pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=model_dtype).to(device)
        pipe.set_progress_bar_config(disable=True)
        if decoder_dtype is not None and self.decoder(pipe) is not None:
            self.decoder(pipe).to(decoder_dtype)
        return pipe

    def denoiser(self, pipe) -> torch.nn.Module:
        return getattr(pipe, "transformer", None) or pipe.unet

    def decoder(self, pipe) -> torch.nn.Module | None:
        return getattr(pipe, "vae", None)

    def encode(self, pipe, prompt: str, device: str = "cuda") -> GenerationConditions:
        out = pipe.encode_prompt(prompt, device=device, num_images_per_prompt=1,
                                 do_classifier_free_guidance=False)
        out = out if isinstance(out, (tuple, list)) else (out,)
        tensors = [t for t in out if torch.is_tensor(t)]
        return GenerationConditions(prompt=prompt, tensors=tensors)

    def run(self, pipe, conditions: GenerationConditions, *, size: int, steps: int,
            seed: int = 0, output_type: str = "pil", callback=None) -> list:
        kwargs: dict[str, Any] = dict(
            height=size, width=size, num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            output_type=output_type, callback_on_step_end=callback, return_dict=False,
        )
        if conditions.tensors:
            kwargs["prompt_embeds"] = conditions.tensors[0].to("cuda")
        out = pipe(**kwargs)
        images = out[0] if isinstance(out, (tuple, list)) else out
        return list(images) if isinstance(images, (list, tuple)) else [images]
