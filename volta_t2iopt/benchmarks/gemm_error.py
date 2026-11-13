"""在**真实激活**上比较 bf16 路径 / fp16 自适应缩放路径 / fp32 参考（S2 的数值诊断）。

为什么需要这一步：端到端 latent 的 relL2 不能判断"是不是实现错了" ——
扩散采样会把每步的微小差异放大（Turbo 只有 9 步，误差不会被平均掉）。
真正的判据是**单层 GEMM 误差**：只要 fp16 路径不比 bf16 路径差，端到端差异就是扩散本身的性质。
"""

from __future__ import annotations

import argparse

import torch

from ..passes.hybrid_linear import required_k
from ..runtime.cache import prepare_conditions
from ..runtime.loader import load_pipe

WATCH = [
    "layers.14.feed_forward.w2",
    "layers.8.feed_forward.w2",
    "layers.0.feed_forward.w1",
    "layers.0.attention.to_q",
    "layers.0.attention.to_out.0",
    "noise_refiner.0.attention.to_out.0",
    "noise_refiner.1.feed_forward.w2",
    "context_refiner.0.attention.to_out.0",
]


def get_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module:
    mod = root
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).float().norm().item() / b.float().norm().item()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="volta-t2iopt microbench gemm-error")
    ap.add_argument("--model", default="z-image-turbo")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--prompt", default="portrait")
    args = ap.parse_args(argv)

    pipe = load_pipe(args.model, model_dtype="bfloat16")
    from ..models import detect_adapter
    adapter = detect_adapter(pipe)
    denoiser = adapter.denoiser(pipe)
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name):
        def hook(_m, inputs, _o):
            if name not in captured and inputs and torch.is_tensor(inputs[0]):
                captured[name] = inputs[0].detach()
        return hook

    for name in WATCH:
        get_module(denoiser, name).register_forward_hook(make_hook(name))

    conditions = prepare_conditions(pipe, adapter, args.prompt)
    adapter.run(pipe, conditions, size=args.size, steps=args.steps, seed=0,
                output_type="latent", callback=None)

    print(f"\n{args.size}px：真实激活上的单层 GEMM 误差（参考 = fp32 权重 × fp32 输入）")
    print(f"{'层':44s} {'K':>6s} {'|x|max':>11s} {'|out|max':>11s} {'rowsum':>8s} "
          f"{'k':>2s} {'bf16 relL2':>11s} {'fp16 relL2':>11s}")
    for name in WATCH:
        if name not in captured:
            print(f"{name:44s} (没抓到)")
            continue
        x = captured[name]
        w = get_module(denoiser, name).weight.data
        ref = x.float() @ w.float().T
        bf = (x @ w.T).float()
        rowsum = w.float().abs().sum(dim=1).max().item()
        k = required_k(float(x.abs().max().item()), float(rowsum))
        s = 2.0**-k
        f16 = ((x * s).half() @ w.half().T).float() / s
        print(f"{name:44s} {w.shape[1]:6d} {x.abs().max().item():11.3e} "
              f"{ref.abs().max().item():11.3e} {rowsum:8.1f} {k:2d} "
              f"{rel(bf, ref):11.2e} {rel(f16, ref):11.2e}")
    return 0
