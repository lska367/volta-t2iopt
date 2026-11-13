"""S2 的数值诊断：在**真实激活**上比较 bf16 路径 / fp16 自适应缩放路径 / fp32 参考。

为什么要这一步：端到端 latent 的 relL2 不能直接判断"是不是我的实现有问题"——
扩散采样会把每步的微小差异放大（尤其 Turbo 只有 9 步，误差不会被平均掉）。
真正的判据是**单层 GEMM 误差**：只要 fp16 路径不比 bf16 路径差，端到端的差异就是扩散本身的性质。

用法：
    python t2i/check_gemm_error.py --size 512 --steps 1
    python t2i/check_gemm_error.py --size 1024 --steps 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t2i.loader import load_embeds, load_pipe  # noqa: E402

WATCH = [  # (名字, 取模块的路径) —— 覆盖溢出层与普通层
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--prompt", default="portrait")
    args = ap.parse_args()

    pipe = load_pipe(torch.bfloat16)
    tr = pipe.transformer
    captured: dict[str, torch.Tensor] = {}

    def cap(name):
        def h(_m, a, _o):
            if name not in captured and a and torch.is_tensor(a[0]):
                captured[name] = a[0].detach()
            return _o
        return h

    for name in WATCH:
        get_module(tr, name).register_forward_hook(cap(name))

    pe = load_embeds(None, args.prompt)
    pipe(prompt=None, prompt_embeds=pe, height=args.size, width=args.size,
         num_inference_steps=args.steps, guidance_scale=0.0,
         generator=torch.Generator("cuda").manual_seed(0), output_type="latent", return_dict=False)

    print(f"\n{args.size}px：真实激活上的单层 GEMM 误差（参考 = fp32 权重×fp32 输入）")
    print(f"{'层':44s} {'K':>6s} {'|x|max':>10s} {'|out|max':>10s} {'rowsum':>8s} "
          f"{'k自适应':>7s} {'bf16 relL2':>11s} {'fp16 relL2':>11s}")
    for name in WATCH:
        if name not in captured:
            print(f"{name:44s} (没抓到)")
            continue
        x = captured[name]
        w = get_module(tr, name).weight.data
        ref = x.float() @ w.float().T
        bf = (x @ w.T).float()
        rowsum = w.float().abs().sum(dim=1).max().item()
        need = x.abs().max().item() * rowsum / 50000.0
        k = max(0, min(20, int(torch.ceil(torch.log2(torch.tensor(max(need, 1e-9)))).item())))
        s = float(2.0**-k)
        f16 = ((x * s).half() @ w.half().T).float() / s
        print(f"{name:44s} {w.shape[1]:6d} {x.abs().max().item():10.3e} {ref.abs().max().item():10.3e} "
              f"{rowsum:8.1f} {k:7d} {rel(bf, ref):11.2e} {rel(f16, ref):11.2e}")


if __name__ == "__main__":
    main()
