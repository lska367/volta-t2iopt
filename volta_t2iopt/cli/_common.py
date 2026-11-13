"""各子命令共用的参数。"""

from __future__ import annotations

import argparse


def add_model_args(p: argparse.ArgumentParser, *, preset_default: str = "auto") -> None:
    p.add_argument("--model", default="z-image-turbo", help="z-image-turbo | sdxl-turbo | 权重目录")
    p.add_argument("--path", default=None, help="显式权重目录（默认按模型名到 cache_root 找）")
    p.add_argument("--preset", default=preset_default, help="auto | v100_sm70 | noop | *.toml")
    p.add_argument("--size", type=int, default=None, help="默认用适配器的 default_size")
    p.add_argument("--steps", type=int, default=None, help="默认用适配器的 default_steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--prompt", default="portrait")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=None, help="产物目录（默认 results/）")
    p.add_argument("--measure", action="store_true", help="能力探针额外跑 GEMM/SDPA 微基准")
    p.add_argument("--quiet", action="store_true")


def add_gate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--baseline", default="", help="baseline stem（如 results/val_z-image-turbo_noop_1024_9step）")
    p.add_argument("--fp32-ref", default="", help="fp32 真值 stem；给定时用它做质量判据")
    p.add_argument("--min-ssim", type=float, default=0.98)
    p.add_argument("--max-rel-l2", type=float, default=None,
                   help="latent relL2 门限；默认只报不判（扩散会放大差异）")
    p.add_argument("--max-latency-ratio", type=float, default=None,
                   help="candidate/base 的每 forward 时间上限（如 1.05）")
