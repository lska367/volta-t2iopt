"""bench：加载 → 优化 → 计时出图（端到端）。"""

from __future__ import annotations

import argparse
import json

from ..paths import results_root
from ..runtime.generate import generate
from ..runtime.loader import load_optimized
from ._common import add_model_args


def add_parser(sub, name: str = "bench") -> None:
    p = sub.add_parser(name, help="端到端：加载 + 优化 + 计时出图")
    add_model_args(p)
    p.add_argument("--runs", type=int, default=2, help="第 2 次起是热缓存，取最后一次")
    p.add_argument("--latents", action="store_true", help="抓最后一步 latent（用于对拍）")
    p.add_argument("--no-save", action="store_true")


def run(args) -> int:
    pipe, sess = load_optimized(args.model, preset=args.preset, path=args.path,
                                measure=args.measure, verbose=not args.quiet)
    print(sess.summary())
    res = generate(pipe, sess.adapter, prompt=args.prompt, size=args.size, steps=args.steps,
                   seed=args.seed, runs=args.runs, capture_latents=args.latents,
                   out_dir=args.out, tag=args.tag or args.preset, save=not args.no_save,
                   verbose=not args.quiet,
                   extra_meta=dict(preset=args.preset, device=sess.capability.device_name,
                                   sm=sess.capability.sm,
                                   passes=[r.line() for r in sess.reports]))
    if not args.quiet:
        print(json.dumps({k: v for k, v in res.to_dict().items() if "per_step_ms" not in k},
                         ensure_ascii=False, indent=2))
    return 0
