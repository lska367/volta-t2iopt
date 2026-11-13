"""profile：一次 forward 的 kernel 归因 + 逐层激活量级。"""

from __future__ import annotations

import argparse

from ..analysis.profiler import profile_run
from ..runtime.loader import load_optimized
from ._common import add_model_args


def add_parser(sub, name: str = "profile") -> None:
    p = sub.add_parser(name, help="kernel 归因 / 逐层量级 / 干净墙钟")
    add_model_args(p)
    # 至少 2：第 1 次挂 profiler、第 2 次测干净墙钟
    p.set_defaults(steps=2)
    p.add_argument("--no-layers", action="store_true", help="跳过逐层统计")


def run(args) -> int:
    if (args.steps or 2) < 2:
        print("--steps 至少 2（要对比 profiler 与干净墙钟）")
        return 2
    pipe, sess = load_optimized(args.model, preset=args.preset, path=args.path,
                                measure=args.measure, verbose=not args.quiet)
    print(sess.summary())
    rep = profile_run(pipe, sess.adapter, prompt=args.prompt,
                      size=args.size or sess.adapter.default_size, steps=args.steps,
                      seed=args.seed, out_dir=args.out, tag=args.preset,
                      collect_layers=not args.no_layers, verbose=not args.quiet)
    print(f"\n结论：forward(profiler) {rep['fwd_profiled_ms']:.0f} ms → "
          f"干净 {rep['fwd_clean_ms']:.0f} ms；VAE {rep['vae_ms']:.0f} ms")
    return 0
