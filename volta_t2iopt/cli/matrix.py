"""matrix：多档 × 多分辨率的 profile 矩阵。

**每个组合起一个子进程**——显存/allocator 状态会互相污染，同进程连跑会让后一档莫名变慢。
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from ..analysis.report import load_jsons, suite_table
from ..paths import results_root


def add_parser(sub, name: str = "matrix") -> None:
    p = sub.add_parser(name, help="多档 × 多分辨率的 profile 矩阵（每格一个子进程）")
    p.add_argument("--presets", nargs="+", default=["noop", "v100_sm70"])
    p.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--model", default="z-image-turbo")
    p.add_argument("--prompt", default="portrait")
    p.add_argument("--out", default=None)
    p.add_argument("--no-layers", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def run(args) -> int:
    out = args.out or str(results_root())
    jobs = [(preset, size) for size in args.sizes for preset in args.presets]
    for preset, size in jobs:
        cmd = [sys.executable, "-m", "volta_t2iopt.cli", "profile",
               "--model", args.model, "--preset", preset, "--size", str(size),
               "--steps", str(args.steps), "--prompt", args.prompt, "--out", out]
        if args.no_layers:
            cmd.append("--no-layers")
        print("+ " + " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, check=False)
    rows = load_jsons(out, "profile_*.json")
    print("\n" + suite_table(rows))
    return 0
