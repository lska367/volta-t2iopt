"""microbench：分项证据（gemm / prescale / attention / vae / gemm-error）。"""

from __future__ import annotations

import argparse

from ..benchmarks import BENCHMARKS, run as run_bench


def add_parser(sub, name: str = "microbench") -> None:
    p = sub.add_parser(name, help="分项微基准（每条结论的证据）")
    p.add_argument("what", choices=list(BENCHMARKS))
    p.add_argument("rest", nargs=argparse.REMAINDER, help="透传给具体微基准的参数")


def run(args) -> int:
    return run_bench(args.what, args.rest)
