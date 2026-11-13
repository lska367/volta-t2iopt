"""report：把落盘的 JSON 汇总成表（终端或 markdown）。"""

from __future__ import annotations

import argparse

from ..analysis.report import bench_table, load_jsons, render_markdown_table, suite_table
from ..analysis.trace import analyze_trace, format_kernel_stats
from ..paths import results_root


def add_parser(sub, name: str = "report") -> None:
    p = sub.add_parser(name, help="汇总已落盘的结果")
    p.add_argument("--dir", default=None, help="结果目录（默认 results/）")
    p.add_argument("--what", choices=["auto", "bench", "profile", "trace"], default="auto")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--limit", type=int, default=20)


def run(args) -> int:
    d = args.dir or str(results_root())
    if args.what in ("auto", "bench"):
        rows = load_jsons(d, "val_*.json") + load_jsons(d, "*.json")
        print("== 端到端 ==")
        print(bench_table(rows))
    if args.what in ("auto", "profile"):
        rows = load_jsons(d, "profile_*.json")
        print("\n== profile 矩阵 ==")
        print(suite_table(rows))
        if args.markdown:
            print("\n" + render_markdown_table(rows, ["preset", "size", "fwd_profiled_ms",
                                                       "fwd_clean_ms", "vae_ms", "peak_gib"]))
    if args.what == "trace":
        traces = sorted(__import__("pathlib").Path(d).glob("trace_*.json*"))[: args.limit]
        for t in traces:
            print(f"\n== {t.name} ==")
            print(format_kernel_stats(analyze_trace(t)))
    return 0
