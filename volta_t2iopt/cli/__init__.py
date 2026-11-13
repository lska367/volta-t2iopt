"""命令行入口：`volta-t2iopt <subcommand>`。

设计原则：CLI 只做参数解析 + 打印，所有逻辑在库里。
这样同一件事既能 `from volta_t2iopt import generate` 编程调用，也能在 shell 里复现。
"""

from __future__ import annotations

import argparse

from . import bench, cache, matrix, microbench, probe, profile, report, validate

COMMANDS = {
    "probe": probe,
    "bench": bench,
    "validate": validate,
    "profile": profile,
    "matrix": matrix,
    "report": report,
    "cache": cache,
    "microbench": microbench,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="volta-t2iopt",
        description="在 pre-Ampere GPU 上把 bf16-native 扩散模型迁到 fp16 张量核",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  volta-t2iopt probe --measure                     # 这台卡到底缺什么
  volta-t2iopt bench --preset auto --size 1024      # 端到端计时出图
  volta-t2iopt bench --preset noop --size 1024      # 基线（A/B）
  volta-t2iopt validate --preset v100_sm70 --baseline results/val_... --fp32-ref results/val_...
  volta-t2iopt profile --preset v100_sm70 --size 1024
  volta-t2iopt matrix --presets noop v100_sm70 --sizes 512 1024
  volta-t2iopt microbench prescale                  # 分项证据
""")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    for name, mod in COMMANDS.items():
        mod.add_parser(sub, name)
    args = ap.parse_args(argv)
    return int(COMMANDS[args.cmd].run(args) or 0)


__all__ = ["main", "COMMANDS"]
