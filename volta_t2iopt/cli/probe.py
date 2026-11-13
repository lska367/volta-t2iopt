"""probe：打印能力 + 由能力推导出的配置（不加载模型）。"""

from __future__ import annotations

import argparse
import json

import torch

from ..capability import probe
from ..config import OptimizerConfig


def add_parser(sub, name: str = "probe") -> None:
    p = sub.add_parser(name, help="打印 GPU 能力与推导出的配置")
    p.add_argument("--measure", action="store_true", help="跑 GEMM/SDPA 微基准（约 2 s）")
    p.add_argument("--dtype", default="bfloat16", help="模型原生精度（决定是否迁移）")
    p.add_argument("--preset", default="auto")
    p.add_argument("--json", action="store_true")


def run(args) -> int:
    cap = probe(measure=args.measure)
    cfg = OptimizerConfig.from_preset(args.preset, capability=cap,
                                      model_dtype=args.dtype)
    if args.json:
        print(json.dumps(dict(capability=cap.__dict__, config=cfg.to_dict()), indent=2,
                         ensure_ascii=False, default=str))
        return 0
    print(cap.summary())
    print(f"\n{cfg.describe()}")
    print("\n" + "=" * 70)
    print("结论：" + _conclusion(cfg))
    return 0


def _conclusion(cfg: OptimizerConfig) -> str:
    names = cfg.enabled_passes()
    if not names:
        return "不需要任何迁移（" + "; ".join(cfg.notes) + "）"
    return "启用 " + ", ".join(names)
