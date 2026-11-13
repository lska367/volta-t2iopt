"""cache：离线产物（bf16 权重 / prompt 条件）。"""

from __future__ import annotations

import argparse

from ..models import get_adapter
from ..runtime.cache import build_prompt_embeds, build_weight_cache, dir_size
from ..paths import cache_root


def add_parser(sub, name: str = "cache") -> None:
    p = sub.add_parser(name, help="构建离线缓存（权重 / 文本条件）")
    p.add_argument("what", choices=["all", "weights", "embeds"], nargs="?", default="all")
    p.add_argument("--model", default="z-image-turbo")
    p.add_argument("--src", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--force", action="store_true")


def run(args) -> int:
    adapter = get_adapter(args.model)
    if args.what in ("all", "weights"):
        if adapter.cache_spec is None:
            print(f"{adapter.name} 不需要权重缓存（原生精度即可）")
        else:
            dst = build_weight_cache(adapter, src=args.src, force=args.force, dtype=args.dtype)
            print(f"权重缓存：{dst} ({dir_size(dst):.1f} GiB)")
    if args.what in ("all", "embeds"):
        dtype = __import__("torch").bfloat16
        path = build_prompt_embeds(adapter, dtype=dtype, force=args.force)
        print(f"条件缓存：{path} ({path.stat().st_size / 2**20:.1f} MiB)")
    print(f"cache_root = {cache_root()}")
    return 0
