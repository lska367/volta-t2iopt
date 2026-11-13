"""路径口径（缓存目录 / 结果目录）集中在这里，不在业务代码里硬编码。"""

from __future__ import annotations

import os
from pathlib import Path

# 模型缓存根：本机放在 /home/lskam/models，先用环境变量覆盖
CACHE_ENV = "VOLTA_T2IOPT_CACHE"
OUT_ENV = "VOLTA_T2IOPT_OUT"


def cache_root() -> Path:
    return Path(os.environ.get(CACHE_ENV, "/home/lskam/models"))


def results_root() -> Path:
    """实验产物根目录（图 / 汇总 JSON / trace）。默认是当前工作目录下的 results/。"""
    root = Path(os.environ.get(OUT_ENV) or (Path.cwd() / "results"))
    root.mkdir(parents=True, exist_ok=True)
    return root
