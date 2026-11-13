"""chrome trace → "只算 kernel"的统计。

`key_averages()` 会把 `aten::*` 和真 kernel 重复计，所以这里直接解析 trace 里的
`ph == "X"` 且 `cat in (kernel, gpu_memcpy, gpu_memset)` 的事件。

顺带算一件事：**GPU 空转比例** = 1 − Σkernel 时间 / 时间轴跨度。
它是 CUDA Graph 的收益上限 —— 但注意空转会**被 profiler 放大**（S3 档显示 16.5%，
去掉 profiler 后真实空转几乎为 0），所以这个数只能用来判断"值不值得挖"，不能当结论。
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path

# kernel 名 → 类别。规则先粗后细：按 GEMM / attention 的强特征分，剩下的都算 other。
_GEMM = re.compile(r"gemm|xmma|cutlass|nvjet|volta_|ampere_|sm70|sm80|magma_sgemm|hfma|"
                   r"wgrad|dgrad|884|16816|1688", re.I)
_ATTN = re.compile(r"flash|fmha|attention|scaled_dot|softmax|bmm", re.I)
_ELT = re.compile(r"elementwise|vectorized|unrolled|silu|gelu|mul|add|div|rms|layer_norm|"
                  r"layernorm|reduce|copy|cat|pad|slice|contiguous|fill|cast|convert|index|"
                  r"gather|scatter|transpose", re.I)


def classify(name: str) -> str:
    # 顺序有讲究：attention 必须先判。
    # 旧脚本是先判 GEMM，而 fmha 的 kernel 名里同时带 `cutlass` 和 `sm70`
    # （fmha_cutlassF_f16_aligned_32x128_rf_sm70），结果被算进了 gemm 桶。
    if _ATTN.search(name):
        return "attn"
    if _GEMM.search(name):
        return "gemm"
    if _ELT.search(name):
        return "elt"
    return "other"


def load_kernels(path: Path | str) -> list[dict]:
    p = Path(path)
    # 归档时 trace 会 gzip（180 MB → 15 MB），这里直接支持 .gz
    text = gzip.open(p, "rt").read() if p.suffix == ".gz" else p.read_text()
    blob = json.loads(text)
    events = blob["traceEvents"] if isinstance(blob, dict) else blob
    return [e for e in events
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset")
            and e.get("dur", 0) > 0]


def kernel_stats(path: Path | str, top: int = 12) -> dict:
    kernels = load_kernels(path)
    if not kernels:
        return dict(kernel_ms=0.0, span_ms=0.0, idle_pct=0.0, launches=0, distinct=0,
                    cat_ms={}, cat_launches={}, top=[], empty=True)
    total = sum(k["dur"] for k in kernels)
    span = max(k["ts"] + k["dur"] for k in kernels) - min(k["ts"] for k in kernels)
    cat: dict[str, float] = {}
    cat_n: Counter = Counter()
    per: dict[str, list[int]] = {}
    for k in kernels:
        c = classify(k["name"])
        cat[c] = cat.get(c, 0.0) + k["dur"]
        cat_n[c] += 1
        slot = per.setdefault(k["name"], [0, 0])
        slot[0] += k["dur"]
        slot[1] += 1
    return dict(
        kernel_ms=round(total / 1e3, 1),
        span_ms=round(span / 1e3, 1),
        idle_pct=round((span - total) / span * 100, 1),
        launches=len(kernels),
        distinct=len(per),
        us_per_kernel=round(total / max(len(kernels), 1), 1),
        cat_ms={c: round(v / 1e3, 1) for c, v in sorted(cat.items(), key=lambda kv: -kv[1])},
        cat_launches=dict(cat_n),
        top=[dict(name=n[:70], ms=round(d / 1e3, 1), calls=c, us=round(d / c, 1),
                  cat=classify(n))
             for n, (d, c) in sorted(per.items(), key=lambda kv: -kv[1][0])[:top]],
        empty=False,
    )


def analyze_trace(path: Path | str, top: int = 25) -> dict:
    """完整统计（top-N 更长），也是 CLI `report trace` 用的口径。"""
    stats = kernel_stats(path, top=top)
    stats["file"] = str(path)
    return stats


def format_kernel_stats(stats: dict, top: int = 12) -> str:
    if stats.get("empty"):
        return "没有 kernel 事件（检查 trace 是否包含 CUDA 活动）"
    lines = [f"kernel {stats['kernel_ms'] / 1000:.2f} s | 时间轴 {stats['span_ms'] / 1000:.2f} s | "
             f"空转 {stats['idle_pct']:.1f}% (profiler 口径偏高)",
             f"发射 {stats['launches']} 次 / {stats['distinct']} 种 | "
             f"平均 {stats['us_per_kernel']:.1f} µs/kernel",
             "分类: " + "  ".join(f"{c} {v:.0f} ms" for c, v in stats["cat_ms"].items()),
             f"{'kernel':66s} {'ms':>8s} {'次数':>6s} {'µs':>8s} 类别"]
    for t in stats["top"][:top]:
        lines.append(f"{t['name'][:66]:66s} {t['ms']:8.1f} {t['calls']:6d} {t['us']:8.1f} "
                     f"{t['cat']}")
    return "\n".join(lines)
