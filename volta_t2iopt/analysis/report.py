"""把落盘的 JSON 汇总成表（终端 / markdown 两种）。"""

from __future__ import annotations

import json
from pathlib import Path


def render_markdown_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "(空)"
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = ["| " + " | ".join(str(r.get(c, "")) for c in columns) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def load_jsons(directory: Path | str, pattern: str = "*.json") -> list[dict]:
    rows = []
    for f in sorted(Path(directory).glob(pattern)):
        try:
            blob = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(blob, dict):
            blob.setdefault("file", f.name)
        rows.append(blob)
    return rows


def suite_table(rows: list[dict]) -> str:
    """profile / matrix 结果的对照表：档位 × 分辨率 × 时间 × 分类占比。"""
    rows = [r for r in rows if "size" in r and ("fwd_clean_ms" in r or "fwd_profiled_ms" in r)]
    if not rows:
        return "没有可汇总的 profile 结果"
    rows.sort(key=lambda r: (-int(r.get("size", 0)), str(r.get("preset", ""))))
    head = (f"{'档位':10s} {'分辨率':>6s} {'fwd(prof)':>10s} {'fwd(净)':>8s} {'kernel':>8s} "
            f"{'空转%':>6s} {'发射':>6s} {'VAE':>7s} {'峰值GiB':>7s} | 分类 ms")
    lines = [head]
    for r in rows:
        k = r.get("kernels", {}) or {}
        lines.append(f"{str(r.get('preset', '?')):10s} {int(r.get('size', 0)):6d} "
                     f"{r.get('fwd_profiled_ms', 0):10.1f} {r.get('fwd_clean_ms', 0):8.1f} "
                     f"{k.get('kernel_ms', 0):8.1f} {k.get('idle_pct', 0):6.1f} "
                     f"{k.get('launches', 0):6d} {r.get('vae_ms', 0):7.1f} "
                     f"{r.get('peak_gib', 0):7.2f} | {k.get('cat_ms', {})}")
    return "\n".join(lines)


def bench_table(rows: list[dict]) -> str:
    """端到端结果对照：preset × 分辨率 × 总时间 × 每 forward × 质量指标。"""
    rows = [r for r in rows if "total_ms" in r]
    if not rows:
        return "没有可汇总的端到端结果"
    rows.sort(key=lambda r: (-int(r.get("size", 0)), str(r.get("preset", ""))))
    head = (f"{'preset':10s} {'模型':14s} {'分辨率':>6s} {'步数':>4s} {'总 s':>7s} "
            f"{'ms/fwd':>8s} {'VAE s':>6s} {'峰值GiB':>7s} {'std':>5s} NaN")
    lines = [head]
    for r in rows:
        lines.append(f"{str(r.get('preset', '?')):10s} {str(r.get('model', '?')):14s} "
                     f"{int(r.get('size', 0)):6d} {int(r.get('steps', 0)):4d} "
                     f"{r.get('total_ms', 0) / 1000:7.2f} {r.get('ms_per_fwd', 0):8.0f} "
                     f"{r.get('vae_ms', 0) / 1000:6.2f} {r.get('peak_gib', 0):7.2f} "
                     f"{r.get('img_std', 0):5.3f} {'YES' if r.get('img_nan') else 'no'}")
    return "\n".join(lines)
