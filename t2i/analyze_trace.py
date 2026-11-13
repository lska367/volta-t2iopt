"""从 chrome trace 里做**只算 kernel**的统计（`key_averages()` 会把 aten::* 和真 kernel 重复计）。

除时间分布外还算一件事：GPU 空转比例 = 1 − Σkernel 时间 / 时间轴跨度。
它直接给出 CUDA Graph 的收益上限 —— 空转就是发射/同步等出来的。

用法：
    python t2i/analyze_trace.py t2i/out/trace_1024.json [--top 25]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

_GEMM = re.compile(r"gemm|xmma|cutlass|nvjet|volta_|ampere_|sm70|sm80|magma_sgemm|hfma|wgrad|884|16816", re.I)
_ATTN = re.compile(r"flash|fmha|attention|scaled_dot|softmax|bmm", re.I)
_ELT = re.compile(r"elementwise|vectorized|unrolled|silu|gelu|mul|add|div|rms|layer_norm|layernorm|"
                  r"reduce|copy|cat|pad|slice|contiguous|fill|cast|convert|index|gather|scatter|transpose", re.I)


def classify(name: str) -> str:
    if _GEMM.search(name):
        return "gemm"
    if _ATTN.search(name):
        return "attn"
    if _ELT.search(name):
        return "elt"
    return "other"


def load_kernels(path: Path) -> list[dict]:
    blob = json.loads(path.read_text())
    evts = blob["traceEvents"] if isinstance(blob, dict) else blob
    return [e for e in evts
            if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel", "gpu_memcpy", "gpu_memset")
            and e.get("dur", 0) > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    path = Path(args.trace)
    ks = load_kernels(path)
    if not ks:
        print("没有 kernel 事件（检查 trace 是否包含 CUDA 活动）")
        return

    total = sum(k["dur"] for k in ks)
    span = max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)
    cat: Counter = Counter()
    cat_n: Counter = Counter()
    per: dict[str, list[int]] = {}
    for k in ks:
        c = classify(k["name"])
        cat[c] += k["dur"]
        cat_n[c] += 1
        slot = per.setdefault(k["name"], [0, 0])
        slot[0] += k["dur"]
        slot[1] += 1

    print(f"trace      : {path.name}")
    print(f"kernel 时间 : {total/1e6:.2f} s | 时间轴跨度 {span/1e6:.2f} s | "
          f"GPU 空转 {(span-total)/span*100:.1f}%")
    print(f"kernel 发射 : {len(ks)} 次 / {len(per)} 种 | 平均 {total/max(len(ks),1):.1f} µs/kernel")
    print("\n[分类]")
    for c, v in cat.most_common():
        print(f"  {c:6s} {v/1e6:7.2f} s  {100*v/total:5.1f}%   {cat_n[c]:6d} 次")
    print(f"\n{'kernel':64s} {'总 ms':>9s} {'次数':>6s} {'占比':>6s} {'µs/次':>8s} 类别")
    for name, (dur, n) in sorted(per.items(), key=lambda kv: -kv[1][0])[:args.top]:
        print(f"{name[:64]:64s} {dur/1e3:9.1f} {n:6d} {100*dur/total:5.1f}% "
              f"{dur/max(n,1):8.1f} {classify(name)}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(dict(
            kernel_s=round(total / 1e6, 3), span_s=round(span / 1e6, 3),
            idle_pct=round((span - total) / span * 100, 1), launches=len(ks),
            cat_s={c: round(v / 1e6, 3) for c, v in cat.most_common()},
            cat_launches=dict(cat_n),
            top=[dict(name=n, ms=round(d / 1e3, 2), calls=c, us_per_call=round(d / c, 1))
                 for n, (d, c) in sorted(per.items(), key=lambda kv: -kv[1][0])[:args.top]],
        ), indent=2, ensure_ascii=False))
        print(f"\n-> {args.json_out}")


if __name__ == "__main__":
    main()
