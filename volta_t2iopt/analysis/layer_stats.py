"""逐层记录 `nn.Linear` 的 |x|max / |out|max，算出每层需要的 2 的幂预缩放 k。

用途有二：
1. **诊断**：哪些层在 fp16 下会溢出（|out|max > 65504 就是硬失败，不是精度问题）。
2. **解释**：为什么 pass 不能在加载时定死 k —— 把 1024px 与 512px 两次统计一对比，
   同一个 `noise_refiner.1.feed_forward.w2` 的输入 |max| 是 5824 vs 25472。

限制：只能在**未替换**的模型上用（替换后是 HybridLinear，不是 nn.Linear）。
"""

from __future__ import annotations

import torch

from ..passes.hybrid_linear import FP16_MAX, OUT_LIMIT, required_k


class LinearStats:
    """挂 forward hook，只存标量（不占显存）。"""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._pending: dict[int, tuple[str, torch.Tensor]] = {}
        self.enabled = True

    def _pre(self, name: str, mod: torch.nn.Module, args: tuple) -> None:
        if not self.enabled or not args:
            return
        x = args[0]
        if torch.is_tensor(x):
            self._pending[id(mod)] = (name, x)

    def _post(self, name: str, mod: torch.nn.Module, args: tuple, out: torch.Tensor) -> None:
        if not self.enabled or id(mod) not in self._pending:
            return
        _, x = self._pending.pop(id(mod))
        x_max = float(x.abs().max().item())
        out_max = float(out.abs().max().item())
        weight = getattr(mod, "weight", None)
        rowsum = (float(weight.detach().abs().sum(dim=1).max().item())
                  if torch.is_tensor(weight) else 0.0)
        self.rows.append(dict(
            name=name, m=x.numel() // max(x.shape[-1], 1),
            dim_in=int(x.shape[-1]), dim_out=int(out.shape[-1]),
            x_max=round(x_max, 4), out_max=round(out_max, 4), rowsum=round(rowsum, 1),
            x_fp16_ok=bool(x_max < FP16_MAX), out_fp16_ok=bool(out_max < FP16_MAX),
            k=required_k(x_max, rowsum),
            scale=2.0 ** -required_k(x_max, rowsum),
        ))

    def attach(self, module: torch.nn.Module | None, base: str = "") -> int:
        if module is None:
            return 0
        n = 0
        for name, mod in module.named_modules():
            if isinstance(mod, torch.nn.Linear):
                full = f"{base}.{name}" if base else name
                mod.register_forward_pre_hook(lambda m, a, n=full: self._pre(n, m, a))
                mod.register_forward_hook(lambda m, a, o, n=full: self._post(n, m, a, o))
                n += 1
        return n


def recommend_k(rows: list[dict], out_limit: float = OUT_LIMIT) -> list[dict]:
    """按实测 |x|max / |out|max 给出每层的 k（只用于诊断报告，运行时不依赖它）。"""
    for r in rows:
        need = max(r.get("x_max", 0.0), r.get("out_max", 0.0)) / out_limit
        k = 0
        while 2.0**k < need:
            k += 1
        r["k_static"] = min(k, 20)
    return rows


def format_layer_table(rows: list[dict], sort_key: str = "x_max", limit: int = 20) -> str:
    if not rows:
        return "没有逐层统计"
    need = [r for r in rows if r["k"] > 0]
    unsafe = [r for r in rows if not (r["x_fp16_ok"] and r["out_fp16_ok"])]
    head = (f"{len(rows)} 个 Linear | 需要预缩放 {len(need)} | "
            f"fp16 不安全 {len(unsafe)}（|out|max > 65504）")
    lines = [head, f"{'层':46s} {'M':>5s} {'K':>6s} {'N':>6s} {'|x|max':>11s} "
                   f"{'|out|max':>11s} {'rowsum':>8s} {'k':>2s}"]
    for r in sorted(rows, key=lambda r: -max(r[sort_key], r["out_max"]))[:limit]:
        lines.append(f"{r['name'][:46]:46s} {r['m']:5d} {r['dim_in']:6d} {r['dim_out']:6d} "
                     f"{r['x_max']:11.3e} {r['out_max']:11.3e} {r['rowsum']:8.1f} {r['k']:2d}")
    return "\n".join(lines)
