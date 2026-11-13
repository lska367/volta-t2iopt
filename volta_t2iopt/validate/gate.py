"""验收门限：把"看起来差不多"变成可判定的 PASS / FAIL。

两个门限，一个默认开、一个默认关 —— 依据是实测而不是"看起来严格"：

1. **SSIM ≥ 0.98 + 无 inf/nan（默认判定）**。这是质量结论的正确口径，必须对 **fp32 真值**比。
   ⚠️ 拿最终版对 *bf16 基线* 比：9 步图 SSIM 0.978，会被判 FAIL，但那不是回归
   （对 fp32 真值 0.982 反而比 bf16 基线的 0.968 更好）。所以 `reference` 只影响措辞，
   真正的规矩是"质量看 fp32"。
2. **latent relL2（默认只报不判）**。历史上写的 2e-3 在 latent 尺度上从来没达到过：
   实测 1 次 forward 的 bf16 vs fp16-hybrid latent relL2 ≈ 2.7e-2，9 步 ≈ 8e-2 ——
   扩散采样会把每步微小差异放大，relL2 不是"实现错没错"的判据（那个判据在单层 GEMM 上，
   见 `microbench gemm-error`，量级 1e-4）。要卡就显式传 `max_rel_l2`。
3. **延迟比（可选）**：`max_latency_ratio` 给定时按 `candidate/base` 判。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .metrics import psnr, rel_l2, ssim


@dataclass
class GateCriteria:
    min_ssim: float = 0.98
    max_rel_l2: float | None = None          # None = 只报不判（见模块 docstring 第 2 条）
    max_latency_ratio: float | None = None   # candidate / baseline 的每 forward 时间上限
    forbid_nonfinite: bool = True
    reference: str = "fp32"                  # 报告措辞：这条门限是对谁定的

    def to_dict(self) -> dict:
        return dict(min_ssim=self.min_ssim, max_rel_l2=self.max_rel_l2,
                    max_latency_ratio=self.max_latency_ratio,
                    forbid_nonfinite=self.forbid_nonfinite, reference=self.reference)


@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    criteria: GateCriteria = field(default_factory=GateCriteria)
    notes: list[str] = field(default_factory=list)

    def line(self) -> str:
        detail = " ".join(f"{c['name']}={'ok' if c['ok'] else 'FAIL'}" for c in self.checks)
        return f"GATE {'PASS' if self.passed else 'FAIL'}  {detail}"

    def format(self) -> str:
        rows = [self.line()]
        for c in self.checks:
            rows.append(f"  {c['name']:14s} {c['value']:>14s}   (门限 {c['threshold']}) "
                        f"{'PASS' if c['ok'] else 'FAIL'}")
        for k, v in self.metrics.items():
            if k not in {c["name"] for c in self.checks}:
                rows.append(f"  {k:14s} {v}")
        rows.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return dict(passed=self.passed, checks=self.checks, metrics=self.metrics,
                    criteria=self.criteria.to_dict(), notes=self.notes)


def check_gate(baseline, candidate, criteria: GateCriteria | None = None) -> GateResult:
    """baseline / candidate 可以是 `RunResult`、`RunResult.to_dict()` 或含同名字段的 dict。"""
    criteria = criteria or GateCriteria()
    checks: list[dict] = []
    metrics: dict = {}
    notes: list[str] = []

    b_lat = _field(baseline, "latents")
    c_lat = _field(candidate, "latents")
    b_img = _to_array(_field(baseline, "image"))
    c_img = _to_array(_field(candidate, "image"))

    # --- latent ---
    if b_lat is not None and c_lat is not None:
        try:
            import torch
            rel = rel_l2(c_lat, b_lat)
            finite = bool(torch.isfinite(torch.as_tensor(c_lat)).all())
        except Exception:
            rel, finite = float("nan"), True
        metrics["latent_rel_l2"] = f"{rel:.3e}"
        if criteria.max_rel_l2 is not None:
            checks.append(dict(name="latent_relL2", value=f"{rel:.3e}",
                               threshold=f"< {criteria.max_rel_l2:g}",
                               ok=bool(rel < criteria.max_rel_l2)))
        else:
            notes.append("latent relL2 只作诊断：扩散会放大每步差异（1 forward ≈ 2.7e-2），"
                         "质量结论看 SSIM；要卡门限请显式传 max_rel_l2")
        if criteria.forbid_nonfinite:
            checks.append(dict(name="finite", value="ok" if finite else "inf/nan",
                               threshold="无 inf/nan", ok=finite))

    # --- 图像 ---
    if b_img is not None and c_img is not None:
        if b_img.shape != c_img.shape:
            notes.append(f"图像尺寸不同：{b_img.shape} vs {c_img.shape}，跳过图像指标")
        else:
            s = ssim(b_img, c_img)
            metrics["ssim"] = f"{s:.5f}"
            metrics["psnr_db"] = f"{psnr(b_img, c_img):.2f}"
            metrics["max_abs_diff"] = f"{np.abs(b_img - c_img).max():.4f}"
            metrics["mean_abs_diff"] = f"{np.abs(b_img - c_img).mean():.5f}"
            checks.append(dict(name="ssim", value=f"{s:.5f}",
                               threshold=f">= {criteria.min_ssim:g}",
                               ok=bool(s >= criteria.min_ssim)))

    # --- 时间 ---
    b_ms = _timing(_field(baseline, "timing"), baseline, "ms_per_fwd")
    c_ms = _timing(_field(candidate, "timing"), candidate, "ms_per_fwd")
    if b_ms and c_ms:
        ratio = c_ms / b_ms
        metrics["latency_ratio"] = f"{ratio:.3f}×"
        if criteria.max_latency_ratio is not None:
            checks.append(dict(name="latency", value=f"{ratio:.3f}×",
                               threshold=f"<= {criteria.max_latency_ratio:g}×",
                               ok=bool(ratio <= criteria.max_latency_ratio)))
        notes.append(f"每 forward：baseline {b_ms:.0f} ms → candidate {c_ms:.0f} ms "
                     f"（{ratio:.2f}×）")

    if criteria.reference != "fp32" or _field(baseline, "mode", "meta") in ("bf16", "baseline"):
        notes.append("对 bf16 基线的 SSIM 会低于对 fp32 真值的 SSIM（实测 0.9779 vs 0.982），"
                     "判质量时应以 fp32 真值为参考")

    passed = bool(checks) and all(c["ok"] for c in checks)
    if not checks:
        notes.append("没有可比指标（缺 latent / 图）")
    return GateResult(passed=passed, checks=checks, metrics=metrics, criteria=criteria,
                      notes=notes)


def compare_results(baseline, candidate, criteria: GateCriteria | None = None) -> GateResult:
    return check_gate(baseline, candidate, criteria)


def compare_files(base_stem: str | Path, cand_stem: str | Path,
                  criteria: GateCriteria | None = None) -> GateResult:
    """比对 `generate(save=True)` 落盘的三件套（png / latent.pt / json）。"""
    import torch
    from PIL import Image

    base, cand = Path(base_stem), Path(cand_stem)
    b: dict = {}
    c: dict = {}
    for tag, stem, slot in (("base", base, b), ("cand", cand, c)):
        png, pt, js = stem.with_suffix(".png"), stem.with_suffix(".latent.pt"), stem.with_suffix(".json")
        if png.exists():
            slot["image"] = Image.open(png).convert("RGB")
        if pt.exists():
            slot["latents"] = torch.load(pt, map_location="cpu", weights_only=False)
        if js.exists():
            meta = json.loads(js.read_text())
            slot["timing"] = meta
            slot["meta"] = meta
    return check_gate(b, c, criteria)


# --- 内部工具 ---
def _field(obj, name: str, nested: str | None = None):
    if obj is None:
        return None
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        if nested and isinstance(obj.get(nested), dict):
            return obj[nested].get(name)
        return None
    val = getattr(obj, name, None)
    if val is not None:
        return val
    if nested:
        inner = getattr(obj, nested, None)
        if isinstance(inner, dict):
            return inner.get(name)
    return None


def _timing(val, obj, key: str) -> float:
    if isinstance(val, dict):
        return float(val.get(key) or 0.0)
    if isinstance(obj, dict) and key in obj:
        return float(obj[key] or 0.0)
    return 0.0


def _to_array(img) -> np.ndarray | None:
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        return img.astype(np.float32)
    try:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:
        try:
            arr = np.asarray(img, dtype=np.float32)
        except Exception:
            return None
    return arr if arr.ndim >= 2 else None
