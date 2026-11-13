"""质量指标。故意保持"笨但确定"：没有依赖 scipy/skimage，数字全程可复现。"""

from __future__ import annotations

import numpy as np


def rel_l2(a, b) -> float:
    """`||a-b|| / ||b||`；a/b 可以是 numpy 或 torch 张量。"""
    if hasattr(a, "detach"):
        a = a.detach().float().cpu().numpy()
    if hasattr(b, "detach"):
        b = b.detach().float().cpu().numpy()
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def ssim(a: np.ndarray, b: np.ndarray, c1: float = 0.01**2, c2: float = 0.03**2) -> float:
    """全局 SSIM（11×11 均值滤波近似）。支持 (H,W) 与 (H,W,C)，多通道取平均。

    这是本项目历史结论使用的同一套算法 —— 换实现会改变 0.982 这种数字，所以不要"顺手升级"。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 3:
        return float(np.mean([ssim(a[..., c], b[..., c], c1, c2) for c in range(a.shape[-1])]))

    k, pad = 11, 5
    ap = np.pad(a, pad, mode="reflect")
    bp = np.pad(b, pad, mode="reflect")
    wa = sliding_window_view(ap, (k, k)).mean(axis=(-1, -2))
    wb = sliding_window_view(bp, (k, k)).mean(axis=(-1, -2))
    va = sliding_window_view(ap**2, (k, k)).mean(axis=(-1, -2)) - wa**2
    vb = sliding_window_view(bp**2, (k, k)).mean(axis=(-1, -2)) - wb**2
    cov = sliding_window_view(ap * bp, (k, k)).mean(axis=(-1, -2)) - wa * wb
    s = ((2 * wa * wb + c1) * (2 * cov + c2)) / ((wa**2 + wb**2 + c1) * (va + vb + c2))
    return float(s.mean())


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2).mean())
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def image_stats(arr: np.ndarray) -> dict:
    return dict(mean=round(float(arr.mean()), 4), std=round(float(arr.std()), 4),
                min=float(arr.min()), max=float(arr.max()),
                nan=bool(np.isnan(arr).any()))
