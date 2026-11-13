"""质量指标与门限。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from volta_t2iopt.validate.gate import GateCriteria, check_gate
from volta_t2iopt.validate.metrics import psnr, rel_l2, ssim


def test_ssim_of_identical_images_is_one():
    a = np.random.default_rng(0).random((32, 32, 3), dtype=np.float32)
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-9)


def test_ssim_drops_with_noise():
    rng = np.random.default_rng(1)
    a = rng.random((32, 32, 3), dtype=np.float32)
    b = np.clip(a + rng.normal(0, 0.2, a.shape).astype(np.float32), 0, 1)
    assert 0.0 < ssim(a, b) < 0.99


def test_rel_l2_and_psnr():
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)
    assert rel_l2(a, b) == 0.0
    assert psnr(np.ones((4, 4)), np.ones((4, 4))) > 100


def _fake_result(lat, img, ms_per_fwd=1000.0):
    return dict(latents=torch.as_tensor(lat), image=img,
                timing=dict(ms_per_fwd=ms_per_fwd))


def test_latent_rel_l2_is_advisory_by_default():
    img = np.ones((16, 16, 3), dtype=np.float32)
    base = _fake_result(np.ones((1, 4)), img)
    other = _fake_result(np.full((1, 4), 3.0), img)
    gate = check_gate(base, other)          # 默认不判 relL2
    assert gate.passed
    assert "latent_rel_l2" in gate.metrics
    assert not any(c["name"] == "latent_relL2" for c in gate.checks)
    strict = check_gate(base, other, GateCriteria(max_rel_l2=2e-3))
    assert not strict.passed


def test_gate_pass_and_fail():
    img = np.random.default_rng(2).random((32, 32, 3), dtype=np.float32)
    base = _fake_result(np.ones((1, 4)), img)
    good = _fake_result(np.ones((1, 4)), img)
    gate = check_gate(base, good)
    assert gate.passed, gate.format()

    bad = _fake_result(np.full((1, 4), 9.0), np.zeros_like(img))
    gate = check_gate(base, bad)
    assert not gate.passed
    assert not all(c["ok"] for c in gate.checks)


def test_latency_ratio_is_checked_when_requested():
    img = np.ones((16, 16, 3), dtype=np.float32)
    base = _fake_result(np.ones((1, 2)), img, ms_per_fwd=1000.0)
    slow = _fake_result(np.ones((1, 2)), img, ms_per_fwd=700.0)
    crit = GateCriteria(max_latency_ratio=0.9)
    gate = check_gate(base, slow, crit)
    assert gate.metrics["latency_ratio"] == "0.700×"
    assert gate.passed
