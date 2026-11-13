"""会话层：多 pass 编排、失败降级、报告、回滚。"""

from __future__ import annotations

import torch

from volta_t2iopt.capability import Capability
from volta_t2iopt.config import OptimizerConfig, PassConfig
from volta_t2iopt.passes.hybrid_linear import HybridLinear
from volta_t2iopt.session import OptimizationSession


class FakeAdapter:
    """适配器协议的最小实现（只测会话逻辑，不需要真 pipeline / GPU）。"""

    native_dtype = torch.bfloat16

    def __init__(self) -> None:
        self._denoiser = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.Linear(8, 4))
        self._decoder = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1))

    def denoiser(self, pipe=None):
        return self._denoiser

    def decoder(self, pipe=None):
        return self._decoder


class ExplodingAdapter(FakeAdapter):
    def denoiser(self, pipe=None):
        raise RuntimeError("boom")


def capability() -> Capability:
    return Capability(device_name="fake", sm=70, memory_gib=32, torch_version="2.8.0",
                     arch_list=("sm_70",))


def make_session(**passes) -> tuple[OptimizationSession, FakeAdapter]:
    cfg = OptimizerConfig(preset="test", model_dtype=torch.bfloat16, device="cpu")
    for name, enabled in passes.items():
        cfg.passes[name] = PassConfig(enabled)
    adapter = FakeAdapter()
    sess = OptimizationSession(config=cfg, capability=capability(), adapter=adapter,
                               pipe=None, model_dtype=torch.bfloat16)
    return sess, adapter


def test_apply_reports_each_pass_and_reverts():
    sess, adapter = make_session(hybrid_linear=True)
    sess.apply()
    assert sess.applied
    assert [r.pass_name for r in sess.reports] == ["hybrid_linear"]
    assert sess.reports[0].applied
    assert any(isinstance(m, HybridLinear) for m in adapter._denoiser.modules())
    sess.revert()
    assert not any(isinstance(m, HybridLinear) for m in adapter._denoiser.modules())


def test_disabled_pass_is_not_listed():
    sess, _ = make_session(hybrid_linear=False)
    sess.apply()
    assert sess.reports == []


def test_context_manager_reverts_on_exit():
    sess, adapter = make_session(hybrid_linear=True)
    with sess:
        assert any(isinstance(m, HybridLinear) for m in adapter._denoiser.modules())
    assert not any(isinstance(m, HybridLinear) for m in adapter._denoiser.modules())


def test_summary_contains_config_and_reports():
    sess, _ = make_session(hybrid_linear=True)
    sess.apply()
    text = sess.summary()
    assert "volta-t2iopt session" in text
    assert "hybrid_linear" in text
    assert "applied" in text


def test_apply_is_idempotent():
    sess, _ = make_session(hybrid_linear=True)
    sess.apply()
    n = len(sess.reports)
    sess.apply()
    assert len(sess.reports) == n


def test_pass_exception_is_degraded_to_skipped_report():
    cfg = OptimizerConfig(preset="test", model_dtype=torch.bfloat16, device="cpu")
    cfg.passes["hybrid_linear"] = PassConfig(True)
    sess = OptimizationSession(config=cfg, capability=capability(), adapter=ExplodingAdapter(),
                               pipe=None, model_dtype=torch.bfloat16)
    sess.apply()   # 不抛
    rep = sess.reports[0]
    assert not rep.applied and "boom" in rep.reason


def test_inapplicable_pass_reports_reason():
    sess, _ = make_session(attn_fp16=True)
    sess.model_dtype = torch.float16     # 非 bf16 → attn pass 不适用
    sess.apply()
    assert not sess.reports[0].applied and sess.reports[0].reason
