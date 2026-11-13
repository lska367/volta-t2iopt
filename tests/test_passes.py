"""pass 协议：幂等、可回滚、会自述原因。"""

from __future__ import annotations

import torch

from volta_t2iopt.capability import Capability
from volta_t2iopt.config import OptimizerConfig, PassConfig
from volta_t2iopt.passes import Context, get_pass, order_passes
from volta_t2iopt.passes.hybrid_linear import HybridLinear, HybridLinearPass


class FakeAdapter:
    native_dtype = torch.bfloat16

    def __init__(self, denoiser, decoder=None):
        self._denoiser, self._decoder = denoiser, decoder

    def denoiser(self, pipe=None):
        return self._denoiser

    def decoder(self, pipe=None):
        return self._decoder


def make_ctx(denoiser, decoder=None, sm: int = 70, **passes) -> Context:
    cap = Capability(device_name="fake", sm=sm, memory_gib=32, torch_version="2.8.0",
                     arch_list=("sm_70",))
    cfg = OptimizerConfig(preset="test", model_dtype=torch.bfloat16, device="cpu")
    for name, params in passes.items():
        cfg.passes[name] = PassConfig(**params) if isinstance(params, dict) else params
    return Context(pipe=None, adapter=FakeAdapter(denoiser, decoder), capability=cap,
                   config=cfg, model_dtype=torch.bfloat16)


def test_pass_order_is_stable():
    assert order_passes(["decoder_fp16", "attn_fp16", "hybrid_linear"]) == \
        ["hybrid_linear", "attn_fp16", "decoder_fp16"]


def test_hybrid_pass_refuses_on_ampere():
    model = torch.nn.Sequential(torch.nn.Linear(8, 4))
    ok, reason = HybridLinearPass(make_ctx(model, sm=80)).applicable()
    assert not ok and "bf16" in reason


def test_hybrid_pass_refuses_on_fp16_model():
    ctx = make_ctx(torch.nn.Sequential(torch.nn.Linear(8, 4)))
    ctx.model_dtype = torch.float16
    ok, _ = HybridLinearPass(ctx).applicable()
    assert not ok


def test_hybrid_pass_apply_revert_roundtrip():
    model = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.Linear(16, 4))
    p = HybridLinearPass(make_ctx(model, hybrid_linear=PassConfig(True)))
    rep = p.apply()
    assert rep.applied and rep.modules == 2
    assert any(isinstance(m, HybridLinear) for m in model.modules())
    p.revert()
    assert not any(isinstance(m, HybridLinear) for m in model.modules())
    assert all(isinstance(m, torch.nn.Linear) for m in model)


def test_attn_pass_apply_is_idempotent_and_reversible():
    from diffusers.models import attention_dispatch

    from volta_t2iopt.passes.attn_fp16 import AttentionFp16Pass

    orig = attention_dispatch.dispatch_attention_fn
    ctx = make_ctx(torch.nn.Identity(), attn_fp16=PassConfig(True))
    p = AttentionFp16Pass(ctx)
    rep = p.apply()
    try:
        assert rep.applied
        patched = attention_dispatch.dispatch_attention_fn
        assert patched is not orig
        assert getattr(patched, "_vt2_attn_fp16", False)
        second = AttentionFp16Pass(ctx).apply()
        assert not second.applied and "幂等" in second.reason
    finally:
        p.revert()
    assert attention_dispatch.dispatch_attention_fn is orig


def test_decoder_pass_reports_noop_when_already_fp16():
    from volta_t2iopt.passes.decoder_fp16 import DecoderFp16Pass

    vae = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1)).to(torch.float16)
    ctx = make_ctx(torch.nn.Identity(), decoder=vae, decoder_fp16=PassConfig(True))
    ctx.config.decoder_dtype = torch.float16
    rep = DecoderFp16Pass(ctx).apply()
    assert rep.applied and rep.details.get("already") is True


def test_decoder_pass_casts_and_reverts_to_original_dtype():
    from volta_t2iopt.passes.decoder_fp16 import DecoderFp16Pass

    vae = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1))     # 原始 fp32
    ctx = make_ctx(torch.nn.Identity(), decoder=vae, decoder_fp16=PassConfig(True))
    ctx.config.decoder_dtype = torch.float16
    p = DecoderFp16Pass(ctx)
    rep = p.apply()
    assert rep.applied and rep.details["from_dtype"] == "float32"
    assert next(vae.parameters()).dtype == torch.float16
    p.revert()
    assert next(vae.parameters()).dtype == torch.float32


def test_registry_has_expected_passes():
    for name in ("hybrid_linear", "attn_fp16", "decoder_fp16"):
        assert get_pass(name).name == name
