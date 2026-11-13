"""能力探针 → 配置的推导（这是"框架"与"脚本"的分界线，必须有测试）。"""

from __future__ import annotations

import pytest
import torch

from volta_t2iopt.capability import Capability
from volta_t2iopt.config import auto_config


def cap(sm: int, measured: dict | None = None) -> Capability:
    return Capability(device_name=f"fake-sm{sm}", sm=sm, memory_gib=32.0,
                      torch_version="2.8.0", arch_list=("sm_70", "sm_80"), measured=measured or {})


def test_volta_bf16_model_gets_all_passes():
    cfg = auto_config(cap(70), torch.bfloat16)
    assert cfg.enabled_passes() == ["hybrid_linear", "attn_fp16", "decoder_fp16"]
    assert cfg.effective_decoder_dtype() == torch.float16
    assert any("bf16" in n for n in cfg.notes)


def test_ampere_bf16_model_gets_nothing():
    cfg = auto_config(cap(80), torch.bfloat16)
    assert cfg.enabled_passes() == []
    assert cfg.effective_decoder_dtype() == torch.bfloat16
    assert any("张量核" in n for n in cfg.notes)


def test_fp16_native_model_gets_nothing_on_volta():
    cfg = auto_config(cap(70), torch.float16)
    assert cfg.enabled_passes() == []
    assert cfg.effective_decoder_dtype() == torch.float16


def test_pascal_gets_nothing_because_no_fp16_tensorcore():
    cfg = auto_config(cap(60), torch.bfloat16)
    assert cfg.enabled_passes() == []


def test_measured_ratio_is_quoted_in_notes():
    cfg = auto_config(cap(70, dict(gemm_fp16_tflops=20.0, gemm_bf16_tflops=2.4)), torch.bfloat16)
    assert any("0.12" in n for n in cfg.notes)


def test_missing_cubin_warning():
    c = Capability(device_name="x", sm=70, memory_gib=32, torch_version="2.9.0",
                   arch_list=("sm_80", "sm_90"))
    cfg = auto_config(c, torch.bfloat16)
    assert any("arch_list" in n for n in cfg.notes)


@pytest.mark.parametrize("sm,expected", [(70, True), (75, True), (80, True), (60, False)])
def test_fp16_tensorcore_threshold(sm, expected):
    assert cap(sm).fp16_tensorcore is expected


@pytest.mark.parametrize("sm,expected", [(70, False), (75, False), (80, True), (90, True)])
def test_bf16_tensorcore_threshold(sm, expected):
    assert cap(sm).bf16_tensorcore is expected
