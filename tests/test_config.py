"""preset 文件 / 环境变量 / 显式覆盖 的优先级。"""

from __future__ import annotations

import pytest
import torch

from volta_t2iopt.capability import Capability
from volta_t2iopt.config import (OptimizerConfig, dtype_from_str, dtype_name,
                                 preset_path)


def cap() -> Capability:
    return Capability(device_name="fake", sm=70, memory_gib=32, torch_version="2.8.0",
                      arch_list=("sm_70",))


def test_preset_toml_is_parsed():
    cfg = OptimizerConfig.from_preset("v100_sm70", model_dtype="bfloat16")
    assert cfg.model_dtype == torch.bfloat16
    assert cfg.effective_decoder_dtype() == torch.float16
    assert cfg.enabled_passes() == ["hybrid_linear", "attn_fp16", "decoder_fp16"]
    assert cfg.pass_config("hybrid_linear").get("out_limit") == 50000.0


def test_noop_preset_turns_everything_off():
    cfg = OptimizerConfig.from_preset("noop", model_dtype="bfloat16")
    assert cfg.enabled_passes() == []


def test_env_disable_overrides_preset(monkeypatch):
    monkeypatch.setenv("VOLTA_T2IOPT_DISABLE", "hybrid_linear, attn_fp16")
    cfg = OptimizerConfig.from_preset("v100_sm70", model_dtype="bfloat16")
    assert cfg.enabled_passes() == ["decoder_fp16"]


def test_env_enable_overrides_preset(monkeypatch):
    monkeypatch.setenv("VOLTA_T2IOPT_ENABLE", "hybrid_linear")
    cfg = OptimizerConfig.from_preset("noop", model_dtype="bfloat16")
    assert "hybrid_linear" in cfg.enabled_passes()


def test_auto_uses_capability(monkeypatch):
    cfg = OptimizerConfig.from_preset("auto", capability=cap(), model_dtype="bfloat16")
    assert cfg.preset == "auto"
    assert cfg.enabled_passes() == ["hybrid_linear", "attn_fp16", "decoder_fp16"]


def test_unknown_preset_raises():
    with pytest.raises(KeyError):
        preset_path("does-not-exist")


def test_dtype_roundtrip():
    for name in ("float16", "bfloat16", "float32"):
        assert dtype_name(dtype_from_str(name)) == name


def test_describe_mentions_every_pass():
    text = OptimizerConfig.from_preset("v100_sm70", model_dtype="bfloat16").describe()
    assert "hybrid_linear" in text and "decoder=" in text
