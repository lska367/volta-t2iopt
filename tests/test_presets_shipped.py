"""随包发布的 preset 必须都能解析、且与文档里的档位一致。"""

from __future__ import annotations

import torch

from volta_t2iopt.config import PRESET_DIR, OptimizerConfig

EXPECTED = {
    "v100_sm70": ["hybrid_linear", "attn_fp16", "decoder_fp16"],
    "s3": ["hybrid_linear", "attn_fp16"],
    "s2": ["hybrid_linear"],
    "noop": [],
}


def test_all_shipped_presets_load():
    names = {p.stem for p in PRESET_DIR.glob("*.toml")}
    assert names == set(EXPECTED)


def test_preset_pass_sets_match_docs():
    for name, passes in EXPECTED.items():
        cfg = OptimizerConfig.from_preset(name, model_dtype="bfloat16")
        assert cfg.enabled_passes() == passes, name


def test_ablation_presets_keep_decoder_in_model_dtype():
    for name in ("s2", "s3"):
        cfg = OptimizerConfig.from_preset(name, model_dtype="bfloat16")
        assert cfg.effective_decoder_dtype() == torch.bfloat16


def test_final_preset_moves_decoder_to_fp16():
    cfg = OptimizerConfig.from_preset("v100_sm70", model_dtype="bfloat16")
    assert cfg.effective_decoder_dtype() == torch.float16
