"""分析层与计时口径。"""

from __future__ import annotations

import json

import pytest
import torch

from volta_t2iopt.analysis.layer_stats import LinearStats, recommend_k
from volta_t2iopt.analysis.trace import classify, kernel_stats
from volta_t2iopt.runtime.timing import StageTiming, StepRecorder


def test_kernel_classification():
    assert classify("cutlass::gemm::device::Gemm") == "gemm"
    assert classify("fmha_cutlassF_f16_aligned_32x128_rf_sm70") == "attn"
    assert classify("void at::native::elementwise_kernel<128>") == "elt"
    assert classify("something_unknown") == "other"


def test_kernel_stats_from_fake_trace(tmp_path):
    trace = {"traceEvents": [
        {"ph": "X", "cat": "kernel", "name": "cutlass_gemm", "ts": 0, "dur": 1000},
        {"ph": "X", "cat": "kernel", "name": "fmha_f16", "ts": 1200, "dur": 500},
        {"ph": "X", "cat": "cpu_op", "name": "aten::mul", "ts": 0, "dur": 9999},
    ]}
    p = tmp_path / "t.json"
    p.write_text(json.dumps(trace))
    s = kernel_stats(p)
    assert s["kernel_ms"] == 1.5
    assert s["launches"] == 2
    # 时间轴跨度 1700 µs、kernel 1500 µs → 空转 200/1700
    assert s["idle_pct"] == pytest.approx(11.8, abs=0.2)
    assert s["cat_ms"]["gemm"] == 1.0


def test_kernel_stats_supports_gzip(tmp_path):
    import gzip

    trace = {"traceEvents": [{"ph": "X", "cat": "kernel", "name": "x", "ts": 0, "dur": 10}]}
    p = tmp_path / "t.json.gz"
    with gzip.open(p, "wt") as f:
        f.write(json.dumps(trace))
    assert kernel_stats(p)["launches"] == 1


def test_linear_stats_records_scalars():
    model = torch.nn.Sequential(torch.nn.Linear(16, 8))
    stats = LinearStats()
    assert stats.attach(model, "denoiser") == 1
    model(torch.randn(4, 16))
    assert len(stats.rows) == 1
    row = stats.rows[0]
    assert row["dim_in"] == 16 and row["dim_out"] == 8 and row["m"] == 4
    assert row["out_fp16_ok"] is True


def test_recommend_k_marks_overflowing_layers():
    rows = [dict(name="a", x_max=1.0, out_max=1.0),
            dict(name="b", x_max=208896.0, out_max=7.8e5)]
    out = recommend_k(rows)
    assert out[0]["k_static"] == 0
    assert out[1]["k_static"] > 0


def test_stage_timing_derived_fields():
    t = StageTiming(total_ms=9000, denoise_ms=8000, vae_ms=1000, forwards=8, peak_gib=17.5)
    assert t.ms_per_fwd == 1000.0
    assert t.vae_pct == pytest.approx(11.111, abs=1e-2)
    assert "9.00 s" in t.summary_line()
    assert t.to_dict()["forwards"] == 8


def test_step_recorder_intervals_and_latent_capture():
    rec = StepRecorder(capture_latents_at=2).start()
    lat = torch.ones(1, 2)
    for step in range(3):
        rec(pipe=None, step=step, timestep=step, kwargs={"latents": lat * (step + 1)})
    assert rec.calls == 3
    assert len(rec.intervals()) == 3
    assert rec.latents is not None and float(rec.latents[0, 0]) == 3.0
