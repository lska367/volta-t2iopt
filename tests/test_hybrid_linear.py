"""HybridLinear 的数值契约：不溢出、不丢精度、比 bf16 更准。

这些用例是"预缩放为什么必须在线算"的可执行证据 —— 静态 k 表在放大输入下直接失败。
"""

from __future__ import annotations

import torch

from volta_t2iopt.passes.hybrid_linear import (HybridLinear, required_k, replace_linears)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item() / b.float().norm().item()


def test_prescale_prevents_overflow_where_raw_fp16_fails():
    torch.manual_seed(0)
    k, n = 3840, 64
    x = (torch.randn(n, k) * 2e5).bfloat16()          # 输本身超过 fp16 上限
    w = (torch.randn(128, k) * 0.02).bfloat16()
    ref = x.float() @ w.float().T

    raw = (x.half() @ w.half().T).float()
    assert not torch.isfinite(raw).all() or raw.abs().max() > 65504

    layer = HybridLinear(w, None)
    out = layer(x)
    assert torch.isfinite(out).all()
    assert _rel(out, ref) < 1e-2
    # 上界保证：max|x| * rowsum / 2^k <= out_limit
    bound = x.abs().max().item() * layer.rowsum / 2 ** required_k(
        x.abs().max().item(), layer.rowsum, layer.out_limit)
    assert bound <= layer.out_limit * 1.0001


def test_prescale_scaling_is_numerically_free():
    """缩放 2^-k 只动指数不动尾数：k>0 与 k=0 的相对误差应当同量级。

    （"最终版比 bf16 基线更准"是端到端结论 —— 用 `validate --fp32-ref` 验证，
    不能在单层尺度上断言：单层误差主要由输出写回 bf16 的那次舍入决定。）
    """
    torch.manual_seed(3)
    w = (torch.randn(128, 512) * 0.02).bfloat16()
    layer = HybridLinear(w, None)

    x_small = torch.randn(64, 512).bfloat16()                  # 不需要缩放
    x_big = (torch.randn(64, 512) * 2e4).bfloat16()           # 必须缩放
    k_small = required_k(x_small.abs().max().item(), layer.rowsum)
    k_big = required_k(x_big.abs().max().item(), layer.rowsum)
    assert k_small == 0 and k_big > 0

    err_small = _rel(layer(x_small), x_small.float() @ w.float().T)
    out_big = layer(x_big)
    err_big = _rel(out_big, x_big.float() @ w.float().T)
    assert torch.isfinite(out_big).all()
    assert err_big < 3 * err_small          # 同量级，不是"缩放带来精度损失"


def test_bias_is_applied_in_fp32():
    x = torch.randn(32, 256).bfloat16()
    w = torch.randn(64, 256).bfloat16() * 0.02
    b = torch.randn(64)
    out = HybridLinear(w, b)(x)
    ref = x.float() @ w.float().T + b
    assert _rel(out, ref) < 1e-2


def test_required_k_is_monotone_and_zero_when_safe():
    assert required_k(0.0, 100.0) == 0
    assert required_k(1.0, 10.0) == 0
    ks = [required_k(a, 100.0) for a in (1.0, 1e2, 1e4, 1e6)]
    assert ks == sorted(ks)
    assert ks[-1] > 0


def test_no_prescale_mode_matches_plain_fp16():
    torch.manual_seed(2)
    x = torch.randn(16, 128).bfloat16()
    w = torch.randn(64, 128).bfloat16() * 0.02
    a = HybridLinear(w, None, prescale=False)(x)
    b = (x.half() @ w.half().T).float().bfloat16()
    assert torch.allclose(a, b, atol=1e-3)


def test_replace_linears_is_reversible():
    model = torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(),
                                torch.nn.Linear(32, 8))
    original = [m for m in model if isinstance(m, torch.nn.Linear)]
    undo, skipped = replace_linears(model, prefix="m", keep_originals=True)
    assert skipped == []
    assert sum(isinstance(m, HybridLinear) for m in model.modules()) == 2
    for parent, name, mod in undo:
        setattr(parent, name, mod)
    assert sum(isinstance(m, HybridLinear) for m in model.modules()) == 0
    assert [id(m) for m in model if isinstance(m, torch.nn.Linear)] == [id(m) for m in original]


def test_replace_linears_drops_originals_by_default():
    """默认不保留原模块（bf16 权重 11.7 GiB，留着回滚会让峰值显存翻倍）。"""
    model = torch.nn.Sequential(torch.nn.Linear(16, 8))
    w_before = model[0].weight.detach().clone()
    undo, _ = replace_linears(model)
    assert undo[0][2] is None
    # 用 fp16 权重重建，数值等价到 bf16 舍入
    hyb = model[0]
    lin = torch.nn.Linear(hyb.w.shape[1], hyb.w.shape[0])
    lin.weight.data = hyb.w.detach().to(torch.bfloat16)
    assert torch.allclose(lin.weight.data.float(), w_before.float(), atol=1e-2)


def test_unsafe_weight_is_skipped_not_crashing():
    model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(70000.0)   # 超出 fp16 上限
    undo, skipped = replace_linears(model)
    assert undo == [] and len(skipped) == 1
