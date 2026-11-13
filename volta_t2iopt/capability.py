"""硬件能力探针：本库所有"要不要迁移"的判断只依赖这一层。

为什么需要一个探针而不是固定的 if：

  - "慢"不是模型属性，是 **模型原生精度 × 卡的张量核能力** 的组合。
    Volta/Turing：fp16 张量核有、bf16 张量核没有（sm < 80）→ bf16-native 模型掉进 fp32 SIMT。
    Ampere/Ada/Hopper：两种都有 → 什么都不用做，"优化"反而是负收益。
  - 同一张卡上还有第二层差异：attention。sm_70 的 mem-efficient 后端只有 **f16** 的
    cutlass fmha 内核，bf16 输入直接 fallback 到 math 后端（还会转 fp32 走 SIMT GEMM）。
  - 结论必须能被实测复核，所以 `measure=True` 会真跑两组 GEMM + SDPA 把 TFLOPS/ms 记下来，
    和静态判断放在同一份报告里 —— 两者不一致时以实测为准，并在 `notes` 里标出来。

用法：

    cap = probe()                       # 只读设备属性（毫秒级）
    cap = probe(measure=True)           # 额外跑微基准（约 1-2 s）
    print(cap.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

# sm → 架构家族。只用于展示，不做判断依据（判断一律用张量核能力）。
_FAMILY = {
    60: "Pascal", 61: "Pascal", 62: "Pascal",
    70: "Volta", 72: "Volta", 75: "Turing",
    80: "Ampere", 86: "Ampere", 87: "Ampere", 89: "Ada", 90: "Hopper", 100: "Blackwell",
}

# fp16 张量核从 Volta(sm_70) 开始；bf16 张量核从 Ampere(sm_80) 开始。
_FP16_TC_MIN_SM = 70
_BF16_TC_MIN_SM = 80


@dataclass(frozen=True)
class Capability:
    """一张卡的精度能力快照（值对象，构造后不改）。"""

    device_name: str
    sm: int
    memory_gib: float
    torch_version: str
    arch_list: tuple[str, ...] = ()
    measured: dict = field(default_factory=dict, compare=False)

    # --- 张量核能力 ---
    @property
    def family(self) -> str:
        return _FAMILY.get(self.sm, f"sm_{self.sm}")

    @property
    def fp16_tensorcore(self) -> bool:
        return self.sm >= _FP16_TC_MIN_SM

    @property
    def bf16_tensorcore(self) -> bool:
        return self.sm >= _BF16_TC_MIN_SM

    @property
    def has_kernel_for_arch(self) -> bool:
        """torch 的预编译 kernel 里是否有本卡的 cubin（CUDA 13 已移除 Volta）。"""
        if not self.arch_list:
            return True
        return f"sm_{self.sm}" in self.arch_list

    # --- 实测数值（只有 measure=True 才有）---
    @property
    def measured_ratio(self) -> float | None:
        """bf16 / fp16 GEMM 的实测倍数。<1 表示 bf16 更慢。"""
        m = self.measured or {}
        fp16, bf16 = m.get("gemm_fp16_tflops"), m.get("gemm_bf16_tflops")
        if not fp16 or not bf16:
            return None
        return bf16 / fp16

    def summary(self) -> str:
        lines = [
            f"device      : {self.device_name} (sm_{self.sm} / {self.family}, "
            f"{self.memory_gib:.0f} GiB, torch {self.torch_version})",
            f"tensor core : fp16 {'yes' if self.fp16_tensorcore else 'NO'} | "
            f"bf16 {'yes' if self.bf16_tensorcore else 'NO'}",
        ]
        if self.arch_list and not self.has_kernel_for_arch:
            lines.append(f"WARNING     : torch arch_list 里没有 sm_{self.sm}，kernel 只能走 PTX JIT")
        if self.measured:
            m = self.measured
            if "gemm_fp16_tflops" in m:
                lines.append(f"gemm 实测   : fp16 {m['gemm_fp16_tflops']:.2f} TFLOPS | "
                             f"bf16 {m['gemm_bf16_tflops']:.2f} TFLOPS | "
                             f"bf16/fp16 = {self.measured_ratio:.2f}×")
            if "sdpa_fp16_ms" in m:
                lines.append(f"sdpa 实测   : fp16 {m['sdpa_fp16_ms']:.1f} ms | "
                             f"bf16 {m['sdpa_bf16_ms']:.1f} ms | "
                             f"kernel {m.get('sdpa_fp16_kernel', '?')}")
        return "\n".join(lines)


def probe(device: str | int = "cuda", measure: bool = False) -> Capability:
    """读设备属性；`measure=True` 时额外跑 GEMM/SDPA 微基准（有副作用：占显存、要同步）。"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备；本库的精度迁移只对真实 GPU 有意义")

    idx = device if isinstance(device, int) else torch.device(device).index or 0
    props = torch.cuda.get_device_properties(idx)
    try:
        arch_list = tuple(torch.cuda.get_arch_list())
    except Exception:  # pragma: no cover - 老版本 torch
        arch_list = ()

    cap = Capability(
        device_name=props.name,
        sm=props.major * 10 + props.minor,
        memory_gib=props.total_memory / 2**30,
        torch_version=torch.__version__,
        arch_list=arch_list,
    )
    if not measure:
        return cap
    return Capability(**{**cap.__dict__, "measured": measure_ops(idx)})


def measure_ops(device: int = 0) -> dict:
    """实测：DiT 真实形状的 GEMM（fp16 vs bf16）与一次 SDPA（fp16 vs bf16）。

    形状取自 1024² 的 Z-Image（dim=3840，tokens=64×64，heads=30，head_dim=128），
    这样测出来的倍数能直接和端到端预算对上，而不是一个抽象的 TFLOPS 数字。
    """
    out: dict = {}
    m, n, k = 4096, 3840, 3840
    flops = 2 * m * n * k

    def _bench(fn, reps: int = 5) -> float:
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return sorted(ts)[len(ts) // 2]

    with torch.no_grad():
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            x = torch.randn(m, k, device=device, dtype=dt)
            w = torch.randn(n, k, device=device, dtype=dt) * 0.02
            ms = _bench(lambda: x @ w.T)
            out[f"gemm_{name}_tflops"] = round(flops / (ms * 1e-3) / 1e12, 2)
            del x, w
            torch.cuda.empty_cache()

        # attention：sm_70 上 bf16 SDPA 会掉 math 后端，这里把耗时和 kernel 名一起记
        q = torch.randn(1, 30, 4160, 128, device=device, dtype=torch.float16)
        for name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
            qq, kk, vv = q.to(dt), q.to(dt), q.to(dt)
            ms = _bench(lambda: torch.nn.functional.scaled_dot_product_attention(qq, kk, vv), reps=3)
            out[f"sdpa_{name}_ms"] = round(ms, 1)
            del qq, kk, vv
            torch.cuda.empty_cache()
        out["sdpa_fp16_kernel"] = _sdpa_kernel_name(device)
    return out


def _sdpa_kernel_name(device: int = 0) -> str:
    """抓一次 fp16 SDPA 实际用的 kernel 名（判断有没有落进 cutlass fmha）。"""
    try:
        q = torch.randn(1, 4, 512, 64, device=device, dtype=torch.float16)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        for e in prof.key_averages():
            name = e.key
            if any(t in name.lower() for t in ("fmha", "flash", "attention", "sgemm", "softmax")):
                return name[:60]
    except Exception:  # pragma: no cover
        pass
    return "?"
