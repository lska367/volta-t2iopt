"""volta_t2iopt —— 在 pre-Ampere GPU 上把 bf16-native 扩散模型安全地迁到 fp16 张量核。

核心事实（本项目全部结论的来源）：Volta/Turing 只有 fp16 张量核，没有 bf16 张量核，
所以 bf16-native 的 DiT 在这些卡上会掉进 fp32 SIMT 回退（实测 bf16 2.4 TFLOPS vs fp16 20 TFLOPS）。
本库做的事：探测硬件 → 按策略把 GEMM / attention / decoder 迁到 fp16，
并用「在线自适应 2 的幂预缩放」保证不溢出，最后用质量 gate 验收。

典型用法：

    import volta_t2iopt as vt

    pipe, sess = vt.load_optimized("z-image-turbo", preset="auto")
    print(sess.summary())                       # 每个 pass 为什么开 / 为什么跳
    res = vt.generate(pipe, sess.adapter, prompt="portrait", size=1024, steps=9)

自带 pipeline 也行：with vt.optimize(pipe) as sess: ...（auto 走能力探针）

不想用上下文管理器时：sess = vt.optimize(pipe); sess.apply(); ...; sess.revert()
"""

from .capability import Capability, probe
from .config import OptimizerConfig, PassConfig
from .models import ModelAdapter, detect_adapter, get_adapter
from .runtime.generate import RunResult, generate
from .runtime.loader import load_optimized, load_pipe
from .session import OptimizationSession, optimize
from .validate.gate import GateCriteria, GateResult, check_gate
from .validate.metrics import rel_l2, ssim

__version__ = "0.1.0"

__all__ = [
    "Capability", "probe",
    "OptimizerConfig", "PassConfig",
    "ModelAdapter", "detect_adapter", "get_adapter",
    "RunResult", "generate", "load_pipe", "load_optimized",
    "OptimizationSession", "optimize",
    "GateCriteria", "GateResult", "check_gate",
    "rel_l2", "ssim",
    "__version__",
]
