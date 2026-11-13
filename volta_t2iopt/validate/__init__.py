"""验收层：质量指标 + 门限判定。"""

from .gate import GateCriteria, GateResult, check_gate, compare_files, compare_results
from .metrics import image_stats, psnr, rel_l2, ssim

__all__ = ["GateCriteria", "GateResult", "check_gate", "compare_files", "compare_results",
           "image_stats", "psnr", "rel_l2", "ssim"]
