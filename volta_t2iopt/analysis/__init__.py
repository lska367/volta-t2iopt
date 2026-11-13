"""分析层：把 chrome trace / 逐层统计读成可以下的结论。"""

from .layer_stats import LinearStats, format_layer_table, recommend_k
from .profiler import profile_run
from .report import render_markdown_table, suite_table
from .trace import analyze_trace, classify, format_kernel_stats, kernel_stats, load_kernels

__all__ = ["LinearStats", "format_layer_table", "recommend_k", "profile_run",
           "render_markdown_table", "suite_table", "analyze_trace", "classify",
           "format_kernel_stats", "kernel_stats", "load_kernels"]
