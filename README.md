# t2i_deploy_opt — V100 上文生图模型的部署优化

从 `~/work/cuda_learning` 分出来的独立工作目录（2026-11-13），只收纳**文生图（DiT/UNet）推理部署与优化**相关的内容，
与仓库里 LLM 推理算子 bench 的部分互不依赖、可独立演进。

环境：Tesla V100-PCIE-32GB（sm_70, 32 GB）、CUDA 12.9、torch 2.8.0+cu128、diffusers 0.42.1。
模型缓存在 `/home/lskam/models/{z-image-turbo,sdxl-turbo}`（与本目录无关，不进仓库）。

## 做了什么

被测模型：**Z-Image-Turbo（6B DiT，bf16-native）** 与对照 **SDXL-turbo（fp16-native UNet）**。
核心结论：慢的根因不是模型大，而是 **bf16-native 在 sm_70 上没有张量核**（bf16 GEMM 2.5 TFLOPS vs fp16 20 TFLOPS）。

1024² / 9 步端到端：

| 阶段 | 耗时 | 倍数 | 做了什么 |
|---|---|---|---|
| 基线 bf16（V100 上是 fp32 模拟） | 228.4 s | 1.00× | — |
| S2 | 76.7 s | 2.98× | `HybridLinear`：fp16 权重 × fp16 输入 → fp32 累加，输入侧在线自适应 2 的幂预缩放 |
| S3 | 46.4 s | 4.92× | SDPA 输入转 fp16 → 命中 `fmha_cutlassF_f16_sm70`（bf16 没有对应内核） |
| S4 | **42.3 s** | **5.40×** | VAE 解码 fp16（5.26 → 1.02 s，SSIM 0.99966） |

质量：对 fp32 真值 SSIM **0.982**，优于 bf16 基线的 0.968（fp16 尾数 10 位 vs bf16 8 位）。

## 目录

```
t2i/
├── README.md             # 入口：选型结论 + 优化记录（先读这个）
├── OPTIMIZATION_PLAN.md  # 完整方案与实测记录（S1–S5 分阶段）
├── PROFILE_RESULTS.md    # profile 快照：4 档 × 3 分辨率 × 逐 kernel / 逐层 / 微基准
├── hybrid_linear.py      # S2：fp16 GEMM + fp32 累加 + 在线预缩放
├── fp16_attention.py     # S3：SDPA 输入 fp16（含 sys.modules 全引用替换）
├── validate.py           # 质量对拍（对 fp32 真值的 relL2 / SSIM）
├── bench_zimage.py / sweep_zimage.py / sweep_sdxl.py   # 端到端计时
├── profile_zimage.py / profile_suite.py / analyze_trace.py  # 逐 kernel 归因
├── microbench_gemm.py / check_gemm_error.py / check_prescale.py / check_vae.py  # 分项微基准
├── build_cache.py / loader.py / prompts.py
└── out/                  # 产物（图 + 汇总 JSON + chrome trace gz，约 44 MB）
```

## 复现（在仓库根目录执行，脚本里按 `t2i.*` 导入，所以必须保留这一层）

```bash
python t2i/build_cache.py                       # 一次性：bf16 权重缓存 + prompt embeds
python t2i/sweep_zimage.py --dtype bf16 --sizes 1024 768 --steps 9   # 基线
python t2i/validate.py --mode hybrid --size 1024 --steps 9 --tag opt # 优化档 + 质量对拍
python t2i/profile_suite.py                     # 4 档 × 3 分辨率 profile 采集
```

环境开关：`T2I_HYBRID=0` / `T2I_FP16_ATTN=0` 可一键回退到基线。

## 环境

见 `requirements.txt`（torch 2.8.0 的 cu128 wheel 仍含 sm_70；不要升到 CUDA 13 / torch 2.9+，Volta 已被移除）。
