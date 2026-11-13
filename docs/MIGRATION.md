# 从脚本集合到库：迁移记录（2026-11-13）

原来 `t2i/` 下是 17 个脚本，优化逻辑以三种形式散落：

- **策略写在环境变量里**：`T2I_HYBRID=0` / `T2I_FP16_ATTN=0` 藏在算子的 `import` 之后；
- **档位定义写在某个脚本的字典里**：`profile_suite.py` 的 `VARIANTS`；
- **同一个函数复制三份**：`classify()` 在 `analyze_trace.py` 和 `profile_zimage.py` 各一份，
  `kernel_stats` 在两处同构实现。

重构后：逻辑进库（`volta_t2iopt/`），脚本要么变 CLI 子命令，要么作为证据留在 `benchmarks/`。
`t2i/` 目录已删除（历史提交里还能找到）。

## 命令对照

| 旧 | 新 |
|---|---|
| `python t2i/build_cache.py` | `volta-t2iopt cache all` |
| `python t2i/bench_zimage.py --dtype fp16 --size 1024` | `volta-t2iopt bench --preset noop --path <fp16 权重>` |
| `python t2i/sweep_zimage.py --dtype bf16 --sizes 1024 768 --steps 9` | `volta-t2iopt bench --preset noop --size 1024 --steps 9`（逐组合跑） |
| `python t2i/sweep_sdxl.py` | `volta-t2iopt bench --model sdxl-turbo` |
| `python t2i/validate.py --mode hybrid --size 1024 --steps 9 --tag opt` | `volta-t2iopt validate --preset v100_sm70 --size 1024 --steps 9 --tag opt` |
| `python t2i/validate.py --compare A B` | `volta-t2iopt validate ... --baseline A`（判定逻辑进 `validate/gate.py`） |
| `python t2i/profile_zimage.py --hybrid` | `volta-t2iopt profile --preset v100_sm70 --size 1024` |
| `python t2i/profile_suite.py --variant final --size 1024` | `volta-t2iopt profile --preset v100_sm70 --size 1024` |
| `python t2i/profile_suite.py --report` | `volta-t2iopt report --what profile` |
| `python t2i/analyze_trace.py results/trace_1024.json --top 25` | `volta-t2iopt report --what trace --dir results` |
| `python t2i/microbench_gemm.py` | `volta-t2iopt microbench gemm` |
| `python t2i/check_prescale.py` | `volta-t2iopt microbench prescale` |
| `python t2i/check_gemm_error.py` | `volta-t2iopt microbench gemm-error` |
| `python t2i/check_vae.py --latent X` | `volta-t2iopt microbench vae --latent X` |
| `T2I_HYBRID=0 T2I_FP16_ATTN=0 ...` | `--preset noop`，或 `VOLTA_T2IOPT_DISABLE=hybrid_linear,attn_fp16` |

档位也从字典变成了文件：`presets/{noop,s2,s3,v100_sm70}.toml`，可以 diff、可以拷到别的机器上跑对照。

## 重构时修掉的 3 个口径问题

这三个都不是"搬家弄坏了"，而是搬家过程中被测试/复跑抓出来的**原有问题**。

### 1. `classify()` 的顺序：fmha 被算进了 GEMM 桶

`_GEMM` 正则里含 `cutlass` 和 `sm70`，而 sm_70 的 attention kernel 名字是
`fmha_cutlassF_f16_aligned_32x128_rf_sm70` —— 两个特征都命中，而旧实现**先判 GEMM**。
用历史 trace 复核（`results/traces/trace_final_1024.json.gz`，kernel 总时间 4.60 s）：

| 口径 | gemm | attn | elt |
|---|---|---|---|
| 旧（先 GEMM） | 3.36 s | —（被并进去） | 1.23 s |
| 新（先 attention） | **2.35 s** | **1.02 s** | 1.23 s |

修正后的分桶与文档里手写的结论（gemm 2338 + fmha 1017 + elt 1233 ms）一致。
bf16 档不受影响：它的 attention 走 math 后端，本来就是 `volta_sgemm`，归 gemm 是对的。

### 2. latent `relL2 < 2e-3` 这条门限从来没成立过

`OPTIMIZATION_PLAN.md` §4 把 latent relL2 2e-3 写成验收线。用历史产物一算：

| 对比 | 1 次 forward | 9 步 |
|---|---|---|
| bf16 vs fp16-hybrid latent relL2 | 2.7e-2 | 8.1e-2 |

扩散采样会把每步微小差异放大，所以 latent 尺度的 relL2 只能当**诊断**。
2e-3 那个量级真正成立的地方是**单层 GEMM**（`microbench gemm-error`：fp16 路径 2e-4，bf16 路径 1.7e-3）。
现在 `GateCriteria.max_rel_l2` 默认 `None`（只报不判），要卡就显式传。

### 3. 保留原权重做回滚，会把峰值显存翻倍

第一版 `revert()` 依赖"保留被替换掉的 `nn.Linear`"。在 V100 上实测：
512²/2 步峰值从 23.9 GiB → **12.3 GiB**（改成默认不保留之后），因为 11.7 GiB 的 bf16
transformer 和不保留时释放掉的权重同时驻留。
现在默认不保留，`revert()` 用 fp16 权重重建 `nn.Linear`（等价到 bf16 舍入）；需要精确回滚传 `keep_originals`。

## 删除的脚本与原因

| 脚本 | 原因 |
|---|---|
| `bench_hybrid.py` | 引用 `hybrid_linear.load_k_map` —— 那是"静态 k 表"版本的残留，函数在方案改成在线预缩放时就删了，脚本早已失效 |
| `sweep_zimage.py` / `sweep_sdxl.py` | 被 `bench` + preset 覆盖（多分辨率循环留给 shell/CLI） |
| `bench_zimage.py` | 被 `bench` 覆盖（分阶段计时口径已进 `runtime/timing.py`） |
| `profile_zimage.py` / `profile_suite.py` / `analyze_trace.py` | 合并为 `analysis/profiler.py` + `analysis/trace.py` + `cli/{profile,matrix}.py`，`classify`/`kernel_stats` 去重 |
| `build_cache.py` / `loader.py` / `prompts.py` | 进 `runtime/cache.py`、`runtime/loader.py`、`prompts.py`（缓存支持旧文件名，迁移期不用重建） |
| `check_*.py` / `microbench_gemm.py` | 进 `benchmarks/`，成为可被 CLI 调用的证据脚本 |
| `validate.py` | 拆成 `validate/metrics.py`（指标）+ `validate/gate.py`（门限）+ `runtime/generate.py`（跑与落盘） |

## 复跑对照（证明重构没有改变行为）

同一台 V100，Z-Image-Turbo：

| | 脚本时代 | 重构后（本库） |
|---|---|---|
| 1024²/9 步总时间 | 42.3 s | **41.72 s** |
| 每 forward | 4587 ms | **4537 ms** |
| VAE 解码 | 1.02 s | **0.89 s** |
| 峰值显存 | 14.1 GiB | **14.13 GiB** |
| SSIM（对 fp32，512²/2 步） | 0.982 | **0.9822** |
| SSIM（bf16 基线对 fp32） | 0.968 | **0.9681** |

指标模块换了实现但算法一致（SSIM 用同一套 11×11 均值滤波），所以历史数字可直接对比。
