# volta_t2iopt

**在 pre-Ampere GPU（Volta / Turing）上，把 bf16-native 的文生图扩散模型安全迁到 fp16 张量核。**

不是又一个采样器封装：这个库只解决一件事 —— **模型的原生精度和这张卡的张量核不匹配**。
它先探硬件，再按能力选 pass，最后用质量门限验收；每一步都能回滚、都会说明自己为什么被启用或跳过。

```
Tesla V100 (sm_70) 上，Z-Image-Turbo 6B DiT，1024² / 9 步：
bf16 基线 228.4 s  →  41.7 s   （5.40×，SSIM 对 fp32 真值 0.968 → 0.982）
```

## 为什么需要它

V100 慢不是因为模型大，而是因为 **Volta 只有 fp16 张量核，没有 bf16 张量核**（bf16 从 Ampere/sm_80 才有）：

| | 实测（V100，DiT 真实形状） |
|---|---|
| fp16 GEMM（fp32 累加） | **20–21 TFLOPS**（张量核） |
| bf16 GEMM | **2.4 TFLOPS**（torch 回退到 fp32 SIMT） |

于是 bf16-native 的现代 DiT 在这张卡上凭空损失 9 倍算力。而"直接转 fp16"会炸：注意力的输出投影
（量级 ~400 × 3840 维累加）超过 65504 → inf → nan。**这是范围不够，不是精度瑕疵，调不了**。
本库的做法是逐层、逐次调用地在线定一个 2 的幂缩放因子，让 fp16 永远不越界，且数值无损。

## 安装

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e .            # 或 pip install -e .
pytest -q                      # 62 个 CPU-only 用例（含能力决策、pass 幂等/回滚、门限）
```

> torch 要 2.8.0+cu128：**CUDA 13 / torch 2.9+ 的 wheel 已移除 Volta**，装上去 sm_70 只剩 PTX JIT。

## 五分钟

```bash
volta-t2iopt probe --measure          # 这张卡缺什么？→ 推导出该开哪些 pass
volta-t2iopt cache all                # 一次性：bf16 权重 + 文本条件缓存（跳过 7.5 GB 的 TE）
volta-t2iopt bench --size 1024 --steps 9 --runs 2     # 优化档，端到端计时
volta-t2iopt bench --preset noop --size 1024 --steps 9  # 基线（A/B）
volta-t2iopt validate --size 1024 --steps 9 --baseline results/val_..._noop_1024_9step
volta-t2iopt profile --size 1024      # kernel 归因 + 逐层激活量级 + 干净墙钟
volta-t2iopt report                   # 汇总已落盘的结果
```

代码里用：

```python
import volta_t2iopt as vt

print(vt.probe(measure=True).summary())            # 能力探针：缺什么、实测多少

pipe, sess = vt.load_optimized("z-image-turbo", preset="auto")   # 加载 + 探针 + 打补丁
print(sess.summary())                            # 每个 pass 的决策与理由
res = vt.generate(pipe, sess.adapter, prompt="portrait", size=1024, steps=9)
print(res.timing.summary_line())
sess.revert()                                    # 一键回到基线
```

`preset="auto"` 会按探针推导：sm_70 + bf16-native → 三个 pass 全开；A100 + bf16-native → 全关
（再迁移是净损失）。所有决策都写进 `sess.summary()`，不是静默加速比。

## 它做了什么（三个 pass）

| pass | 解决什么 | 实测 | 什么时候自动关 |
|---|---|---|---|
| `hybrid_linear` | bf16 GEMM 没张量核 → 换 fp16 权重 + **在线自适应 2 的幂预缩放** + fp32 累加 | 每 forward 24.9 s → 7.9 s | 卡有 bf16 张量核 / 模型本来就是 fp16 |
| `attn_fp16` | sm_70 的 mem-efficient 后端只有 **f16** 版 cutlass fmha，bf16 会掉 math 后端（还转 fp32 走 SIMT） | (1,30,4160,128)：137.5 → **30.1 ms** | 同上 |
| `decoder_fp16` | VAE 解码是纯带宽活，cudnn 有原生 fp16 kernel | 1024² 解码 5.26 → **1.02 s**（SSIM 0.99966） | decoder 目标精度与模型一致 |

注意第一个 pass 的**关键设计**：k 不能在加载时定死。同一个 `noise_refiner.1.feed_forward.w2`，
输入 |max| 在 1024² 下是 5824、在 512² 下是 **25472**，而且同一 run 里后续 step 还会涨。
所以 k 由 `|x|max · rowsum(W)` 在 GPU 上现算（一次 `amax` 归约，不回 host、不产生同步）——
用严格上界换取永不溢出。

## 组织方式

```
volta_t2iopt/
├── capability.py     # 硬件探针：sm / 有无 bf16 张量核 / 实测 TFLOPS → 唯一读硬件的地方
├── config.py         # OptimizerConfig + preset 解析 + env 覆盖 + auto 推导（带理由）
├── presets/          # v100_sm70（=final）| s3 | s2 | noop（A/B 用）
├── passes/           # base（契约）+ hybrid_linear | attn_fp16 | decoder_fp16
├── models/           # zimage | sdxl | generic —— 新模型只加一个文件，pass 层不动
├── runtime/          # loader / cache / timing（CUDA event 口径）/ generate
├── analysis/         # kernel 分类、逐层激活统计、profiler、报表
├── validate/         # metrics（SSIM/relL2）+ gate（PASS/FAIL）
├── benchmarks/       # 分项证据：gemm / prescale / attention / vae / gemm-error
└── cli/              # probe bench validate profile matrix report cache microbench
```

四层单向依赖：`capability → config → passes → runtime`。
pass 之间不互相调用，只认 `Context`；`session` 是唯一知道顺序、失败降级、回滚的地方。

## 质量与口径

- **质量结论一律对 fp32 真值**：最终版 SSIM **0.982**，bf16 基线 0.968（fp16 尾数 10 位 > bf16 的 8 位）。
  对 *bf16 基线* 比 9 步图是 0.978 —— 那不是回归，只是两条数值路径本来就会分叉。
- **latent relL2 只作诊断**：1 次 forward 的 bf16 vs fp16 relL2 ≈ 2.7e-2，9 步 ≈ 8e-2。
  扩散采样会放大每步的微小差异，它不是"实现错没错"的判据（那个判据在单层 GEMM，量级 1e-4）。
- **VAE 必须直接包住 `vae.decode` 计时**：早期用"总数 − 每步间隔"估算，把 VAE 记成 30.5 s
  （真实 5.26 s），因为 callback 只覆盖 forward *之间*，最后一次 forward 漏进了 VAE 桶。
- **profiler 会伪造空转**：同一档位要同时记"挂 profiler"和"干净"两个 forward 时间。
  S3 档曾显示 16.5% 空转，去掉 profiler 后 kernel 4.56 s ≈ 整 forward 4.58 s —— CUDA Graph 因此被否决。

## 复现质量与性能

```bash
# 优化档 + 基线（两条命令，各自一个进程，避免显存状态互相污染）
volta-t2iopt validate --preset noop      --size 1024 --steps 9 --tag base
volta-t2iopt validate --preset v100_sm70 --size 1024 --steps 9 --tag opt \
  --baseline results/val_z-image-turbo_noop_1024_9step_base

# fp32 真值（Z-Image 原始 F32 存储；1024² 放不进 32 GB，用 512²/2 步）
volta-t2iopt validate --preset noop --path /path/to/z-image-turbo \
  --size 512 --steps 2 --tag f32ref

# 4 档 × 3 分辨率的 profile 矩阵（每格一个子进程）
volta-t2iopt matrix --presets noop s2 s3 v100_sm70 --sizes 512 768 1024 --steps 2
```

本次重构后用本库重跑（V100-32GB，2026-11-13）：
`1024²/9 步 41.72 s | 4537 ms/forward | VAE 0.89 s | 峰值 14.13 GiB` —— 与脚本时代的 42.3 s / 4587 ms / 14.1 GiB 一致。
512²/2 步 A/B：`6002 ms/fwd + VAE 1.29 s`（noop）→ `1099 ms/fwd + VAE 0.21 s`（v100_sm70）。

## 文档

- [`docs/EVIDENCE.md`](docs/EVIDENCE.md) —— 实测记录：V100 上现代文生图模型能不能用、选型结论、踩过的坑
- [`docs/OPTIMIZATION_PLAN.md`](docs/OPTIMIZATION_PLAN.md) —— 完整分阶段方案与逐项实测账（含被实测否决的选项）
- [`docs/PROFILE_RESULTS.md`](docs/PROFILE_RESULTS.md) —— profile 快照：4 档 × 3 分辨率 × 逐 kernel / 逐层 / 微基准
- [`docs/DESIGN.md`](docs/DESIGN.md) —— 架构、pass 契约、新增模型/pass 的步骤
- [`docs/MIGRATION.md`](docs/MIGRATION.md) —— 从脚本集合重构到本库：命令对照、修掉的 3 处口径问题

## 边界（诚实清单）

- **只在 V100（sm_70）上做过真机验证**；Ampere+ 的逻辑（自动全关）由单元测试覆盖，没跑过真模型。
- 非 Volta 卡、非 diffusers pipeline、非 `nn.Linear` 结构的模型（如融合 kernel 的自定义实现）不适用。
- 未实现：Triton 混合 GEMM（方案里的 M2，预期再省 0.6–0.8 s/forward）、元素级融合（S5）。
- CUDA Graph 已实测否决：真实空转 ≈ 0，收益上限不存在。
- `results/` 是本地产物目录（图 / JSON / chrome trace），**不入库**。
