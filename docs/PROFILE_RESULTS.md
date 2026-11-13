# Z-Image-Turbo 完整 profile 快照（V100-PCIE-32GB / SM70）

> **历史文档（2026-11-12，脚本时代）**：结论与实测数字全部保留，只是代码形态变了 ——
> `t2i/*.py` 脚本已重构为 `volta_t2iopt` 库，命令对照见 [MIGRATION.md](MIGRATION.md)。
> 文中路径已更新：`t2i/out/` → `results/`（本地产物，不入库）。 各档位名（bf16/s2/s3/final）与 `presets/` 的对应见「档位定义」一节。

采集时间：2026-11-12 | 环境：torch 2.8.0+cu128、diffusers 0.42.1、CUDA 12.9、30 GB 内存
驱动：`volta-t2iopt matrix`（12 次采集，每格一个子进程）+ `volta-t2iopt microbench {gemm,attention,vae}`
原始数据：`results/suite_{variant}_{size}.json`（汇总，492 KB）
+ `results/traces/trace_{variant}_{size}.json.gz`（chrome trace，压缩后共 15 MB）

档位定义（也就是优化的演化线）。重构后对应 `presets/` 里的文件：

| 档位 | preset | fp16 张量核 GEMM | attention 进 fp16 | VAE 精度 |
|---|---|---|---|---|
| `bf16` | `noop` | ✗ | ✗ | bf16 |
| `s2` | `s2` | ✓ | ✗ | bf16 |
| `s3` | `s3` | ✓ | ✓ | bf16 |
| `final` | `v100_sm70` | ✓ | ✓ | fp16 |

---

## 0. 读这份数据前必须知道的三条口径

**① 只算 CUDA kernel，别用 `key_averages()`。**
它会把 `aten::*` 和它们调用的真 kernel 都算进去，重复计数。早期版本因此得出过
"单 forward CUDA 107 s"的假数（真实值 24.8 s）。正确做法是解析 chrome trace 里
`ph=X, cat=kernel` 的事件（`volta_t2iopt/analysis/trace.py`）。

**② profiler 会放大耗时和空转，所以每次都测两遍。**
同一次 run 里第 1 个 forward 挂 profiler、第 2 个不挂：

| 档位 @1024² | 挂 profiler | 干净墙钟 | 放大 |
|---|---|---|---|
| bf16 | 25421 ms | 24866 ms | +2.2% |
| final | 5072 ms | 4570 ms | +11.0% |

所以下面的"空转率"只用来看**有没有 CPU 侧瓶颈**，绝对时间一律以干净墙钟为准。

**③ 阶段拆分的坑（本次修正了一处历史错误）。**
`callback_on_step_end` 的间隔只覆盖 forward *之间*，最后一次 forward 会漏到"VAE"桶里。
旧 `sweep_bf16.json` 的 `denoise_s=196.91 / ms_per_fwd=24.613` 正好是 **8.0** 个间隔，
也就是说那 30.5 s 的 `vae_s` = 第 9 次 forward(24.6 s) + 真实 VAE(5.9 s)。
现在改成直接包住 `pipe.vae.decode` 计时（`validate.py` / `profile_suite.py` 均已修正）。

> **修正后的基线拆账（1024²/9 步，227.4 s）**：
> 9 × forward 221.5 s（97.4%） + VAE **5.26 s**（2.3%） + TE 0.33 s。
> 之前说的"VAE 占 13%、30.5 s、第二大单笔"是基于上面那个错误口径，作废。
> 这也解释了 S4 实测只省 4.1 s（46.4 → 42.3 s）：VAE 本来就没占那么多。

---

## 1. 单 forward 预算：4 档 × 3 分辨率

`kernel` 列之和 = 该 forward 的 GPU 忙碌时间；`干净墙钟` = 不挂 profiler 的一次 forward。

| 档位 | 分辨率 | token | kernel ms | gemm | attn | elt | 干净墙钟 ms | 发射次数 | 空转%(profiler) |
|---|---|---|---|---|---|---|---|---|---|
| bf16 | 1024 | 4096 | 24789.6 | 22678.1 | 550.8 | 1556.7 | **24866** | 7378 | 1.7 |
| bf16 | 768 | 2304 | 12910.3 | 11987.1 | 213.4 | 705.8 | 13082 | 7378 | 2.4 |
| bf16 | 512 | 1024 | 5923.9 | 5596.3 | 24.7 | 299.2 | 6020 | 7374 | 5.2 |
| s2 | 1024 | 4096 | 7972.3 | 5424.3 | 563.5 | 1982.3 | 7927 | 7247 | 5.1 |
| s2 | 768 | 2304 | 3547.9 | 2336.4 | 230.2 | 979.4 | 3527 | 7247 | 7.1 |
| s2 | 512 | 1024 | 1320.4 | 878.7 | 26.9 | 413.0 | 1322 | 7247 | 17.7 |
| s3 | 1024 | 4096 | 4599.7 | 3365.8† | — | 1231.8 | **4548** | 6907 | 5.4 |
| s3 | 768 | 2304 | 2441.3 | 1707.6† | — | 731.7 | 2430 | 6907 | 9.5 |
| s3 | 512 | 1024 | 1106.0 | 748.0† | — | 356.1 | 1111 | 6907 | 19.0 |
| final | 1024 | 4096 | 4598.5 | 3363.2† | — | 1233.1 | **4570** | 6907 | 5.1 |
| final | 768 | 2304 | 2445.1 | 1710.5† | — | 732.7 | 2434 | 6907 | 9.1 |
| final | 512 | 1024 | 1106.9 | 748.6† | — | 356.5 | 1104 | 6907 | 18.4 |

† s3/final 档的 attention 已进 `fmha_cutlassF_f16` 内核，被归类到 gemm（名字里没有 attention 关键字）。
按 kernel 名单独拆：`fmha` 1016 ms + `cutlass s884gemm` 2342 ms = 3358 ms。

**每档的 forward 加速比（1024²）**：bf16 → s2 **3.14×** → s3 **5.47×** → final **5.44×**（VAE 不影响 forward）。
**分辨率对加速比的影响**：1024² 3.14× / 768² 3.71× / 512² 4.55×（小分辨率下 GEMM 权重高，attention 占比低）。

**VAE 解码（同一 latent）**：bf16 5259 / 2873 / 1289 ms → fp16 1103 / 714 / 472 ms，
即 **4.77× / 4.02× / 2.73×**；峰值显存 17.62 → 14.12 GiB。

---

## 2. 瓶颈迁移：1024² 各档的 top kernel

### bf16（24866 ms forward）
| ms | 次数 | µs/次 | 类别 | kernel |
|---|---|---|---|---|
| 19741.7 | 241 | 81915.9 | gemm | `magma_sgemmEx_kernel<float, bf16, bf16>` ← bf16 无张量核，走 fp32 模拟 |
| 1456.9 | 30 | 48562.4 | gemm | `volta_sgemm_64x64_nn` ← attention 被 math 后端转 fp32 |
| 1299.0 | 30 | 43299.2 | gemm | `volta_sgemm_64x64_tn` ← 同上 |
| 518.1 | 30 | 17271.6 | attn | `cunn_SoftMaxForwardReg<float,float,float>` |
| 366.4 | 34 | 10775.5 | elt | `elementwise_kernel<128,2,gpu_kernel>`（RoPE 的 float 转换） |
| 186.1 | 342 | 544.1 | elt | `vectorized_elementwise_kernel<4, bfloat16_copy>` |

### s2（7927 ms，GEMM 换 fp16 张量核）
| ms | 次数 | µs/次 | 类别 | kernel |
|---|---|---|---|---|
| 1826.4 | 192 | 9512.7 | gemm | `cutlass_70_tensorop_f16_s884gemm` ← **我们的 GEMM** |
| 1452.8 | 30 | 48427.1 | gemm | `volta_sgemm_64x64_nn` ← attention 的 fp32 SIMT，**没动过** |
| 1404.5 | 30 | 46815.9 | gemm | `volta_sgemm_64x64_tn` |
| 545.3 | 32 | 17040.7 | gemm | `cutlass s884gemm`（第二组形状） |
| 530.3 | 30 | 17675.1 | attn | softmax |
| 365.6 | 34 | 10751.9 | elt | RoPE 的 float 转换 |

→ **attention 变成第一大头（2.86 s + 0.53 s = 43%）**，这是 S3 的触发依据。

### s3 / final（4570 ms）
| ms | 次数 | µs/次 | 类别 | kernel |
|---|---|---|---|---|
| 1793.7 | 192 | 9342.3 | gemm | `cutlass_70_tensorop_f16_s884gemm` |
| 1016.5 | 34 | 29898.1 | gemm | `fmha_cutlassF_f16_aligned_32x128_rf_sm70` ← attention，softmax 已融合 |
| 544.4 | 32 | 17014.0 | gemm | `cutlass s884gemm` |
| 324.9 | 481 | 675.5 | elt | `elementwise_kernel<128,4,gpu_kernel>` |
| 208.8 | 608 | 343.5 | elt | `elementwise_kernel<128,4,gpu_kernel>` |
| 207.6 | 549 | 378.2 | elt | `unrolled_elementwise_kernel<direct_copy>` |

发射次数从 7378 → 6907（省掉 softmax 和 q/k/v 的 fp32 转换）。

---

## 3. 逐层激活量级与预缩放需求（bf16 档，276 个 `nn.Linear`）

每个 Linear 都挂了 hook 记录 `|x|max` / `|out|max`。注意 `suite_bf16_*.json` 的
`linear` 数组有 **552 行 = 276 层 × 2 次 forward**（每次 forward 各记一行，两个 step 的量级不同），
下表已按层名去重取最大。

| 分辨率 | 层数 | 需要 k>0 | 输入本身超 fp16 上限(65504) | 最坏 \|out\|max |
|---|---|---|---|---|
| 1024 | 276 | 18 | 2 | 782336 |
| 768 | 276 | 20 | 2 | 753664 |
| 512 | 276 | 20 | 1 | 794624 |

最坏的 5 层（1024²）：

| 层 | M | \|x\|max | \|out\|max | k（静态表用） |
|---|---|---|---|---|
| `layers.0.feed_forward.w2` | 4160 | 208896 | 782336 | 4 |
| `noise_refiner.1.feed_forward.w2` | 4096 | 27008 | 288768 | 3 |
| `layers.14.feed_forward.w2` | 4160 | 65536 | 251904 | 3 |
| `noise_refiner.0.feed_forward.w2` | 4096 | 4256 | 240640 | 3 |
| `context_refiner.0.attention.to_out.0` | 64 | 206 | 94720 | 1 |

三个要点：
1. **溢出主力是 `feed_forward.w2`**（silu-gated 张量量级大），不是原计划假设的 `attention.to_out`。
2. **`|x|max` 自己就能到 208896**，超过 fp16 上限三倍 —— 所以预缩放**必须先乘再转**
   （`(x*s).half()` 对，`x.half()*s` 直接 inf）。
3. 需要缩放的层数在 3 个分辨率下分别为 18/20/20，**集合本身随分辨率变化**，
   这就是"静态 k 表"方案失败的直接证据（详见 `OPTIMIZATION_PLAN.md` §0.4）。
   实际实现用在线自适应，静态表只作为参考。

---

## 4. 微基准（单独跑，不依赖完整模型）

### GEMM：DiT 真实形状（M=4096）
| 精度路线 | qkv 4096×11520×3840 | to_out 4096×3840×3840 | ffn_in 4096×20480×3840 | ffn_out 4096×3840×10240 | 中位 |
|---|---|---|---|---|---|
| fp16 + fp32 累加 | 20.00 | 22.14 | 19.26 | 19.36 | **20.0 TF** |
| fp16 + fp16 累加 | 19.62 | 19.86 | 19.54 | 20.41 | 19.9 TF |
| bf16（原路径） | 2.52 | 2.45 | 2.52 | 2.52 | **2.5 TF** |
| fp32 SGEMM | 3.25 | 3.21 | 3.22 | 3.27 | 3.3 TF |

`nvidia-smi` 显示 `HW Power Brake Slowdown: Active` —— fp32 SIMT 只跑到标称 15.7 TF 的 **21%**，
而 fp16 张量核能到 ~28 TF 的 71%。**这就是"fp32 兜底方案"不可行的硬证据。**

### attention：(1,30,4160,128)
| 输入 | 时间 | kernel |
|---|---|---|
| bf16 + SDPA | 137.5 ms | math 后端 → 3× `direct_copy` + `volta_sgemm` + 独立 softmax |
| **fp16 + SDPA** | **30.1 ms** | `fmha_cutlassF_f16_aligned_32x128_rf_sm70` |
| fp16 手写 math | 42.8 ms | tensor core bmm + 独立 softmax |

sm_70 上 flash / cudnn 都不可用，mem-efficient 的 cutlass 内核**只有 f16 版本** —— 这是 bf16 慢的根因。

### VAE 解码（1024²，同一 latent）
| | bf16 | fp16 |
|---|---|---|
| 时间 | 5093 ms | **874 ms**（4.4~5.8×，视 warmup） |
| 与 bf16 的 SSIM | — | **0.99966**（max\|Δ\|=0.032） |
| inf/nan | 无 | 无 |

---

## 5. 逐 step 稳定性（final 档，1024²/9 步）

```
逐 step ms: [4581.0, 4615.7, 4568.5, 4554.0, 4573.7, 4572.6, 4576.7, 4576.3]
```

平坦 ±1%。虽然第 3 节的激活量级在后续 step 明显上涨（`w2` 输入从 65536 涨到 208896），
但**时间不受影响** —— 自适应预缩放每次现场定 k，不改变 kernel 形状。

端到端：**42.48 s** = 去噪 41.46 s（9 × 4607 ms） + VAE 1.02 s，峰值显存 14.1 GiB。

---

## 6. 结论：瓶颈迁移与剩余余量

```
bf16   : gemm 22.68 (91%)  ←── bf16 无张量核（其中 attention 的 fp32 GEMM 2.76 混在里面）
s2     : gemm 5.42 (68%)   + attention fp32 2.86  ←── 变成第一大头
s3/final: gemm 3.36 (73%)  + elt 1.23 (27%)     ←── 只剩元素级
```

1024² 最终 forward 4570 ms 的账：

| 项 | ms | 说明 |
|---|---|---|
| cutlass fp16 GEMM | 2338 | 276 个 Linear，~20 TFLOPS（V100 峰值的 71%） |
| cutlass fmha | 1017 | 34 次，4.4 TFLOPS 有效值，softmax 已融合 |
| 元素级 | 1233 | 约 4900 次发射；含我们的预缩放（1 次归约 + mul + unscale） |
| GPU 空转 | ~0 | 干净墙钟 4570 ≈ kernel 4599 |

**唯一还有肉的是元素级**：silu-gate 融合、adaLN 调制、RMSNorm+scale、残差加，
估计能省 0.6~0.8 s（S5）。GEMM 已接近张量核 roofline，attention 要更快得自写融合 kernel。
**CUDA Graph 不做**：真实空转 ≈ 0，profile 里那 5~19% 的空转是 profiler 注入的。

---

## 7. 复现

```bash
# 完整矩阵（4 档 × 3 分辨率，每格一个子进程，约 10 分钟）
volta-t2iopt matrix --presets noop s2 s3 v100_sm70 --sizes 512 768 1024 --steps 2
volta-t2iopt report --what profile

# 档位对照：noop≈bf16 基线 | s2=只换 GEMM | s3=+attention | v100_sm70=+VAE fp16（当年的 final）

# 单个 kernel 明细 / 微基准
zcat results/traces/trace_final_1024.json.gz > /tmp/t.json
volta-t2iopt report --what trace --dir results --limit 3
volta-t2iopt microbench gemm
volta-t2iopt microbench vae --latent results/val_hybrid_1024_9stepfinal.latent.pt

# 端到端 + 质量
volta-t2iopt validate --preset v100_sm70  --size 1024 --steps 9     # 实际取 hybrid + fp16 attn + fp16 VAE
volta-t2iopt validate --preset noop   --size 1024 --steps 9 --tag base
volta-t2iopt validate --preset v100_sm70 --size 1024 --steps 9 \
  --baseline results/val_z-image-turbo_noop_1024_9step
```
