# Z-Image-Turbo @ V100 32GB 优化方案

> **历史文档（2026-11-12，脚本时代）**：结论与实测数字全部保留，只是代码形态变了 ——
> `t2i/*.py` 脚本已重构为 `volta_t2iopt` 库，命令对照见 [MIGRATION.md](MIGRATION.md)。
> 文中路径已更新：`t2i/out/` → `results/`（本地产物，不入库）。

> 现状基线（已实测，1024² / 9 步 / bf16）：**227.4 s/张**
> 目标：**≤55 s（4×）**，力争 **≤45 s（5×）**

## 0. 现状拆解

| 阶段 | 1024² / 9 步 | 768² / 9 步 | 说明 |
|---|---|---|---|
| 文本编码 (TE) | 0.33 s | 0.33 s | Qwen3-4B，可忽略（可缓存） |
| 去噪 (9 × DiT forward) | **221.5 s**（9 × 24.6 s/fwd） | 117.7 s | 唯一的真正瓶颈（97.4%） |
| VAE 解码 | **5.26 s** | 2.87 s | 只占 2.3%（**见下方修正**） |
| 峰值显存 | 17.6 GiB | 15.1 GiB | 32 GB 卡还有余量 |

> **修正（2026-11-12）**：本表最初的 "去噪 196.9 s / VAE 30.5 s / 峰值 28.8 GiB" 是错的。
> `callback_on_step_end` 的间隔只覆盖 forward *之间*，最后一次 forward 会漏进 VAE 桶：
> `196.91 s ÷ 24.613 s = 8.0` 个间隔，所以那 30.5 s 里含第 9 次 forward（24.6 s）。
> 现在直接包住 `pipe.vae.decode` 计时，真实 bf16 VAE 是 **5.26 s**。
> 影响：VAE 优化（S4）的天花板从"省 20+ s"降到"省 4.4 s"，实测 46.4 → 42.3 s 正好对上。
> 完整 profile 见 [`PROFILE_RESULTS.md`](PROFILE_RESULTS.md)。

**慢的根本原因不是模型大，是精度**：这台 V100 上
`bf16 GEMM = 2.4 TFLOPS`（torch 回退到 fp32 模拟）vs `fp16 GEMM = 21.5 TFLOPS`，**差 9 倍**。
而 Z-Image 是 bf16-native，直接 fp16 会在 `noise_refiner.0.attention.to_out` 溢出成 inf（输出 > 65504）。

**关键观察**：溢出只发生在**输出**（fp16 存储），而**输入**是安全的
（`attention_norm1` 输出 ~15，注意力输出 ~412，norm 输出都是 O(10~100)）。
也就是说：**只要累加和输出留在 fp32，权重和输入走 fp16 张量核，就能拿到 9 倍算力**。
这是全部优化的支点。

### 0.1 微基准实测（`volta_t2iopt/benchmarks/gemm.py`，已在四组真实形状上跑过）

| 精度路线 | qkv | to_out | ffn_in | ffn_out | 中位 | 相对 fp16 |
|---|---|---|---|---|---|---|
| fp16 + fp32 累加 | 20.00 | 22.14 | 19.26 | 19.36 | **20.0 TF** | 1.00× |
| fp16 + fp16 累加 | 19.62 | 19.86 | 19.54 | 20.41 | 19.9 TF | 0.99× |
| bf16（现状） | 2.52 | 2.45 | 2.52 | 2.52 | **2.5 TF** | 0.13× |
| fp32 SGEMM | 3.25 | 3.21 | 3.22 | 3.27 | 3.3 TF | 0.16× |

两个结论直接改写了路线：

1. **fp32 兜底方案是死路**：3.3 TFLOPS 只比 bf16 快 1.3×。V100 的 fp32 标称 15.7 TF，实测只有 21% ——
   `nvidia-smi` 显示 `HW Power Brake Slowdown: Active`，重 SIMT 负载被功率刹车压住；
   而 fp16 张量核能跑到 20 TF（标称的 71%），说明这不是全局降频，是**能效差异**。
   所以只有张量核这一条路可走。
2. **fp16 混合精度比现状更准**：`relL2 vs fp32` = 2.07e-4，而 bf16 路径是 1.66e-3（fp16 尾数 10 位 > bf16 的 8 位）。
   优化不会牺牲数值质量，反而提升。

## 0.2 一次 forward 的真实预算（`volta-t2iopt report --what trace results/trace_1024.json`）

1024²、bf16、1 次 DiT forward，**只统计 CUDA kernel**（`key_averages()` 会把 `aten::*` 和真 kernel 重复计，不能用）：

| 项 | 时间 | 占比 | kernel 数 | 下一步 |
|---|---|---|---|---|
| bf16 GEMM 仿真（`magma_sgemmEx_kernel<float,bf16,bf16>`） | **19.80 s** | 79.6% | 241 | S2 换 fp16 张量核 → 预计 2.5 s |
| attention 的 fp32 SIMT GEMM（`volta_sgemm_64x64_*`） | 2.76 s | 11.1% | 68 | SDPA 掉到 **math 后端**且在 fp32 跑 → S3 换 fp16 |
| softmax（`cunn_SoftMaxForwardReg`） | 0.55 s | 2.2% | 30 | 带宽活，flash 不可用（sm_70），基本不动 |
| 元素级 + reduce + copy | 1.56 s | 6.3% | 4914 | S3 融合 adaLN / silu-mul |
| **GPU 空转** | 0.42 s | **1.7%** | — | **推翻原计划的「launch 占 1.9 s」**：现在 kernel 动辄 80 ms，发射完全藏得下 |
| 合计 | 24.87 s（时间轴跨度 25.29 s） | | 7378 次发射 | |

两个与原计划不一致、必须修正的判断：

1. **CUDA Graph 的收益上限不是 1.9 s，现在是 0.42 s**（M1 之后 kernel 变短，最多也就 0.4~0.8 s）。
   它从"M3 的核心项"降级成"S3 的收尾项"。
2. **attention 不是 SDPA mem-efficient，而是 math 后端 + fp32 SIMT GEMM**：trace 里是
   `volta_sgemm_64x64_nn/tn` 而不是 `flash`/`fmha`，前面还有 3 个 `direct_copy_kernel_cuda` 在做 q/k/v 转 fp32。
   sm_70 上 flash / mem-efficient 都不可用，所以只能靠 **fp16 进 SDPA** 拿张量核，预计 2.76 s → ~0.35 s。

## 0.3 逐层激活量级（`results/profile_1024.json`，276 个 Linear）

- fp16 **不安全**的层 14 个；需要预缩放的 **17 个 / 276 个**（94% 的层 k=0，热路径零开销）。
- 溢出主力**不是** `attention.to_out`，而是 **`feed_forward.w2`**（silu-gated 输入本身就有 6.5e4 量级）。
  最坏：`layers.14.feed_forward.w2`，输入 |max|=65536、输出 |max|=**251904** → k=3（÷8）。
- 17 个需要 k>0 的层里 14 个是 `feed_forward.w2`，2 个是 `attention.to_out`（noise_refiner.0 / context_refiner.0）。
  所以必须按层查表设 k —— 统一 k=3 会让 94% 的层白付一次乘法。

## 0.4 S2 结果（HybridLinear：fp16 张量核 + 在线预缩放）

### 方案被实测改写了一次

原计划"按 profile 定死每层 k"**不成立**：512px 一跑就 inf。原因是激活量级依赖分辨率与步数 ——
`noise_refiner.1.feed_forward.w2` 的输入 |max| 在 1024 下是 5824、512 下是 25472（4.4×），
同一个 run 里后续 step 还在涨（step0 的 w2 输出 45312 → step1 直接溢出）。

改成**在线自适应**：用权重的 ∞-范数（加载时一次算好）给出输出上界 `|out| ≤ max|x| · max_i Σ_j|w_ij|`，
每层每次调用现场取 `k = ceil(log2(amax_x · rowsum / 50000))`，`s = 2^-k`。
- **保证不溢出**（任何分辨率/步数/提示词），全程 GPU 侧 0 维张量运算，**不回 host、不产生同步**。
- 代价：每层多一次归约（~0.07 ms）+ 一次 mul；实测 elt 从 1.56 → 2.00 s。
- 上界比真实值松 1~2 个数量级（实测 k 落在 0~13），但**放大缩写在数值上免费**：
  实测 k=0/4/8/10 的 relL2 都是 2.94e-4，与不缩放完全一致（fp16 指数范围远宽于激活动态范围）。

### 数值：比 bf16 基线更准，不是"用质量换速度"

单层 GEMM 误差（真实激活，参考 = fp32 权重×fp32 输入，`volta_t2iopt/benchmarks/gemm_error.py`）：

| 层 | K | \|out\|max | bf16 relL2 | **fp16 自适应 relL2** |
|---|---|---|---|---|
| layers.14.feed_forward.w2 | 10240 | 2.4e5 | 1.71e-3 | **2.89e-4** |
| noise_refiner.0.attention.to_out | 3840 | 1.4e5 | 2.10e-3 | **2.56e-4** |
| layers.0.attention.to_q | 3840 | 1.6e2 | 1.66e-3 | **2.08e-4** |

端到端拿 **fp32 真值**对拍（512²/2 步，`volta-t2iopt validate --preset noop --path <z-image-turbo 原始 F32 目录>`）：

| 路径 | latent relL2 vs fp32 | SSIM vs fp32 | 每 forward |
|---|---|---|---|
| bf16（原基线） | 9.51e-2 | 0.968 | 5.85 s |
| **hybrid（fp16）** | **6.69e-2** | **0.978** | **1.32 s** |

即：**比它替换掉的基线更接近 fp32 轨迹，同时快 4.4×**。单 forward 的 latent relL2 是 2.7e-2，
但这是网络末层的抵消放大（隐状态 1e3~1e4 → 输出 4），不应用它当判据 —— 判据应为"单层 GEMM 误差 + 端到端 SSIM"。

### 1024² / 9 步

| | bf16 基线 | hybrid | |
|---|---|---|---|
| 每 forward | 24784 ms | **7953 ms** | 3.12× |
| 去噪 9 步 | 196.9 s | **63.6 s** | |
| VAE | 30.5 s | 13.1 s | |
| 总计 | 228.4 s | **76.7 s** | **2.98×** |
| SSIM（vs bf16） | — | 0.9838 | ≥0.98 ✓ |

### 新的 forward 预算（`trace_1024_hyb.json`，kernel 8.05 s / 空转 5.2%）

| 项 | S2 前 | S2 后 | 说明 |
|---|---|---|---|
| GEMM | 19.80 s（bf16 仿真） | **2.39 s**（cutlass fp16 `s884gemm`） | 与微基准的 20 TF 完全对上 |
| attention | 2.76 s | **2.89 s** | 没动过；现在变成**第一大头（36%）** |
| softmax | 0.55 s | 0.54 s | 不变 |
| 元素级 | 1.56 s | 2.00 s | 预缩放多出 ~0.44 s |
| GPU 空转 | 0.42 s | 0.43 s | 不变 |

**S3 的目标顺序因此变成**：attention fp16（2.89 → ~0.35 s）≫ 元素级（0.44 s）>> CUDA Graph（≤0.4 s）。

> 完整的逐 kernel / 逐层 / 逐档位数据见 [`PROFILE_RESULTS.md`](PROFILE_RESULTS.md)。

## 0.5 S3/S4 结果：attention 进 fmha + VAE 进 fp16

### S3：bf16 的 attention 掉到 math 后端，是因为 **cutlass fmha 只有 f16 版本**

V100(sm_70) 上 flash / cudnn 都不支持；mem-efficient 的 cutlass 内核叫
`fmha_cutlassF_f16_aligned_32x128_rf_sm70` —— **只吃 fp16**。所以 bf16 输入直接掉到 math 后端，
而且 math 后端会把 q/k/v 转 fp32 走 SIMT GEMM（上一节 trace 里的 `volta_sgemm_64x64`）。

同形状（(1,30,4160,128)）实测：

| 输入 | 时间 | 走的 kernel |
|---|---|---|
| bf16 + SDPA | 137.5 ms | math → 3× direct_copy + `volta_sgemm` + 独立 softmax |
| **fp16 + SDPA** | **30.1 ms** | `fmha_cutlassF_f16_..._sm70`（softmax 已融合） |
| fp16 手写 math | 42.8 ms | tensor core bmm + 独立 softmax |

做法（`volta_t2iopt/passes/attn_fp16.py`）：把 `dispatch_attention_fn` 包一层，输入是 bf16 就转 fp16 再调，
输出转回 bf16。**注意坑**：`transformer_z_image.py` 在 import 时就把函数对象绑进了自己的命名空间，
只改 `attention_dispatch` 的模块属性无效 —— 要扫 `sys.modules` 把所有持有旧引用的模块都替换掉（实测 57 处）。

安全性：qk-norm 之后 q/k 元素量级 ~6，logits O(10)，softmax ∈[0,1]，v ~400，全在 fp16 范围内。

### S4：VAE 直接用 fp16，不必分段回退

| | bf16 | **fp16** | |
|---|---|---|---|
| 1024² 解码 | 5093 ms | **874 ms** | 5.83× |
| 与 bf16 的图 | — | SSIM **0.99966** | max\|Δ\|=0.032 |
| inf/nan | — | 无 | latent \|max\|=5.3，范围安全 |

顺带峰值显存 17.6 → 14.1 GiB。原计划里"若溢出只对 `mid_block` 保 fp32"的预案**用不上**。

### 最终账（1024² / 9 步）

| 阶段 | 时间 | 倍数 |
|---|---|---|
| bf16 基线 | 228.4 s | 1.00× |
| S2（fp16 张量核 GEMM） | 76.7 s | 2.98× |
| S3（attention → fmha fp16） | 46.4 s | 4.92× |
| S4（VAE → fp16） | **42.3 s** | **5.40×** |

每 forward：24784 → **4587 ms**（5.40×）；VAE 30.5 → **1.02 s**；峰值显存 17.6 → 14.1 GiB。

### 质量（对 fp32 真值，512²/2 步 —— 1024² 的 fp32 transformer 加激活放不进 32 GB）

| 路径 | latent relL2 | SSIM | 每 forward |
|---|---|---|---|
| bf16 基线 | 9.51e-2 | 0.968 | 5.85 s |
| S2（S2 的 hybrid） | 6.69e-2 | 0.978 | 1.32 s |
| **S2+S3（最终）** | **6.06e-2** | **0.982** | 1.11 s |

注：S2+S3 对 bf16 基线的 9 步 SSIM 是 0.9779（**略低于** 0.98），但那不是质量回归 ——
拿 fp32 真值做裁判时，最终版比基线更接近（0.982 vs 0.968）。判据应当是"对 fp32 的 SSIM"，
"对基线的 SSIM" 只是稳定性指示。

### 还剩什么可做（按实测余量排）

| 项 | 实测余量 | 判断 |
|---|---|---|
| 元素级 1.22 s | 融合 silu-mul / adaLN 调制 / RMSNorm+scale / 残差加 | **唯一还有肉的一块**，估计能省 0.6~0.8 s |
| fp16 GEMM 2.32 s | 已经是 20 TFLOPS（V100 fp16 峰值 ~28，71%）；融合 epilogue 可再省掉下游的 mul/加 | Triton 版可试，收益主要在省 elt |
| attention 1.01 s | fmha 已是 4.4 TFLOPS 有效值；自写 Triton 融合注意力理论上能到 ~2× | 要写 kernel，属于练习项 |
| GPU 空转 | **实测几乎为 0**（trace 里 16.5% 是 profiler 自己注入的：kernel 4.56 s vs 无 profiler 时整 forward 4.58 s） | **CUDA Graph 彻底降级为"不做"** |

## 1. 优化项总表

| # | 优化 | 原理 | 预期收益 | 代价 | 风险 |
|---|---|---|---|---|---|
| **P0-a** | 混合精度 GEMM：fp16 权重 × fp16 输入 → fp32 累加，**输入侧 2 的幂预缩放** | 用 `mma.sync.f16.f32`，绕开 bf16 的 fp32 回退；预缩放把输出压回 fp16 范围 | 前向 24.6 → 3–4 s（20 TF 下 GEMM 只要 2.4 s） | 纯 PyTorch 一个 wrapper | 低（已数值验证，见 §0.2） |
| **P0-b** | 上 P0-a 的 Triton 版并融合 epilogue（bias + FFN 的 `silu(g)*u`） | 少一次 63 MB 张量往返 + 少 2 次 launch | 前向 → 4–5 s（目标 ≥12 TFLOPS） | 半天 | 中（要调 tile/num_stages） |
| **P1-a** | CUDA Graph 捕获每一步 | 这台机器 launch = **10.8 µs**，每 forward ~180 个 kernel ≈ 1.9 s | **修正：实测 GPU 空转仅 1.7%（0.42 s）**，收益上限 0.4~0.8 s | 半天（每步一个静态图） | 中（timestep 是动态输入，需静态缓冲） |
| **P1-b** | 融合 adaLN 调制 + RMSNorm + 残差加 | 每层约 6 个元素级 kernel，全是带宽/launch 活 | 前向 −10~20% | 1–2 天 | 低 |
| **P2-a** | VAE 解码改 fp16 | latent 是 O(1)，conv 在 fp16 下 cudnn 有原生加速 | 30.5 → ~10 s | 1 小时（先验证不溢出） | 低 |
| **P2-b** | attention 用 fp16 SDPA（q/k 已 qk-norm，v ~400） | **已实测：SDPA 掉到 math 后端，q/k/v 被转 fp32 走 SIMT GEMM，吃掉 2.76 s（11%）** | 2.76 → ~0.35 s | 1 小时 | 低 |
| **P2-c** | 步数 9 → 8（官方 NFE）；评估 5 步质量 | 线性 | 9 步 196.9 s → 8 步 175 s / 5 步 110 s | 画质评测 | 中（质量下降） |
| **P2-d** | 分辨率 1024² → 768² | token 数 (64²=4096) → (48²=2304)，0.56× | 前向 ×0.56 | 画质/用途 | 低 |
| **P3** | 导出 fp16 权重的静态图 / TensorRT | 图级融合 + 常量折叠 | 未知，可能 −10~20% | 数天 | 高（TRT 对 sm_70 + DiT 支持差，不建议先做） |

## 2. 分阶段实施

### M0：先 profile，别猜（0.5 天）
- 用 `torch.profiler` 抓**单次 forward** 的 kernel 级时间分布，输出 top-15 kernel。
- 需要的三个数字：
  1. GEMM 占比（预期 ~65%）
  2. attention 占比（预期 10–20%，若 >30% 说明 SDPA 掉到 math 后端，P2-b 提前做）
  3. 元素级 kernel 的合计时间 + kernel 数量（决定 P1-a/P1-b 的优先级）
- **验收**：能给出"GEMM / attention / 其他"三分的时间占比。

### M0.2：预缩放机制（已数值验证，`volta_t2iopt/benchmarks/prescale.py`）

split-K 方案**不需要**了。溢出只发生在「结果写回 fp16」这一步，所以只要让结果本身小 2^k 倍就行：

```python
out = (x.mul(s).half() @ w.half().T).float().mul(1 / s)   # s = 2^-k，k 由逐层实测决定
```

- **必须「先乘再转」**：`x.half() * s` 是错的 —— x 自身就可能超 65504（实测 `attention.to_out` 输入量级 1e5），
  转 fp16 那一刻就 inf 了，后面再缩也救不回来。验证：`|x|max=1.1e5` 时 `(x*s).half()` finite、`x.half()*s` 全 inf。
- **只需要缩一侧**：缩 x 就够（输出等比缩小），权重不用缩 —— 少一份 12 GB 的预处理副本。
- **无损**：2 的幂缩放对 fp16 尾数是精确移位，relL2 与不缩放的 fp16 完全一致（2.07e-4）。
- **k 怎么定**：按层实测 `|out|max`，`k = ceil(log2(max/50000))`，只对需要的层开（预计只有 `attention.to_out` 一族），
  其余层 k=0，热路径不做任何多余乘法。
- **热路径成本**：输入侧的 `mul(s)` 塞进上游 RMSNorm 的 epilogue（等于免费）；输出侧 `float().mul(1/s)` 是 94M 元素的
  带宽活，约 0.3 ms/层，后续可融进残差加。**注意**：若每 call 都现算 `w.mul(s)`，微基准已测掉 11%（20 → 17.8 TF），
  所以权重的任何变换都必须在加载时烘焙。

### M1：P0-a，零代码风险先试（0.5 天）
用 `HybridLinear` 整体替换 `nn.Linear`（`volta_t2iopt/passes/hybrid_linear.py`），改前向、不改模型结构：

```python
class HybridLinear(nn.Module):
    def forward(self, x):
        s = self.x_scale          # 2^-k，加载时按实测 |out|max 决定；k=0 时走原始路径
        out = (x.mul(s).half() @ self.w_f16.T).float().mul(1 / s)
        return (out + self.bias).to(x.dtype)   # 回 bf16，残差流精度不变
```
- 权重在加载时一次性转 fp16 存好（`w_f16`），bf16 激活流不动 —— 影响面只有 GEMM 内部，好回滚。
- 每层一个开关（环境变量 `VOLTA_T2IOPT_DISABLE=hybrid_linear/1`），能一键回 bf16 基线。
- **验收**：前向 24.6 → ≤8 s，且 latent 与 bf16 参考的相对 L2 误差 < 1e-2、无 inf/nan。
  （由于 fp16 比 bf16 更准，实际应好于 1e-2，按 relL2 < 2e-3 卡）

### M2：P0-b，写 Triton 混合 GEMM（1 天）
```python
@triton.jit
def gemm_f16w_f32acc(A, W, Bias, C, M, N, K, ...):
    a = tl.load(...)                       # fp32/bf16 输入
    w = tl.load(...)                       # fp16 权重
    acc = tl.dot(a.to(tl.float16), tl.trans(w), out_dtype=tl.float32)   # ← 关键
    acc += bias
    tl.store(C, acc.to(tl.bfloat16))       # 输出回 bf16（范围大，安全）
```
- 复用 `bench/kernels/csrc` 的工程结构（JIT + 注册 + 对拍），Triton 先行、CUDA 版后补（正好是练习 5–7）。
- FFN 的 `w1/w3 → silu(g)*u` 融进 epilogue。
- **验收**：在真实形状（M=4096, K=3840, N=3×3840 / 10240）上 ≥12 TFLOPS（fp16 峰值 21.5 的 56%）。

### M3：P1 + P2 收尾（1 天）
- CUDA Graph：为每个 timestep 捕获一张静态图（Z-Image 9 步 → 9 张），输入用静态 buffer。
- VAE 改 fp16；若溢出则只对 `decoder.mid_block` 保 fp32。
- 768² / 5 步 作为"快速档"。
- **验收**：1024²/9 步 ≤55 s；768²/5 步 ≤20 s。

## 3. 预期终点

| 配置 | 现在 | 优化后（预期） | 倍数 |
|---|---|---|---|
| 1024² / 9 步 | 227.4 s | **45–55 s** | 4–5× |
| 1024² / 5 步 | 126.5 s | 28–33 s | 4× |
| 768² / 9 步 | 117.7 s | 27–32 s | 4× |
| 768² / 5 步 | — | **15–20 s** | — |

拆账（1024²/9 步）：前向 196.9 → ~36–45 s（10×/−，其中 fp16 算力 9×、graph 再省 1.9 s/步），
VAE 30.5 → ~10 s，TE 0.33 s（可缓存到 0）。

> 现在有了实测 20 TFLOPS（不是估的），拆账变硬：GEMM 48 TFLOP / 20 TF = **2.4 s/fwd**；
> 剩下的 attention + 元素级 + launch（现状 24.6 − 19.2 ≈ 5.4 s）靠 CUDA Graph + adaLN 融合压到 ~1.5 s。
> 合计 3.9 s/forward → 9 步 35 s + VAE 10 s + TE 0.3 s ≈ **45 s**，5× 目标可达。
> 危险区：非 GEMM 的 5.4 s 占优化后总时长的一半以上，M1 一落地它就从"零头"变成"主项"。
> 所以 M0 profile 里 kernel 数和 SDPA 后端必须测清楚，否则会卡在 70 s 不动。

## 4. 验证协议（每个里程碑都要跑）

1. **数值**：与 bf16 参考比 latent 的 `max|Δ|` 和相对 L2；逐层 hook 检查 `|x|max < 65504`（fp16 输入端）与 `|out|` 无 inf。
2. **画质**：固定 5 条 prompt（含中英文字渲染各 1 条），出图目视 + 与参考图 SSIM；prompt 固定、seed 固定。
3. **性能**：分阶段事件计时（TE / 每步 / VAE），3 次取中位；同时记录 SM 频率与功耗（这台机器算力只有标称 19%，跨天对比必须带频率）。
4. **回归**：每一步优化都保留一个开关（环境变量），能一键回到 bf16 基线。

## 5. 明确不做的事

| 不做 | 原因 |
|---|---|
| fp8 / fp4 权重与计算 | sm_70 无 fp8/fp4 张量核；fp8 checkpoint 只能省显存，计算要反量化 |
| Marlin / Machete / bitsandbytes 4bit GEMM | 需要 sm_80+ |
| FlashAttention 2/3、SageAttention | 需要 sm_80+；用 SDPA mem-efficient（已验证可用） |
| TMA / wgmma / cluster | sm_90+ |
| 多卡张量并行 | 只有一张卡（且 PCIe x8，没有 NVLink） |
| TensorRT | 对 sm_70 + 单流 DiT 的算子支持差，收益不确定 |
| 全模型 fp16（不改造累加） | 已实测：`attention.to_out` 输出溢出，数学上无解 |

## 6. 代码落点

| 文件 | 作用 |
|---|---|
| `volta_t2iopt/analysis/profiler.py` | M0：单 forward 的 kernel 级 profile + 逐层激活量级 |
| `volta_t2iopt/passes/hybrid_linear.py` | M1：`HybridLinear`（fp16 权重 / fp32 累加），可整体替换 `nn.Linear` |
| （Triton 混合 GEMM：计划中，未实现；落点会是 `volta_t2iopt/passes/`） | M2：Triton 混合 GEMM（含 bias / silu 融合） |
| （CUDA Graph：已实测否决，见 §0.5） | M3：per-timestep CUDA Graph 捕获 |
| `volta_t2iopt/cli/bench.py` | 分阶段计时（CUDA event，VAE 单独包 decode），所有里程碑共用 |
| `volta_t2iopt/benchmarks/gemm.py` | ✅ 已写：四组真实形状 × 五种精度路线的 GEMM 峰值（§0.1） |
| `volta_t2iopt/benchmarks/prescale.py` | ✅ 已写：2 的幂预缩放的数值验证（§0.2） |

## 7. 执行顺序（按投入产出排，现在按这个跑）

| 步 | 做什么 | 机器时间 | 为什么排这里 | 验收 |
|---|---|---|---|---|
| **S0** | ✅ 完成：bf16 权重缓存 + prompt_embeds 存 `.pt`（**不做 fp16 缓存**，GPU 上就地转换更便宜） | ~15 min（一次性） | 冷启 **715 s → 4.3 s**（跳过 TE 后）；**注意这台机器只有 30 GB 内存**，把 20 GB 整条 pipeline 读进 RAM 会把 swap 打满、加载退化成 100 MB/s | 已出 `/home/lskam/models/z-image-turbo-bf16` + `prompt_embeds_bf16.pt` |
| **S1** | ✅ 完成：`profile_zimage.py` + `analyze_trace.py`（结果见 §0.2 / §0.3） | ~5 min | 定出每层 k，并查明剩下 5.1 s 是 attention 的 fp32 SIMT GEMM（2.76 s）+ softmax（0.55 s）+ 元素级（1.56 s） | `out/profile_1024.json`、`out/kernels_1024.json` |
| **S2** | ✅ 完成：`hybrid_linear.py`（fp16 张量核 + **在线自适应**预缩放）+ `validate.py` + `check_gemm_error.py` | ~2 h | GEMM 19.80 → **2.39 s**；fwd 24.8 → 7.95 s | fwd 7.95 s ✓、SSIM 0.9838 ✓、单层 relL2 2.9e-4（比 bf16 好 6×）✓ |
| **S3** | ✅ 完成：`fp16_attention.py`，SDPA 输入转 fp16 → 走 `fmha_cutlassF_f16_sm70` | ~1 h | bf16 没有 fmha 内核，白付 fp32 SIMT GEMM | 2.89+0.54 → **1.01 s** ✓ |
| **S4** | ✅ 完成：VAE fp16（`check_vae.py` 先验证再上线） | 40 min | 5.09 → **0.87 s**，SSIM 0.99966 | 1024²/9 步 **42.3 s = 5.40×** ✓ |
| **S5** | 元素级融合（silu-mul / adaLN / RMSNorm+scale / 残差）—— 唯一还有肉的一块 | 1 天 | 1.22 s 中约 0.6~0.8 s 可省 | fwd 4.59 → ~3.8 s |

S2 之前不要写 Triton kernel：纯 PyTorch 已经能拿到 20 TF 中的 17–18 TF，Triton 只在 S3 证明 GEMM 仍是主项时才值得。
