# V100 上跑现代文生图模型：实测记录

> **历史文档（2026-11-12，脚本时代）**：结论与实测数字全部保留，只是代码形态变了 ——
> `t2i/*.py` 脚本已重构为 `volta_t2iopt` 库，命令对照见 [MIGRATION.md](MIGRATION.md)。
> 文中路径已更新：`t2i/out/` → `results/`（本地产物，不入库）。

环境：Tesla V100-PCIE-32GB（sm_70）、CUDA 12.9、torch 2.8.0+cu128、diffusers 0.42.1。
结论先写：**V100 上能不能用，不取决于参数量，取决于模型是不是 bf16-native。**

## 1. Z-Image-Turbo（ELO 936，Apache-2.0，6B）

### fp16 直接崩：注意力输出投影溢出

```
noise_refiner.0.attention.norm_q    max 5.8        ← qk-norm 后正常
noise_refiner.0.attention.to_out.0  max inf        ← 输出投影累加超过 65504
noise_refiner.0.attention_norm2     max nan        ← 残差流已污染，后面全 nan
```

原因链：`t_embedder(t * t_scale)` 里 t∈[0,1000]×1000 = 1e6 > fp16 上限（这一处可用
`timestep.float()` 修掉，让乘法提升到 fp32）；但**注意力的输出投影本身**在 fp16 下就会溢出
（注意力输出量级 ~400，3840 维累加 → >65504）。这不是精度瑕疵，是范围不够，改不了。

### bf16（V100 上是 fp32 模拟）

| 分辨率 | 步数 | 总耗时 | 每 forward | VAE（直测） | 峰值显存 |
|---|---|---|---|---|---|
| 1024² | 9 | **227.4 s** | 24.68 s | 5.26 s | 17.6 GiB |
| 1024² | 5 | 126.5 s | 24.24 s | 5.26 s | 17.6 GiB |
| 768² | 9 | 117.7 s | 12.76 s | 2.87 s | 15.1 GiB |

> "每 forward" = (总耗时 − VAE) / 步数。**早期的 `vae_s=30.5` 是错的**：那时候 VAE 是用
> `总耗时 − callback 间隔之和` 估的，而 callback 只覆盖 forward *之间*，最后一次 forward
> 会漏进 VAE 桶（`196.91 s ÷ 24.613 s = 8.0` 个间隔）。详见 `PROFILE_RESULTS.md` §0。

图是正常的（`results/zimage_bf16_1024_9.png`，RGB std 0.344）。fp16 路径虽然出 NaN，
但计时有效：**3.63 s/forward**（1024²）—— 也就是说 bf16 让每步慢了 **6.8 倍**。

根因：`bf16 @ bf16` 在 sm_70 上没有张量核，torch 走 fp32 回退，实测 4096³ 矩阵乘
**2.4 TFLOPS** vs fp16 的 **21.5 TFLOPS**（9 倍）。所以 1024² 的每步 ≈ 49 TFLOP / 2.4 TFLOPS ≈ 20 s。

## 2. 对照：fp16-native 的 SDXL-turbo

| 分辨率 | 步数 | 总耗时 | UNet (ms/step) | VAE+TE | 峰值显存 |
|---|---|---|---|---|---|
| 1024² | 4 | **6.40 s** | 1.63 s (543 ms) | 4.78 s | 9.7 GiB |
| 512² | 4 | 1.36 s | 0.51 s (169 ms) | 0.86 s | 7.5 GiB |

同一个 V100，**227 s vs 6.4 s，35 倍**。而且 SDXL 的 VAE 解码就吃掉 3–4 s
（1024² 的 AutoencoderKL 是纯 conv + attention，在这些卡上是显存带宽活）。

## 3. 对选型的影响

| 模型 | 精度可行性 | 1024² 实测/推算 | 说明 |
|---|---|---|---|
| SDXL / SDXL-turbo 系 | fp16 原生 ✅ | **6.4 s** | 质量是 2023 代，但速度是唯一能交互的 |
| Z-Image-Turbo (6B) | fp16 ❌ / bf16 ✅ | **227 s** | 质量最好（ELO 936），但要 4 分钟/张 |
| FLUX.2-klein-4B 等现代 DiT | 大概率同 Z-Image（bf16-native） | 未测 | 若 fp16 也溢出 → bf16 下 ≈ 3.6B×4 步 ×(21.5/2.4) ≈ 40 s |
| Qwen-Image-2.1 / FLUX.2-dev / Krea-2 | 显存就不够 | — | 30.7 / 108 / 33 GiB |

**V100 的正确用法**：要么接受 bf16 的 9 倍算力损失（每张几分钟），要么用 fp16-native 的老一代模型
（秒级）。想同时要"现代质量 + 秒级"，这张卡给不了 —— 换 Ampere/Ada（bf16 张量核 + fp8）才有解。

## 4. 复现

```bash
volta-t2iopt bench --preset noop --size 1024 --steps 9      # 主实验
volta-t2iopt bench --preset noop --path <fp16 权重目录>           # 看 fp16 怎么崩
volta-t2iopt bench --model sdxl-turbo                                                # fp16 对照
```

产物：`results/{sweep_bf16.json, zimage_bf16_*.png, sdxl_fp16_*.png}`。
模型缓存在 `/home/lskam/models/{z-image-turbo,sdxl-turbo}`（z-image 的 transformer 是 F32 存储，
fp16 只占 11.5 GiB 显存）。

## 5. 踩过的坑

- `ZImagePipeline.encode_prompt` 不接受 `num_images_per_prompt`，返回 `(list[Tensor], list[Tensor])`，
  文本条件是**多层特征列表**，不是单个 embedding；transformer 输入是 5 维 `[B,C,F,H,W]`（F=帧数）。
- 9 steps 实际是 8–9 次 DiT forward，用 `callback_on_step_end` 数，别硬编码。
- bf16 的 VAE conv 在 sm_70 上没有加速，解码用 fp32 反而更稳；代价是 1024² 要 30 s。

## 6. 优化记录（V100，2026-11-12）

`OPTIMIZATION_PLAN.md` 是完整方案与实测记录，**`PROFILE_RESULTS.md` 是完整的 profile 快照**
（4 个优化档 × 3 个分辨率 × 逐 kernel / 逐层 / 微基准，含三条阅读口径）。一句话总结：
**慢的根因不是模型大，是 bf16-native 在 sm_70 上没有张量核**（bf16 GEMM 2.5 TFLOPS vs fp16 20 TFLOPS）。

| 阶段 | 1024²/9 步 | 倍数 | 做了什么 |
|---|---|---|---|
| 基线 | 228.4 s | 1.00× | bf16（V100 上是 fp32 模拟） |
| S2 | 76.7 s | 2.98× | `HybridLinear`：fp16 权重 × fp16 输入 → fp32 累加，输入侧**在线自适应 2 的幂预缩放** |
| S3 | 46.4 s | 4.92× | SDPA 输入转 fp16 → 落进 `fmha_cutlassF_f16_sm70`（bf16 没这个内核） |
| S4 | **42.3 s** | **5.40×** | VAE 解码 fp16（5.26 → 1.02 s，SSIM 0.99966） |

质量：对 fp32 真值的 SSIM **0.982**，比 bf16 基线的 0.968 **更准**（fp16 尾数 10 位 vs bf16 的 8 位）。

```bash
volta-t2iopt cache all                                    # 一次性：bf16 权重缓存 + prompt 条件缓存
volta-t2iopt validate --preset noop      --size 1024 --steps 9 --tag base
volta-t2iopt validate --preset v100_sm70 --size 1024 --steps 9 --tag opt
volta-t2iopt validate --preset v100_sm70 --size 1024 --steps 9 \
  --baseline results/val_z-image-turbo_noop_1024_9step       # 对拍 + 门限判定
volta-t2iopt bench --preset noop                             # 一键回基线（旧 T2I_HYBRID=0 的等价物）
```
