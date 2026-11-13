# 架构

## 一句话

把"bf16-native 模型 + 没有 bf16 张量核的卡"这条不匹配，拆成**可判定、可回滚、有理由**的三件事：
探针（缺什么）→ pass（怎么补）→ gate（补对了吗）。

## 分层与依赖方向

```
capability.py  ──►  config.py  ──►  passes/  ──►  runtime/  ──►  cli/
（唯一读硬件）      （策略+preset）   （改模型）     （加载/计时/出图）   （参数解析）
                                        │
                                        ├──► analysis/（trace / 逐层统计）
                                        └──► validate/（SSIM / relL2 / gate）
```

四条规矩，都是为了让"优化"这件事可复核：

1. **只有 `capability` 读硬件**。pass 里出现 `torch.cuda.get_device_capability()` 就算 bug ——
   那意味着同一个 preset 在不同机器上行为不可预测。
2. **pass 之间不互相调用**。需要共享状态就放 `Context.state`，顺序由 `passes/__init__.py` 的注册顺序决定。
3. **`session` 是唯一的编排者**：负责顺序、失败降级（一个 pass 抛异常不影响其它）、回滚。
4. **analysis / validate 不修改模型**，只读。所以它们可以在任意档位上重复跑。

## 能力决策表（`auto` preset）

| 模型原生精度 | 卡的张量核 | 结论 |
|---|---|---|
| bf16 | fp16 有、bf16 无（sm_70/75） | 三个 pass 全开（这就是本库存在的原因） |
| bf16 | 两种都有（sm_80+） | 全关：bf16 已在张量核上，迁移只多一次类型转换 |
| fp16 | fp16 有 | 全关：已经是原生精度 |
| 任意 | 连 fp16 张量核都没有（sm_60/61） | 全关，并在 notes 里明说"本库无收益" |

决策与理由一起进 `OptimizerConfig.notes`，`sess.summary()` 会原样打印。
`probe(measure=True)` 会额外真跑 GEMM/SDPA 微基准，把"实测 bf16/fp16 = 0.12×"这种数字写进 notes。

## pass 契约（`passes/base.py`）

```python
class Pass:
    name: ClassVar[str]
    def applicable(self) -> tuple[bool, str]   # (是否适用, 不适用的原因)
    def apply(self) -> PatchResult             # 幂等；异常自己兜住
    def revert(self) -> None                   # 包括全局状态
```

`PatchResult` 里必须带：`applied / modules / reason / details / notes`。
这不是日志洁癖 —— 同一条命令在不同卡上跑出来的报告应该能解释"为什么这台机器快了 5.4 倍、
那台机器一倍没变"，否则它就只是个静默的猴子补丁集合。

### 顺序

| # | pass | 为什么在这个位置 |
|---|---|---|
| 1 | `hybrid_linear` | 先换 GEMM：它决定了显存占用（权重变 fp16），后面的 pass 都在此基础上测 |
| 2 | `attn_fp16` | 全局函数重定向，与模块结构无关，放在结构改动之后 |
| 3 | `decoder_fp16` | 只碰 VAE，与前两者正交（若加载时已给 fp16，这里就是 no-op） |

### 回滚的代价

`hybrid_linear` 默认**不保留**原 bf16 权重：11.7 GiB 的 bf16 transformer 与 5.9 GiB 的 fp16 同时驻留，
会让峰值显存从 12.3 GiB 涨到 23.9 GiB（实测），32 GB 卡上 1024² 就没余量了。
因此 `revert()` 会用 fp16 权重重建 `nn.Linear`（等价到 bf16 舍入精度）。
显存充裕、或做需要精确回滚的实验时，传 `keep_originals = true`（preset 里可配）。

## 适配器（`models/`）

```python
class ModelAdapter:
    def load(...) -> pipe          # 精度在加载时就定好，别"先 bf16 读进来再全量转"
    def denoiser(pipe) -> nn.Module
    def decoder(pipe) -> nn.Module | None
    def encode(pipe, prompt) -> GenerationConditions
    def run(pipe, conditions, *, size, steps, seed, ...) -> list[Image]
```

接入一个新模型 = 写一个 40–80 行的文件（`AnimaAdapter` 之类的），**pass 层一行都不用改**。
`generic.py` 是兜底：任何有 `vae + (transformer|unet)` 的 diffusers pipeline 都能跑，
但它只保证"能跑、能被优化"，不保证输出语义与官方 pipeline 一致。

## 计时口径（`runtime/timing.py`）

- 每步用 CUDA event 在 `callback_on_step_end` 打点；
- **VAE 单独包住 `vae.decode`**：`denoise = total − vae`，绝不用"总数 − 每步间隔之和"
  （callback 不覆盖最后一次 forward，这个错误曾把 VAE 记成 30.5 s，真实 5.26 s）；
- 没有 CUDA 时自动退回 `perf_counter`，所以计时逻辑本身可以在 CPU 上被测。

## 分析（`analysis/`）

| 模块 | 干什么 | 口径要点 |
|---|---|---|
| `trace.py` | chrome trace → kernel 分类统计 | 只算 `ph=X, cat=kernel`；attention 先判（`fmha_*_sm70` 里同时有 `cutlass` 和 `sm70`，先判 GEMM 会错桶） |
| `layer_stats.py` | 逐层 `|x|max / |out|max / rowsum` → 需要多少 k | 只能在未替换的模型上用（替换后不是 `nn.Linear`） |
| `profiler.py` | 第一次 forward 挂 profiler，第二次记干净墙钟 | 空转率会被 profiler 放大，只能用来判断"值不值得挖" |
| `report.py` | 汇总 JSON → 终端/markdown 表 | — |

## 验收（`validate/`）

`GateCriteria` 的默认值来自实测，不是愿望：

| 检查 | 默认 | 依据 |
|---|---|---|
| `ssim >= 0.98` | **判** | 对 fp32 真值：bf16 0.968 → 最终版 0.982 |
| 无 inf/nan | **判** | fp16 溢出是硬失败，不是精度问题 |
| `latent relL2` | **只报** | 1 次 forward 就有 2.7e-2（扩散放大差异），历史写的 2e-3 在 latent 尺度上从未成立 |
| `latency_ratio` | 可选 | 给 `--max-latency-ratio` 才判 |

## 新增一个 pass 的检查表

1. `passes/<name>.py`，`@register`，`name` 唯一；
2. `applicable()` 里写清**不适用的原因**（一句话，能贴进报告）；
3. `apply()` 幂等；记录 undo 状态；把显存/数量等可核对的数字放进 `details`；
4. `revert()` 必须能卸掉**所有**副作用，包括全局 flag（例如
   `torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction`）；
5. 在 `presets/*.toml` 里给一个开关位置；
6. 加一个 CPU-only 单测：apply → 断言效果 → revert → 断言复原；再画一张"适用/不适用"的真值表。

## 测试策略

| 层 | 方式 | 现状 |
|---|---|---|
| 决策 / 配置 / pass 契约 / 指标 / 分类 / 计时 | CPU-only pytest，不加载模型 | 62 个用例 |
| 端到端 | `volta-t2iopt bench --size 512 --steps 2`（约 20 s） | 手工，见 README |
| 质量 | `validate --fp32-ref` + gate | 手工，需 fp32 权重 |
| 性能回归 | 同一台机器上 `bench` 两次取最后一次（热缓存） | 手工 |

不做假 CI：这个仓库的 GPU 测试是"人跑一次、把数字写进文档"，因为唯一一块 V100 是共享的。
