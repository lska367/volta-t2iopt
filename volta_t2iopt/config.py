"""配置层：把脚本里散落的环境变量 / VARIANTS 字典收成一份可检查的配置。

三层来源，优先级从低到高：

1. **preset 文件**（`presets/*.toml`）—— 人写的、可 diff 的策略
2. **`auto`** —— 由 `capability` 探针推导，等价于"这台机器该怎么配"
3. **环境变量** `VOLTA_T2IOPT_ENABLE` / `VOLTA_T2IOPT_DISABLE`（逗号分隔）——
   只做 A/B 开关，不承载语义

任何一层改完，最终都能 `print(cfg.describe())` 出"为什么这么配"。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .capability import Capability

PRESET_DIR = Path(__file__).resolve().parent / "presets"

_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}

ENV_PRESET = "VOLTA_T2IOPT_PRESET"
ENV_ENABLE = "VOLTA_T2IOPT_ENABLE"
ENV_DISABLE = "VOLTA_T2IOPT_DISABLE"


def dtype_from_str(name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    if name is None:
        return torch.bfloat16
    key = str(name).strip().lower()
    if key not in _DTYPES:
        raise ValueError(f"未知 dtype {name!r}，可选 {sorted(set(_DTYPES))}")
    return _DTYPES[key]


def dtype_name(dt: torch.dtype) -> str:
    return str(dt).replace("torch.", "")


@dataclass
class PassConfig:
    """单个 pass 的开关 + 参数（参数直接透传给 pass）。"""

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def to_dict(self) -> dict:
        return dict(enabled=self.enabled, **self.params)


@dataclass
class OptimizerConfig:
    preset: str = "auto"
    model_dtype: torch.dtype = torch.bfloat16
    decoder_dtype: torch.dtype | None = None
    device: str = "cuda"
    passes: dict[str, PassConfig] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    source: str = "builtin"

    # --- 构造 ---
    @classmethod
    def from_preset(cls, name: str = "auto", *, capability: Capability | None = None,
                    model_dtype: torch.dtype | str | None = None,
                    device: str = "cuda") -> "OptimizerConfig":
        name = name or os.environ.get(ENV_PRESET, "auto")
        mdt = dtype_from_str(model_dtype) if model_dtype is not None else torch.bfloat16

        if name == "auto":
            if capability is None:
                from .capability import probe
                capability = probe()
            cfg = auto_config(capability, mdt, device=device)
        else:
            cfg = cls.from_toml(preset_path(name), model_dtype=mdt, preset_name=name)
            cfg.device = device
        cfg.apply_env_overrides()
        return cfg

    @classmethod
    def from_toml(cls, path: Path, *, model_dtype: torch.dtype | None = None,
                  preset_name: str = "") -> "OptimizerConfig":
        blob = tomllib.loads(Path(path).read_text())
        model = blob.get("model", {})
        cfg = cls(
            preset=preset_name or Path(path).stem,
            model_dtype=dtype_from_str(model.get("dtype", model_dtype or torch.bfloat16)),
            decoder_dtype=(dtype_from_str(model["decoder_dtype"])
                           if "decoder_dtype" in model else None),
            device=str(model.get("device", "cuda")),
            passes={k: PassConfig(enabled=bool(v.get("enabled", True)),
                                  params={kk: vv for kk, vv in v.items() if kk != "enabled"})
                    for k, v in blob.get("passes", {}).items()},
            notes=list(blob.get("meta", {}).get("notes", [])),
            source=str(path),
        )
        if model_dtype is not None:
            cfg.model_dtype = model_dtype
        return cfg

    # --- 查询 ---
    def enabled_passes(self) -> list[str]:
        from .passes import order_passes
        return order_passes([n for n, p in self.passes.items() if p.enabled])

    def pass_config(self, name: str) -> PassConfig:
        return self.passes.get(name, PassConfig(enabled=False))

    def effective_decoder_dtype(self) -> torch.dtype:
        return self.decoder_dtype or self.model_dtype

    # --- 修改 ---
    def apply_env_overrides(self) -> "OptimizerConfig":
        for names, flag in ((os.environ.get(ENV_ENABLE), True), (os.environ.get(ENV_DISABLE), False)):
            if not names:
                continue
            for n in filter(None, (s.strip() for s in names.split(","))):
                cfg = self.passes.setdefault(n, PassConfig(enabled=flag))
                cfg.enabled = flag
                self.notes.append(f"env {(ENV_ENABLE if flag else ENV_DISABLE)}={n}")
        return self

    def with_overrides(self, *, enable: tuple[str, ...] = (), disable: tuple[str, ...] = ()) -> "OptimizerConfig":
        for n in enable:
            self.passes.setdefault(n, PassConfig()).enabled = True
        for n in disable:
            if n in self.passes:
                self.passes[n].enabled = False
            else:
                self.passes[n] = PassConfig(enabled=False)
        return self

    # --- 展示 ---
    def to_dict(self) -> dict:
        return dict(preset=self.preset, model_dtype=dtype_name(self.model_dtype),
                    decoder_dtype=dtype_name(self.effective_decoder_dtype()),
                    device=self.device, source=self.source,
                    passes={k: v.to_dict() for k, v in self.passes.items()},
                    notes=self.notes)

    def describe(self) -> str:
        rows = [f"preset      : {self.preset} (from {self.source})",
                f"dtypes      : model={dtype_name(self.model_dtype)} "
                f"decoder={dtype_name(self.effective_decoder_dtype())} | device={self.device}",
                "passes      :"]
        for name in self.enabled_passes():
            rows.append(f"  + {name:16s} {self.pass_config(name).to_dict()}")
        for name, pc in self.passes.items():
            if not pc.enabled:
                rows.append(f"  - {name:16s} (off)")
        if self.notes:
            rows.append("notes       :")
            rows.extend(f"  · {n}" for n in self.notes)
        return "\n".join(rows)


def preset_path(name: str) -> Path:
    """preset 名（v100_sm70）或文件路径都可以。"""
    p = Path(name)
    if p.suffix == ".toml" and p.exists():
        return p
    cand = PRESET_DIR / f"{name}.toml"
    if not cand.exists():
        avail = sorted(x.stem for x in PRESET_DIR.glob("*.toml")) + ["auto"]
        raise KeyError(f"未知 preset {name!r}，可用：{avail}")
    return cand


def auto_config(capability: Capability, model_dtype: torch.dtype,
                device: str = "cuda") -> OptimizerConfig:
    """由能力探针推导配置 —— 本库的"自动挡"，也是唯一需要读硬件的地方。"""
    cfg = OptimizerConfig(preset="auto", model_dtype=model_dtype, device=device,
                          source="capability probe")
    bf16_native = model_dtype == torch.bfloat16
    m = capability.measured or {}
    ratio = capability.measured_ratio

    if bf16_native and not capability.bf16_tensorcore and capability.fp16_tensorcore:
        cfg.passes["hybrid_linear"] = PassConfig(True, dict(prescale=True,
                                                            out_limit=50000.0, k_max=20))
        cfg.passes["attn_fp16"] = PassConfig(True)
        cfg.passes["decoder_fp16"] = PassConfig(True)
        cfg.decoder_dtype = torch.float16
        why = (f"sm_{capability.sm} 没有 bf16 张量核、有 fp16 张量核："
               f"bf16-native 的模型会掉进 fp32 SIMT 回退")
        if ratio is not None:
            why += f"（实测 bf16/fp16 GEMM = {ratio:.2f}×）"
        cfg.notes.append(why)
        if "sdpa_bf16_ms" in m and "sdpa_fp16_ms" in m:
            cfg.notes.append(f"SDPA 实测 bf16 {m['sdpa_bf16_ms']:.1f} ms vs "
                             f"fp16 {m['sdpa_fp16_ms']:.1f} ms"
                             f"（kernel {m.get('sdpa_fp16_kernel', '?')}）")
        cfg.notes.append("GEMM 需要在线自适应 2 的幂预缩放：激活量级依赖分辨率/步数/提示词，"
                         "静态 k 表在 512px 上会溢出")
    else:
        for name in ("hybrid_linear", "attn_fp16", "decoder_fp16"):
            cfg.passes[name] = PassConfig(False)
        cfg.decoder_dtype = model_dtype
        if not bf16_native:
            cfg.notes.append(f"模型原生精度已是 {dtype_name(model_dtype)}，不需要跨精度迁移")
        elif capability.bf16_tensorcore:
            cfg.notes.append(f"sm_{capability.sm} 有 bf16 张量核，bf16 路径已是张量核，"
                             f"再迁移是净损失（多一次类型转换）")
        else:
            cfg.notes.append(f"sm_{capability.sm} 连 fp16 张量核都没有，本库无收益")
    if not capability.has_kernel_for_arch:
        cfg.notes.append(f"WARNING: torch arch_list 不含 sm_{capability.sm}（CUDA 13 已移除 Volta）")
    return cfg
