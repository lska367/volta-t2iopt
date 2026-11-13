"""离线产物缓存：把"每轮实验都要付的加载成本"一次性付清。

这台机器的真实约束：30 GB 内存 + 8 GB swap，顺序读盘约 100 MB/s。把整条 pipeline（20 GB）
读进 RAM 会直接把 swap 打满。所以缓存的每一步都**只加载需要的组件**。

两类产物（都落在 `cache_root()`，不进仓库）：

1. `z-image-turbo-bf16/`：bf16 权重。原盘是 23 GB 的 F32 存储，每次现转一遍要 2 分钟；
   注意**不再缓存 fp16 权重** —— bf16→fp16 在 GPU 上是就地精转（12 GB 显存带宽，<0.1 s），
   落 15 GB 到磁盘反而要多付 2.5 分钟的写盘。
2. `prompt_embeds_bf16.pt`：固定 prompt 的文本条件。Qwen3-4B 的 TE 在磁盘上 7.5 GB，
   而它只跑一次（0.33 s）；跳过它能把加载时间砍掉一大半。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from ..config import dtype_name
from ..models.base import GenerationConditions, ModelAdapter, WeightCacheSpec
from ..paths import cache_root
from ..prompts import PROMPTS


class Tick:
    def __init__(self) -> None:
        self.t0 = time.time()

    def __call__(self, msg: str) -> float:
        dt = time.time() - self.t0
        print(f"[{dt:7.1f} s] {msg}", flush=True)
        self.t0 = time.time()
        return dt


def dir_size(d: Path) -> float:
    if not d.exists():
        return 0.0
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**30


def _resolve(spec: str):
    import importlib
    mod_name, _, cls_name = spec.partition(":")
    return getattr(importlib.import_module(mod_name), cls_name)


def build_weight_cache(adapter: ModelAdapter, src: Path | None = None, dst: Path | None = None,
                       dtype: torch.dtype = torch.bfloat16, force: bool = False,
                       verbose: bool = True) -> Path:
    """把原始 F32 仓库逐组件转成目标精度另存（只碰 spec 里列出的组件）。"""
    import shutil

    spec: WeightCacheSpec | None = adapter.cache_spec
    if spec is None:
        raise ValueError(f"{adapter.name} 不需要权重缓存（原生精度即可）")
    root = cache_root()
    src = Path(src or root / adapter.source_dirname)
    dst = Path(dst or root / adapter.cache_dirname)

    tick = Tick()
    dst.mkdir(parents=True, exist_ok=True)
    for f in spec.copy_files:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    for comp in spec.copy_dirs:
        if not (src / comp).exists() or (dst / comp).exists():
            continue
        shutil.copytree(src / comp, dst / comp, dirs_exist_ok=True)
        if verbose:
            print(f"    copy {comp} ({dir_size(dst / comp):.2f} GiB)", flush=True)

    for comp, spec_str in spec.components.items():
        target = dst / comp
        if not force and (target / "config.json").exists():
            if verbose:
                print(f"    {comp} 已存在（{dir_size(target):.1f} GiB），跳过", flush=True)
            continue
        cls = _resolve(spec_str)
        model = cls.from_pretrained(src / comp, dtype=dtype)
        model.save_pretrained(target, safe_serialization=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        tick(f"{comp} -> {dtype_name(dtype)} ({dir_size(target):.1f} GiB)")
    return dst


def embed_cache_path(adapter: ModelAdapter, dtype: torch.dtype) -> Path:
    return cache_root() / f"{adapter.name}-cache" / f"prompt_embeds_{dtype_name(dtype)}.pt"


def build_prompt_embeds(adapter: ModelAdapter, *, weights_dir: Path | None = None,
                        dtype: torch.dtype = torch.bfloat16, force: bool = False,
                        device: str = "cuda", verbose: bool = True) -> Path:
    """只加载 TE 跑一遍固定 prompt，把条件存成 .pt。"""
    path = embed_cache_path(adapter, dtype)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tick = Tick()
    pipe = adapter.load(path=weights_dir, model_dtype=dtype, decoder_dtype=dtype,
                        device=device, with_text_encoder=True, verbose=verbose)
    tick("加载整条 pipeline（含 TE）")

    payload: dict[str, Any] = {"prompts": dict(PROMPTS), "from_dtype": dtype_name(dtype),
                               "embeds": {}}
    for name, prompt in PROMPTS:
        cond = adapter.encode(pipe, prompt, device=device)
        payload["embeds"][name] = dict(
            tensors=[t.detach().to("cpu", dtype) for t in cond.tensors],
            extras={k: (v.detach().to("cpu", dtype) if torch.is_tensor(v) else v)
                    for k, v in cond.extras.items()},
        )
        if verbose:
            shapes = [tuple(t.shape) for t in cond.tensors[:2]]
            print(f"    {name:10s} {len(cond.tensors)} 层 {shapes}", flush=True)
    tick(f"编码 {len(PROMPTS)} 条 prompt")
    torch.save(payload, path)
    if verbose:
        print(f"-> {path} ({path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return path


# 迁移期兼容：旧脚本（t2i/build_cache.py）的文件名与格式
_LEGACY_NAMES = {torch.bfloat16: "prompt_embeds_bf16.pt", torch.float16: "prompt_embeds_fp16.pt"}


def _find_embed_file(adapter: ModelAdapter, dtype: torch.dtype) -> Path | None:
    path = embed_cache_path(adapter, dtype)
    if path.exists():
        return path
    legacy = path.with_name(_LEGACY_NAMES.get(dtype, "prompt_embeds_bf16.pt"))
    return legacy if legacy.exists() else None


def load_conditions(adapter: ModelAdapter, prompt: str, *, dtype: torch.dtype | None = None,
                    device: str = "cuda") -> GenerationConditions:
    """从缓存读条件；缓存缺失就报错并给出构建命令（不静默回退到加载 TE）。"""
    dt = dtype or torch.bfloat16
    path = _find_embed_file(adapter, dt)
    if path is None:
        raise FileNotFoundError(
            f"没有条件缓存 {embed_cache_path(adapter, dt)}；先跑："
            f"volta-t2iopt cache embeds --model {adapter.name}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    key = prompt
    if key not in blob["embeds"]:
        from ..prompts import BY_NAME
        key = next((k for k, v in BY_NAME.items() if v == prompt), None)
    if key is None or key not in blob["embeds"]:
        avail = sorted(blob["embeds"])
        raise KeyError(f"条件缓存里没有这条 prompt（可用 {avail}）；或用完整 prompt 重建缓存")
    item = blob["embeds"][key]
    if isinstance(item, list):          # 旧格式：直接是 list[tensor]
        item = dict(tensors=item, extras={})
    return GenerationConditions(prompt=key,
                                tensors=[t.to(device) for t in item["tensors"]],
                                extras={k: (v.to(device) if torch.is_tensor(v) else v)
                                        for k, v in item.get("extras", {}).items()})


def prepare_conditions(pipe, adapter: ModelAdapter, prompt: str, *, use_cache: bool = True,
                       device: str = "cuda") -> GenerationConditions:
    """优先用缓存；没有缓存就走 TE（要求 pipe 里有 TE）。"""
    if use_cache:
        try:
            return load_conditions(adapter, prompt, dtype=torch.bfloat16, device=device)
        except (FileNotFoundError, KeyError):
            pass
    if getattr(pipe, "text_encoder", None) is None:
        raise RuntimeError("pipeline 没有文本编码器，也没有条件缓存；"
                           "先跑 volta-t2iopt cache embeds")
    return adapter.encode(pipe, prompt, device=device).to(device)
