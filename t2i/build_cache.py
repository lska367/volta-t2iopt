"""S0：把"每轮实验都要付的加载成本"一次性付清。

**这台机器的真实约束**：30 GB 内存 + 8 GB swap（已用满过），顺序读盘约 100 MB/s。
把整条 pipeline（20 GB）读进 RAM 会直接把 swap 打满、加载退化成 100 MB/s 的抖动
（`/proc/pressure/io` full avg60 ≈ 59%）。所以这里的每一步都**只加载需要的组件**。

产出（在 /home/lskam/models/ 下，不进仓库）：
    z-image-turbo-bf16/                        bf16 权重（避免每次从 23 GB F32 转换）
    z-image-turbo-cache/prompt_embeds_bf16.pt  5 条固定 prompt 的文本条件（TE 只跑一次）

**不再做 fp16 权重缓存**：bf16→fp16 在 GPU 上是就地精转（12 GB 显存带宽，<0.1 s），
落 15 GB 到磁盘反而要多付 2.5 分钟的写盘。`t2i/loader.py` 里转换即可。

用法：
    python t2i/build_cache.py             # 缺什么建什么
    python t2i/build_cache.py --only embeds
    python t2i/build_cache.py --force --only embeds
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SRC = os.environ.get("ZIMAGE_SRC", "/home/lskam/models/z-image-turbo")
OUT_ROOT = Path(os.environ.get("ZIMAGE_CACHE_ROOT", "/home/lskam/models"))


class Timer:
    def __init__(self) -> None:
        self.t0 = time.time()

    def __call__(self, msg: str) -> float:
        dt = time.time() - self.t0
        print(f"[{dt:7.1f} s] {msg}", flush=True)
        self.t0 = time.time()
        return dt


def dir_size(d: Path) -> float:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**30 if d.exists() else 0.0


def build_bf16(src: Path, dst: Path, force: bool) -> None:
    """逐组件从 F32 原始仓库转 bf16。只碰 transformer / vae，TE 原盘已是 bf16。"""
    import shutil

    from diffusers import AutoencoderKL
    from diffusers.models.transformers import ZImageTransformer2DModel

    tick = Timer()
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("model_index.json",):
        shutil.copy2(src / f, dst / f)

    # 小件直接拷（config 而已）
    for comp in ("scheduler", "tokenizer", "text_encoder"):
        if (dst / comp).exists() and not force:
            continue
        shutil.copytree(src / comp, dst / comp, dirs_exist_ok=True)
        print(f"    copy {comp} ({dir_size(dst/comp):.2f} GiB)", flush=True)

    if force or not (dst / "transformer" / "config.json").exists():
        t = ZImageTransformer2DModel.from_pretrained(src / "transformer", dtype=torch.bfloat16)
        t.save_pretrained(dst / "transformer", safe_serialization=True)
        del t
        tick(f"transformer -> bf16 ({dir_size(dst/'transformer'):.1f} GiB)")

    if force or not (dst / "vae" / "config.json").exists():
        v = AutoencoderKL.from_pretrained(src / "vae", dtype=torch.bfloat16)
        v.save_pretrained(dst / "vae", safe_serialization=True)
        del v
        tick(f"vae -> bf16 ({dir_size(dst/'vae'):.2f} GiB)")


def build_embeds(bf16_dir: Path, cache: Path, force: bool) -> None:
    """只加载 Qwen3 TE + tokenizer 编码 5 条 prompt（不碰 transformer/vae）。"""
    from diffusers import ZImagePipeline
    from transformers import AutoTokenizer, Qwen3Model

    from t2i.prompts import PROMPTS

    tick = Timer()
    cache.mkdir(parents=True, exist_ok=True)
    te = Qwen3Model.from_pretrained(bf16_dir / "text_encoder", dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(bf16_dir / "tokenizer")
    tick(f"加载 TE（{sum(p.numel() for p in te.parameters())/1e9:.2f} B 参数）")

    # text_encoder / tokenizer 允许为 None 的组件，这里只需要 encode_prompt
    pipe = ZImagePipeline(scheduler=None, vae=None, text_encoder=te, tokenizer=tok, transformer=None)
    pipe.to("cuda")

    embeds: dict[str, list[torch.Tensor]] = {}
    for name, prompt in PROMPTS:
        # 注意：encode_prompt 不接受 num_images_per_prompt；返回 (positive, negative)
        pe = pipe.encode_prompt(prompt, device="cuda", do_classifier_free_guidance=False)
        feats = pe[0]
        assert isinstance(feats, (list, tuple)), f"预期 list[tensor]，拿到 {type(feats)}"
        embeds[name] = [t.detach().to("cpu", torch.bfloat16) for t in feats]
        print(f"    {name:10s} {len(feats)} 层 {[tuple(t.shape) for t in feats[:3]]}", flush=True)
    tick("编码 5 条 prompt")

    pt = cache / "prompt_embeds_bf16.pt"
    torch.save({"prompts": dict(PROMPTS), "embeds": embeds, "from_dtype": "bfloat16"}, pt)
    print(f"-> {pt} ({pt.stat().st_size/2**20:.1f} MiB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--only", choices=["bf16", "embeds"], default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    src = Path(args.src)
    bf16_dir = out_root / "z-image-turbo-bf16"
    cache = out_root / "z-image-turbo-cache"

    need_bf16 = args.only in ("", "bf16") and (args.force or not (bf16_dir / "transformer").exists())
    embeds_pt = cache / "prompt_embeds_bf16.pt"
    need_embeds = args.only in ("", "embeds") and (args.force or not embeds_pt.exists())

    print(f"src       : {src}\nbf16 缓存 : {'要建' if need_bf16 else '已存在'} -> {bf16_dir}"
          f"（当前 {dir_size(bf16_dir):.1f} GiB）\nembeds    : {'要建' if need_embeds else '已存在'} -> {embeds_pt}",
          flush=True)
    if not (need_bf16 or need_embeds):
        print("无事可做。")
        return

    if need_bf16:
        build_bf16(src, bf16_dir, args.force)
    if need_embeds:
        build_embeds(bf16_dir, cache, args.force)

    print(f"\nbf16 缓存 {dir_size(bf16_dir):.1f} GiB | embeds {embeds_pt.stat().st_size/2**20:.1f} MiB"
          if embeds_pt.exists() else f"\nbf16 缓存 {dir_size(bf16_dir):.1f} GiB")


if __name__ == "__main__":
    main()
