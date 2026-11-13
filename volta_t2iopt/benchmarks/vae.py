"""S4 的证据：解码能否直接用 fp16（拿一个已存 latent 试，不用重跑 DiT）。

VAE 是纯 conv + attention 的带宽活；sm_70 上 cudnn 有原生 fp16 kernel。
实测：1024² 解码 5093 → 874 ms，与 bf16 输出 SSIM 0.99966，无 inf/nan。
若某天溢出，退一步只把 `decoder.mid_block` 留原精度（`--keep-fp32 mid_block`）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..paths import cache_root
from ..validate.metrics import ssim


def bench(fn, reps: int = 3) -> tuple[float, torch.Tensor]:
    fn()
    torch.cuda.synchronize()
    times, img = [], None
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        img = fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2], img


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="volta-t2iopt microbench vae")
    ap.add_argument("--latent", default="", help=".latent.pt（generate --latents 或 validate 产出）")
    ap.add_argument("--model", default=str(cache_root() / "z-image-turbo-bf16"))
    ap.add_argument("--keep-fp32", default="", help="逗号分隔的子模块名，保持原精度")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    if not args.latent:
        ap.error("需要 --latent（可先用 volta-t2iopt bench --latents 产出）")

    from diffusers import AutoencoderKL

    lat = torch.load(args.latent, map_location="cuda", weights_only=False).float()
    print(f"latent {tuple(lat.shape)} |max|={lat.abs().max().item():.3f}")
    results = {}
    for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        vae = AutoencoderKL.from_pretrained(Path(args.model) / "vae", dtype=dt).to("cuda")
        vae.eval()
        for sub in filter(None, (s.strip() for s in args.keep_fp32.split(","))):
            if hasattr(vae, sub):
                getattr(vae, sub).to(torch.bfloat16)
        scale, shift = vae.config.scaling_factor, vae.config.shift_factor

        def fp(vae=vae, scale=scale, shift=shift):
            with torch.no_grad():
                return vae.decode((lat.to(vae.dtype) / scale) + shift, return_dict=False)[0]

        ms, img = bench(fp)
        arr = ((img.float() / 2 + 0.5).clamp(0, 1)).permute(0, 2, 3, 1)[0].cpu().numpy()
        results[name] = dict(ms=ms, img=arr,
                             finite=bool(torch.isfinite(img).all()))
        print(f"[{name:4s}] {ms:7.1f} ms | 输出 |max|={img.abs().max().item():.4g} "
              f"finite={results[name]['finite']} | mean={arr.mean():.4f} std={arr.std():.4f}")
        del vae, img
        torch.cuda.empty_cache()

    a, b = results["bf16"]["img"], results["fp16"]["img"]
    if np.isfinite(b).all():
        print(f"\nfp16 vs bf16：SSIM = {ssim(a, b):.5f} | max|Δ| = {np.abs(a - b).max():.4f} | "
              f"mean|Δ| = {np.abs(a - b).mean():.5f}")
        print(f"耗时：{results['bf16']['ms']:.0f} → {results['fp16']['ms']:.0f} ms "
              f"({results['bf16']['ms'] / results['fp16']['ms']:.2f}×)")
        if args.out:
            from PIL import Image
            Image.fromarray((b * 255).round().astype(np.uint8)).save(args.out)
            print(f"-> {args.out}")
    else:
        print("\nfp16 输出含 inf/nan，不能用（需要 --keep-fp32 分段回退）")
    return 0
