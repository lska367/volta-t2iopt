"""S4：VAE 解码能不能用 fp16 —— 单独拿一个已存 latent 试，不用重跑 DiT。

bf16 的 VAE 在这台机器上是 fp32 模拟的（sm_70 没有 bf16 张量核），1024² 解码 ~10 s。
VAE 是纯 conv + attention，是带宽活；fp16 下 cudnn 有原生 kernel。

风险是范围：latent 的量级 O(1)，但 conv 层的中间激活可能很大。先跑一遍看有没有 inf/nan，
再看和 bf16 出来的图差多少。若溢出，退一步只把 `decoder.mid_block` 留 bf16（`--keep-fp32`）。

用法：
    python t2i/check_vae.py --latent t2i/out/val_hybrid_1024_9steps3.latent.pt
    python t2i/check_vae.py --latent ... --keep-fp32 mid_block
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t2i.loader import CACHE_ROOT, DEFAULT_DIR  # noqa: E402
from t2i.validate import ssim  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def bench(fp, reps: int = 3) -> tuple[float, torch.Tensor]:
    for _ in range(1):
        fp()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        img = fp()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return sorted(ts)[len(ts) // 2], img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True)
    ap.add_argument("--model", default=str(DEFAULT_DIR))
    ap.add_argument("--keep-fp32", default="", help="逗号分隔的 fp16 保留为 bf16 的子模块名")
    args = ap.parse_args()

    from diffusers import AutoencoderKL

    lat = torch.load(args.latent, weights_only=False).to("cuda", torch.float32)
    print(f"latent {tuple(lat.shape)} |max|={lat.abs().max().item():.3f}")
    results = {}

    for name, dt in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        vae = AutoencoderKL.from_pretrained(Path(args.model) / "vae", dtype=dt).to("cuda")
        vae.eval()
        scale, shift = vae.config.scaling_factor, vae.config.shift_factor

        def fp(vae=vae, scale=scale, shift=shift):
            with torch.no_grad():
                x = (lat.to(vae.dtype) / scale) + shift
                return vae.decode(x, return_dict=False)[0]

        ms, img = bench(fp)
        t = img.float()
        arr = ((t / 2 + 0.5).clamp(0, 1)).permute(0, 2, 3, 1)[0].cpu().numpy()
        results[name] = dict(ms=round(ms, 1), img=arr, peak=torch.cuda.max_memory_allocated() / 2**30,
                             finite=bool(torch.isfinite(t).all()),
                             amax=float(img.abs().max()))
        print(f"[{name:4s}] {ms:7.1f} ms | 输出 |max|={img.abs().max().item():.4g} "
              f"finite={torch.isfinite(t).all().item()} | 统计 mean={arr.mean():.4f} std={arr.std():.4f}")
        del vae, img, t
        torch.cuda.empty_cache()

    a, b = results["bf16"]["img"], results["fp16"]["img"]
    if np.isfinite(b).all():
        print(f"\nfp16 vs bf16：SSIM = {ssim(a, b):.5f} | max|Δ| = {np.abs(a-b).max():.4f} | "
              f"mean|Δ| = {np.abs(a-b).mean():.5f}")
        print(f"耗时：{results['bf16']['ms']:.0f} → {results['fp16']['ms']:.0f} ms "
              f"({results['bf16']['ms']/results['fp16']['ms']:.2f}×)")
        from PIL import Image
        Image.fromarray((b * 255).round().astype(np.uint8)).save(OUT / "vae_fp16.png")
        print(f"-> {OUT / 'vae_fp16.png'}")
    else:
        print("\nfp16 输出含 inf/nan，不能用（需要 --keep-fp32 分段回退）")


if __name__ == "__main__":
    main()
