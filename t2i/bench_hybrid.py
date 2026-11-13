"""S2：混合精度（fp16 张量核 + 按层 2 的幂预缩放）的跑分与数值验证。

三种模式：
    python t2i/bench_hybrid.py --mode bf16   --size 1024 --steps 9 --tag base
    python t2i/bench_hybrid.py --mode hybrid --size 1024 --steps 9 --tag hyb
    python t2i/bench_hybrid.py --compare base hyb

每次跑都会落：`out/hyb_{tag}.png`、`out/latent_{tag}.pt`（去噪后的 latent，fp32）、
`out/bench_{tag}.json`（TE/每步/VAE 分解 + 峰值显存）。

`--compare` 给的是验收指标（`OPTIMIZATION_PLAN.md` §4）：
    latent 的 max|Δ| / relL2（门限 relL2 < 2e-3）、图像 SSIM（门限 ≥ 0.98）、有无 inf/nan。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from t2i.hybrid_linear import load_k_map, patch_linear  # noqa: E402
from t2i.loader import load_embeds, load_pipe  # noqa: E402
from t2i.prompts import BY_NAME  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"


def gaussian_kernel(n: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(n) - (n - 1) / 2
    k = np.exp(-(x**2) / (2 * sigma**2))
    k /= k.sum()
    return k


def _conv1d(a: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """沿指定轴做 same padding 的一维卷积（np.apply_along_axis 太慢，这里用错位累加）。"""
    pad = len(k) // 2
    ap = np.pad(a, [(pad, pad) if i == axis else (0, 0) for i in range(a.ndim)], mode="reflect")
    out = np.zeros_like(a)
    for i, w in enumerate(k):
        out += w * np.take(ap, range(i, i + a.shape[axis]), axis=axis)
    return out


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """灰度 SSIM（11×11 gaussian，sigma=1.5），够用于回归对比。"""
    a = img1.mean(axis=2) if img1.ndim == 3 else img1
    b = img2.mean(axis=2) if img2.ndim == 3 else img2
    k = gaussian_kernel()
    mu1 = _conv1d(_conv1d(a, k, 0), k, 1)
    mu2 = _conv1d(_conv1d(b, k, 0), k, 1)
    s11 = _conv1d(_conv1d(a * a, k, 0), k, 1) - mu1**2
    s22 = _conv1d(_conv1d(b * b, k, 0), k, 1) - mu2**2
    s12 = _conv1d(_conv1d(a * b, k, 0), k, 1) - mu1 * mu2
    c1, c2 = 0.01**2, 0.03**2
    m = ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1**2 + mu2**2 + c1) * (s11 + s22 + c2))
    return float(m.mean())


def run(args) -> None:
    tag = args.tag or args.mode
    pipe = load_pipe(torch.bfloat16)
    if args.mode == "hybrid":
        k_map = load_k_map(args.plan)
        patch_linear(pipe.transformer, "transformer", k_map)

    pe = load_embeds(prompt=args.prompt)

    step_events = [torch.cuda.Event(True) for _ in range(64)]
    n_cb = [0]

    def cb(_p, i, _t, kw):
        step_events[n_cb[0]].record()
        n_cb[0] += 1
        return kw

    gen = torch.Generator("cuda").manual_seed(args.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = pipe(prompt=None, prompt_embeds=pe, height=args.size, width=args.size,
               num_inference_steps=args.steps, guidance_scale=0.0, generator=gen,
               output_type="latent", return_dict=False, callback_on_step_end=cb)
    latents = res[0]
    torch.cuda.synchronize()
    total_ms = (time.time() - t0) * 1000

    n = n_cb[0]
    per = [step_events[i].elapsed_time(step_events[i + 1]) for i in range(n - 1)]
    denoise_ms = sum(per)

    # latent → 图像（用 VAE 单独跑，便于把 VAE 时间单独量出来）
    torch.cuda.synchronize()
    t1 = time.time()
    vae_in = latents.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    img = pipe.vae.decode(vae_in, return_dict=False)[0]
    img = pipe.image_processor.postprocess(img.detach(), output_type="pil")[0]
    torch.cuda.synchronize()
    vae_ms = (time.time() - t1) * 1000

    lat = latents.float().cpu()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(lat, OUT / f"latent_{tag}.pt")
    img.save(OUT / f"hyb_{tag}.png")
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0

    info = dict(mode=args.mode, tag=tag, size=args.size, steps=args.steps, forwards=n,
                total_ms=round(total_ms, 1), denoise_ms=round(denoise_ms, 1),
                ms_per_fwd=round(denoise_ms / max(len(per), 1), 1),
                per_fwd_ms=[round(v, 1) for v in per], vae_ms=round(vae_ms, 1),
                peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                lat_shape=list(lat.shape),
                lat_absmax=round(lat.abs().max().item(), 4),
                lat_std=round(lat.std().item(), 4),
                lat_finite=bool(torch.isfinite(lat).all()),
                img_std=round(float(arr.std()), 4), img_mean=round(float(arr.mean()), 4),
                dtype_hybrid=args.mode == "hybrid")
    (OUT / f"bench_{tag}.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"-> {OUT / f'latent_{tag}.pt'} / {OUT / f'hyb_{tag}.png'}")


def compare(a: str, b: str) -> None:
    la, lb = torch.load(OUT / f"latent_{a}.pt", weights_only=True), torch.load(OUT / f"latent_{b}.pt", weights_only=True)
    if la.shape != lb.shape:
        print(f"shape 不同：{tuple(la.shape)} vs {tuple(lb.shape)}")
        return
    d = (la - lb).abs()
    rel = (la - lb).norm().item() / la.norm().item()
    from PIL import Image

    ia = np.asarray(Image.open(OUT / f"hyb_{a}.png").convert("RGB"), dtype=np.float32) / 255.0
    ib = np.asarray(Image.open(OUT / f"hyb_{b}.png").convert("RGB"), dtype=np.float32) / 255.0
    s = ssim(ia, ib)
    psnr = 10 * np.log10(1.0 / max(float(((ia - ib) ** 2).mean()), 1e-12))
    print(f"\n=== {a} (bf16 基线) vs {b} ===")
    print(f"latent  max|Δ| {d.max().item():.5f} | mean|Δ| {d.mean().item():.6f} | relL2 {rel:.3e}"
          f"  （门限 < 2e-3 → {'PASS' if rel < 2e-3 else 'FAIL'}）")
    print(f"图像    SSIM {s:.4f}（门限 ≥ 0.98 → {'PASS' if s >= 0.98 else 'FAIL'}）| PSNR {psnr:.2f} dB"
          f" | max|Δ| {(ia-ib).__abs__().max():.4f}")
    print(f"数值    {a}: finite={torch.isfinite(la).all().item()} | {b}: finite={torch.isfinite(lb).all().item()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bf16", "hybrid"], default="hybrid")
    ap.add_argument("--compare", nargs=2, metavar=("BASE", "NEW"))
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="portrait", choices=list(BY_NAME))
    ap.add_argument("--plan", default=str(OUT / "profile_1024.json"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
