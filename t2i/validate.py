"""S2 的验收口径：同一 seed 下 bf16 基线 vs hybrid（fp16 张量核），比 latent / 图 / 时间。

一次运行同时产出：
    图片 PNG、去噪前的 latent（`.pt`，从 callback 的最后一个 step 抓）、
    TE(跳过)/每步去噪/VAE 的分阶段计时（CUDA event，不把 host 抖动算进去）、峰值显存。

用法：
    python t2i/validate.py --mode bf16   --size 1024 --steps 9 --tag s2base
    python t2i/validate.py --mode hybrid --size 1024 --steps 9 --tag s2hyb --runs 2
    python t2i/validate.py --compare out/s2base --compare out/s2hyb     # 比数值 + SSIM
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

from t2i.fp16_attention import maybe_enable as maybe_fp16_attn  # noqa: E402
from t2i.hybrid_linear import patch_linear  # noqa: E402
from t2i.loader import load_embeds, load_pipe  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
PROFILE = OUT / "profile_1024.json"


def sync() -> None:
    torch.cuda.synchronize()


def ssim(a: np.ndarray, b: np.ndarray, c1: float = 0.01**2, c2: float = 0.03**2) -> float:
    """全局 SSIM（11×11 均值滤波近似，够用来判"图有没有坏"）。支持 (H,W) 与 (H,W,C)。"""
    from numpy.lib.stride_tricks import sliding_window_view

    if a.ndim == 3:
        return float(np.mean([ssim(a[..., c], b[..., c], c1, c2) for c in range(a.shape[-1])]))

    k = 11
    pad = k // 2
    ap = np.pad(a, pad, mode="reflect")
    bp = np.pad(b, pad, mode="reflect")
    wa = sliding_window_view(ap, (k, k)).mean(axis=(-1, -2))
    wb = sliding_window_view(bp, (k, k)).mean(axis=(-1, -2))
    va = sliding_window_view(ap**2, (k, k)).mean(axis=(-1, -2)) - wa**2
    vb = sliding_window_view(bp**2, (k, k)).mean(axis=(-1, -2)) - wb**2
    cov = sliding_window_view(ap * bp, (k, k)).mean(axis=(-1, -2)) - wa * wb
    s = ((2 * wa * wb + c1) * (2 * cov + c2)) / ((wa**2 + wb**2 + c1) * (va + vb + c2))
    return float(s.mean())


def run(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fp16_attn:
        maybe_fp16_attn()
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32, "hybrid": torch.bfloat16}[args.mode]
    pipe = load_pipe(dtype, vae_dtype=torch.float16 if args.vae_fp16 else None, path=args.model)

    if args.mode == "hybrid":
        # 关掉 fp16 的 split-k 约减，走 fp32 累加（微基准里也更快：20.0 vs 19.9 TF）
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        patch_linear(pipe.transformer, "transformer")
        sync()
        print(f"[hybrid] 显存 {torch.cuda.memory_allocated()/2**30:.2f} GiB（权重已转 fp16）")

    prompt_embeds = load_embeds(None, args.prompt)

    # VAE 单独计时：callback 的间隔只覆盖 forward 之间，最后一次 forward 会漏到 total 里，
    # 所以直接包住 decode，别用减法估算。
    vae_t: list[list[torch.cuda.Event]] = []
    orig_decode = pipe.vae.decode

    def timed_decode(*a, **kw):
        ev = [torch.cuda.Event(True), torch.cuda.Event(True)]
        ev[0].record()
        out = orig_decode(*a, **kw)
        ev[1].record()
        vae_t.append(ev)
        return out

    pipe.vae.decode = timed_decode

    for run_i in range(args.runs):
        torch.cuda.reset_peak_memory_stats()
        vae_t.clear()
        step_ev = [torch.cuda.Event(True) for _ in range(64)]
        n_cb = [0]
        captured: dict[str, torch.Tensor] = {}

        def cb(_p, i, _t, kw):
            step_ev[n_cb[0]].record()
            n_cb[0] += 1
            if i == args.steps - 1:
                captured["latents"] = kw["latents"].detach().clone()
            return kw

        gen = torch.Generator("cuda").manual_seed(args.seed)
        sync()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        step_ev[0].record()
        e0.record()
        img = pipe(prompt=None, prompt_embeds=prompt_embeds, height=args.size, width=args.size,
                   num_inference_steps=args.steps, guidance_scale=0.0, generator=gen,
                   output_type="pil", callback_on_step_end=cb, return_dict=False)[0][0]
        e1.record()
        sync()

        total = e0.elapsed_time(e1)
        per = [step_ev[i].elapsed_time(step_ev[i + 1]) for i in range(n_cb[0] - 1)]
        vae = sum(e[0].elapsed_time(e[1]) for e in vae_t)
        denoise = total - vae
        peak = torch.cuda.max_memory_allocated() / 2**30
        n_fwd = n_cb[0]
        print(f"[{args.mode} run{run_i}] 总 {total/1000:6.2f} s | 去噪 {denoise/1000:6.2f} s "
              f"({denoise/1000/max(n_fwd,1)*1000:7.0f} ms/fwd, {n_fwd} 次 forward) | "
              f"VAE {vae/1000:5.2f} s | 峰值 {peak:5.1f} GiB", flush=True)

        if run_i == args.runs - 1:
            stem = OUT / f"val_{args.mode}_{args.size}_{args.steps}step{args.tag}"
            img.save(stem.with_suffix(".png"))
            if "latents" in captured:
                torch.save(captured["latents"], stem.with_suffix(".latent.pt"))
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            summary = dict(mode=args.mode, fp16_attn=args.fp16_attn, vae_fp16=args.vae_fp16, size=args.size, steps=args.steps, seed=args.seed,
                           prompt=args.prompt, forwards=len(per), total_ms=round(total, 1),
                           denoise_ms=round(denoise, 1),
                           ms_per_fwd=round(denoise / max(len(per), 1), 1),
                           vae_ms=round(vae, 1), peak_gib=round(peak, 2),
                           per_step_ms=[round(x, 1) for x in per],
                           img_mean=round(float(arr.mean()), 4), img_std=round(float(arr.std()), 4),
                           img_nan=bool(np.isnan(arr).any()), stem=stem.name)
            Path(f"{stem}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            print(json.dumps(summary, indent=2, ensure_ascii=False))
    pipe.vae.decode = orig_decode


def compare(a: str, b: str) -> None:
    """比两个 stem（`out/val_xxx_1024_9step`）的 latent 与图片。"""
    pa, pb = Path(a), Path(b)
    la, lb = torch.load(f"{pa}.latent.pt", weights_only=False), torch.load(f"{pb}.latent.pt", weights_only=False)
    diff = (lb - la).float()
    rel = diff.norm().item() / la.float().norm().item()
    print(f"latent  : relL2 = {rel:.3e} | max|Δ| = {diff.abs().max().item():.3e} | "
          f"|ref|max = {la.abs().max().item():.3e} | nan/inf = {not torch.isfinite(lb).all()}")

    from PIL import Image

    ia = np.asarray(Image.open(f"{pa}.png").convert("RGB"), dtype=np.float32) / 255.0
    ib = np.asarray(Image.open(f"{pb}.png").convert("RGB"), dtype=np.float32) / 255.0
    print(f"image   : SSIM = {ssim(ia, ib):.5f} | max|Δ| = {np.abs(ia - ib).max():.3f} | "
          f"mean|Δ| = {np.abs(ia - ib).mean():.5f}")

    ja = json.loads(Path(f"{pa}.json").read_text())
    jb = json.loads(Path(f"{pb}.json").read_text())
    print(f"时间    : fwd {ja['ms_per_fwd']:.0f} → {jb['ms_per_fwd']:.0f} ms "
          f"({ja['ms_per_fwd']/max(jb['ms_per_fwd'],1e-9):.2f}×) | "
          f"总 {ja['total_ms']/1000:.1f} → {jb['total_ms']/1000:.1f} s | "
          f"显存 {ja['peak_gib']:.1f} → {jb['peak_gib']:.1f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bf16", "fp32", "hybrid"], default="bf16",
                    help="fp32 只能配 --model /home/lskam/models/z-image-turbo（原始 F32 存储）")
    ap.add_argument("--model", default="/home/lskam/models/z-image-turbo-bf16")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="portrait")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--vae-fp16", action=argparse.BooleanOptionalAction, default=True,
                    help="VAE 解码用 fp16（实测 5.1 s → 0.87 s，SSIM 0.9997）")
    ap.add_argument("--fp16-attn", action=argparse.BooleanOptionalAction, default=True,
                    help="SDPA 输入转 fp16（sm_70 上才有 cutlass fmha）")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    t0 = time.time()
    run(args)
    print(f"[wall] {time.time()-t0:.0f} s")


if __name__ == "__main__":
    main()
