"""Z-Image-Turbo @ V100(sm_70, 32GB)：一次加载，扫多组 (分辨率, 步数)，分阶段计时。

用法：
    python t2i/sweep_zimage.py --dtype fp16
    python t2i/sweep_zimage.py --dtype bf16 --sizes 1024 --steps 9   # 对照：V100 上 bf16 是 fp32 模拟

分阶段的口径：
    total = TE(文本编码) + denoise(N 次 DiT forward) + VAE decode
    per_step 由 callback_on_step_end 里的 CUDA event 差值得到
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

MODEL = os.environ.get("ZIMAGE_PATH", "/home/lskam/models/z-image-turbo")
OUT = Path(__file__).resolve().parent / "out"

PROMPT = ("A photorealistic close-up portrait of an elderly fisherman mending a net at dawn, "
          "weathered face, soft rim light, shallow depth of field, 85mm lens")


def sync() -> None:
    torch.cuda.synchronize()


def run_one(pipe, size: int, steps: int, prompt_embeds, seed: int = 0) -> dict:
    step_events = [torch.cuda.Event(True) for _ in range(64)]
    n_cb = [0]

    def cb(_p, _i, _t, kw):
        step_events[n_cb[0]].record()
        n_cb[0] += 1
        return kw

    gen = torch.Generator("cuda").manual_seed(seed)
    sync()
    torch.cuda.reset_peak_memory_stats()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    res = pipe(prompt=None, prompt_embeds=prompt_embeds, height=size, width=size,
               num_inference_steps=steps, guidance_scale=0.0, generator=gen,
               output_type="pil", callback_on_step_end=cb, return_dict=False)
    e1.record()
    # return_dict=False → (images,)，images 是 list[PIL]
    imgs = res[0] if isinstance(res, (tuple, list)) else res.images
    img = imgs[0] if isinstance(imgs, (list, tuple)) else imgs
    sync()

    total = e0.elapsed_time(e1)
    n = n_cb[0]
    per = [step_events[i].elapsed_time(step_events[i + 1]) for i in range(n - 1)]
    denoise = sum(per)
    return dict(size=size, steps=steps, forwards=n, total_ms=total, denoise_ms=denoise,
                ms_per_step=denoise / max(len(per), 1), vae_ms=total - denoise,
                peak_gib=torch.cuda.max_memory_allocated() / 2**30, img=img, per=per)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 768, 512])
    ap.add_argument("--steps", type=int, nargs="+", default=[9, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-fp16", default="", help="把 fp16 权重另存一份，后续加载快一倍")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | {args.dtype}", flush=True)

    from diffusers import ZImagePipeline

    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=dtype).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    sync()
    load_s = time.time() - t0
    weights_gib = torch.cuda.memory_allocated() / 2**30
    print(f"[load] {load_s:.0f} s | 权重 {weights_gib:.1f} GiB", flush=True)

    if args.save_fp16:
        pipe.save_pretrained(args.save_fp16, safe_serialization=True)
        print(f"[save] fp16 权重 -> {args.save_fp16}", flush=True)

    # 文本编码单独计时：先预热一次（首次调用包含 cudnn/module 初始化，不预热会高估 100 倍）
    pipe.encode_prompt(PROMPT, device="cuda", do_classifier_free_guidance=False)
    sync()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    pe = pipe.encode_prompt(PROMPT, device="cuda", do_classifier_free_guidance=False)
    e1.record()
    sync()
    prompt_embeds = pe[0] if isinstance(pe, (tuple, list)) else pe
    te_ms = e0.elapsed_time(e1)

    def _desc(e):
        # Z-Image 的文本条件是多层特征（list[tensor]），不是单个 embedding
        if isinstance(e, (list, tuple)):
            return "list[" + ", ".join(str(tuple(x.shape)) for x in e[:2]) + f", …{len(e)}]"
        return str(tuple(e.shape))

    print(f"[TE] {te_ms:.0f} ms | prompt_embeds {_desc(prompt_embeds)}", flush=True)

    rows = []
    for size in args.sizes:
        for steps in args.steps:
            r = run_one(pipe, size, steps, prompt_embeds, args.seed)
            arr = np.asarray(r["img"].convert("RGB"), dtype=np.float32) / 255.0
            name = f"zimage_{args.dtype}_{size}_{steps}step{args.tag}.png"
            r["img"].save(OUT / name)
            tok = (size // 16) ** 2
            row = dict(dtype=args.dtype, size=size, steps=steps, forwards=r["forwards"],
                       tokens=tok, te_ms=round(te_ms, 1), total_ms=round(r["total_ms"], 1),
                       denoise_ms=round(r["denoise_ms"], 1), ms_per_step=round(r["ms_per_step"], 1),
                       vae_ms=round(r["vae_ms"], 1), peak_gib=round(r["peak_gib"], 2),
                       img_mean=round(float(arr.mean()), 4), img_std=round(float(arr.std()), 4),
                       nan=bool(np.isnan(arr).any()), file=name)
            rows.append(row)
            print(f"[{args.dtype} {size}px {steps}steps] total {r['total_ms']/1000:6.2f} s | "
                  f"denoise {r['denoise_ms']/1000:6.2f} s ({r['ms_per_step']:.0f} ms/fwd) | "
                  f"VAE {r['vae_ms']/1000:5.2f} s | peak {r['peak_gib']:.1f} GiB | "
                  f"std {row['img_std']:.3f}{' NaN!' if row['nan'] else ''}", flush=True)

    out = OUT / f"sweep_{args.dtype}{args.tag}.json"
    out.write_text(json.dumps(dict(load_s=round(load_s, 1), weights_gib=round(weights_gib, 2),
                                   rows=rows), indent=2))
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
