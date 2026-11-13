"""Z-Image-Turbo 在 V100(sm_70, 32GB) 上的实测：分阶段计时 + 显存峰值。

用法：
    python t2i/bench_zimage.py --steps 8 --size 1024 --dtype fp16
    python t2i/bench_zimage.py --steps 8 --size 1024 --dtype bf16   # 对照：V100 上 bf16 是模拟的

为什么要分阶段：DiT 的端到端时间 = 文本编码 + N 步去噪 + VAE 解码。
不看分解的话，容易把"8 步去噪很慢"和"VAE/编码器很慢"混在一起。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

MODEL = os.environ.get("ZIMAGE_PATH", "/home/lskam/models/z-image-turbo")
OUT = Path(__file__).resolve().parent / "out"


def sync() -> None:
    torch.cuda.synchronize()


class Phase:
    """CUDA event 打点，避免 host 计时把 launch 抖动算进去。"""

    def __init__(self) -> None:
        self.t = {}

    def __call__(self, name: str):
        self.t.setdefault(name, []).append(torch.cuda.Event(enable_timing=True))
        return self.t[name][-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=2, help="第二次及以后是热缓存，取最后一次")
    ap.add_argument("--prompt", default="A photorealistic close-up portrait of an elderly fisherman "
                                       "mending a net at dawn, weathered face, soft rim light, 85mm lens")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | dtype={args.dtype}")

    from diffusers import ZImagePipeline

    t0 = time.time()
    pipe = ZImagePipeline.from_pretrained(MODEL, torch_dtype=dtype)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    sync()
    load_s = time.time() - t0
    mem_after_load = torch.cuda.memory_allocated() / 2**30
    print(f"[load] {load_s:.1f} s, 权重占用 {mem_after_load:.1f} GiB")

    torch.cuda.reset_peak_memory_stats()

    results = []
    for run in range(args.runs):
        # --- 文本编码 ---
        sync()
        e_te0, e_te1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e_te0.record()
        pe = pipe.encode_prompt(args.prompt, device="cuda", num_images_per_prompt=1,
                                do_classifier_free_guidance=False)
        prompt_embeds = pe[0] if isinstance(pe, (tuple, list)) else pe
        e_te1.record()

        # --- 去噪（用 callback 记每步）---
        # 注意：Turbo 的 9 步其实只有 8 次 DiT forward（最后一次只做 x0 预测），
        # 所以这里按"实际回调次数"统计，不硬编码步数。
        step_events = [torch.cuda.Event(True) for _ in range(64)]
        n_cb = [0]

        def cb(_p, i, _t, kw):
            step_events[n_cb[0]].record()
            n_cb[0] += 1
            return kw

        gen = torch.Generator("cuda").manual_seed(args.seed)
        sync()
        e_all0, e_all1 = torch.cuda.Event(True), torch.cuda.Event(True)
        step_events[0].record()
        e_all0.record()
        img = pipe(
            prompt=None,
            prompt_embeds=prompt_embeds,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            generator=gen,
            output_type="pil",
            callback_on_step_end=cb,
            return_dict=False,
        )[0]
        e_all1.record()
        sync()

        te_ms = e_te0.elapsed_time(e_te1)
        total_ms = e_all0.elapsed_time(e_all1)
        n = n_cb[0]
        per_step = [step_events[i].elapsed_time(step_events[i + 1]) for i in range(n - 1)]
        denoise_ms = sum(per_step)
        vae_ms = total_ms - denoise_ms

        peak = torch.cuda.max_memory_allocated() / 2**30
        results.append(dict(te_ms=te_ms, denoise_ms=denoise_ms, per_step_ms=per_step,
                            vae_ms=vae_ms, total_ms=total_ms, peak_gib=peak))
        print(f"[run {run}] TE {te_ms:7.1f} ms | 8步去噪 {denoise_ms:8.1f} ms "
              f"({sum(per_step)/max(len(per_step),1):6.1f} ms/step) | VAE {vae_ms:6.1f} ms | "
              f"总 {total_ms:8.1f} ms | 峰值显存 {peak:5.1f} GiB")

    r = results[-1]
    import numpy as np
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    name = f"zimage_{args.size}_{args.steps}step_{args.dtype}{args.tag}.png"
    img.save(OUT / name)

    summary = {
        "model": "Tongyi-MAI/Z-Image-Turbo",
        "dtype": args.dtype, "size": args.size, "steps": args.steps,
        "load_s": round(load_s, 1), "weights_gib": round(mem_after_load, 2),
        "te_ms": round(r["te_ms"], 1), "denoise_ms": round(r["denoise_ms"], 1),
        "ms_per_step": round(r["denoise_ms"] / max(len(r["per_step_ms"]), 1), 1),
        "vae_ms": round(r["vae_ms"], 1), "total_ms": round(r["total_ms"], 1),
        "peak_gib": round(r["peak_gib"], 2),
        "img_mean": round(float(arr.mean()), 4), "img_std": round(float(arr.std()), 4),
        "img_min": float(arr.min()), "img_max": float(arr.max()),
        "nan": bool(np.isnan(arr).any()),
    }
    (OUT / f"summary_{args.size}_{args.steps}step_{args.dtype}{args.tag}.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"图片: {OUT / name}")


if __name__ == "__main__":
    main()
