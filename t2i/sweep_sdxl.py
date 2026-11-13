"""SDXL-turbo fp16 基线：证明 V100 上"fp16-native 模型"能跑到秒级。"""
import argparse, json, time
import numpy as np
import torch

OUT = __import__("pathlib").Path(__file__).resolve().parent / "out"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/lskam/models/sdxl-turbo")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 512])
    ap.add_argument("--steps", type=int, nargs="+", default=[4])
    args = ap.parse_args()
    from diffusers import StableDiffusionXLPipeline

    t0 = time.time()
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16,
                                                     variant="fp16", use_safetensors=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    torch.cuda.synchronize()
    print(f"[load] {time.time()-t0:.0f}s | weights {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    rows = []
    for size in args.sizes:
        for steps in args.steps:
            ev = [torch.cuda.Event(True) for _ in range(64)]
            n = [0]

            def cb(_p, _i, _t, kw, ev=ev, n=n):
                ev[n[0]].record(); n[0] += 1; return kw

            torch.cuda.reset_peak_memory_stats()
            g = torch.Generator("cuda").manual_seed(0)
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            img = pipe(prompt="A photorealistic close-up portrait of an elderly fisherman mending a net at dawn",
                       height=size, width=size, num_inference_steps=steps, guidance_scale=0.0,
                       generator=g, output_type="pil", callback_on_step_end=cb, return_dict=False)[0][0]
            e.record()
            torch.cuda.synchronize()
            per = [ev[i].elapsed_time(ev[i + 1]) for i in range(n[0] - 1)]
            total, den = s.elapsed_time(e), sum(per)
            a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
            img.save(OUT / f"sdxl_fp16_{size}_{steps}.png")
            row = dict(size=size, steps=steps, total_s=round(total / 1000, 2),
                       unet_s=round(den / 1000, 2), ms_per_step=round(den / max(len(per), 1)),
                       vae_te_s=round((total - den) / 1000, 2),
                       peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                       std=round(float(a.std()), 3), nan=bool(np.isnan(a).any()))
            rows.append(row)
            print(f"[SDXL fp16 {size}px {steps}step] total {row['total_s']:5.2f}s | UNet {row['unet_s']:5.2f}s "
                  f"({row['ms_per_step']} ms/step) | VAE+TE {row['vae_te_s']:4.2f}s | "
                  f"peak {row['peak_gib']:.1f} GiB", flush=True)
    (OUT / "sweep_sdxl.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
