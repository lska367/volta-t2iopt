"""validate：跑一档并和基线 / fp32 真值对拍，输出 PASS / FAIL。"""

from __future__ import annotations

import argparse

from ..paths import results_root
from ..runtime.generate import generate
from ..runtime.loader import load_optimized
from ..validate.gate import GateCriteria, compare_files
from ._common import add_gate_args, add_model_args


def add_parser(sub, name: str = "validate") -> None:
    p = sub.add_parser(name, help="跑一档 + 质量门限判定")
    add_model_args(p)
    add_gate_args(p)
    p.add_argument("--runs", type=int, default=1)


def run(args) -> int:
    pipe, sess = load_optimized(args.model, preset=args.preset, path=args.path,
                                measure=args.measure, verbose=not args.quiet)
    print(sess.summary())
    size = args.size or sess.adapter.default_size
    steps = args.steps or sess.adapter.default_steps
    stem = f"val_{sess.adapter.name}_{args.preset}_{size}_{steps}step{args.tag}"
    res = generate(pipe, sess.adapter, prompt=args.prompt, size=size, steps=steps,
                   seed=args.seed, capture_latents=True, out_dir=args.out, stem=stem,
                   verbose=not args.quiet,
                   extra_meta=dict(preset=args.preset, passes=[r.line() for r in sess.reports]))
    stem_path = res.files.get("json", stem).replace(".json", "")

    criteria = GateCriteria(min_ssim=args.min_ssim, max_rel_l2=args.max_rel_l2,
                            max_latency_ratio=args.max_latency_ratio)
    rc = 0
    for label, ref in (("质量判据 vs fp32 真值", args.fp32_ref),
                       ("诊断对比 vs baseline", args.baseline)):
        if not ref:
            continue
        print(f"\n--- {label} ---")
        gate = compare_files(ref, stem_path,
                             criteria if "fp32" in label else
                             GateCriteria(min_ssim=0.0, max_rel_l2=criteria.max_rel_l2,
                                          forbid_nonfinite=criteria.forbid_nonfinite,
                                          reference="bf16"))
        print(gate.format())
        if "fp32" in label and not gate.passed:
            rc = 1
    if not (args.fp32_ref or args.baseline):
        print("\n(没有 --baseline / --fp32-ref，只产出结果不判门限)")
    print(f"\n结果：{stem_path}.{{png,json,latent.pt}}  (目录 {args.out or results_root()})")
    return rc
