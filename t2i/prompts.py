"""所有 Z-Image 实验共用的固定 prompt（画质对比的唯一事实源）。

5 条：2 条英文、1 条中文、1 条英文文字渲染、1 条中文文字渲染 —— 对应验证协议
（`OPTIMIZATION_PLAN.md` §4）里"固定 5 条 prompt，含中英文字渲染各 1 条"。
"""

PROMPTS: list[tuple[str, str]] = [
    ("portrait", "A photorealistic close-up portrait of an elderly fisherman mending a net at dawn, "
                 "weathered face, soft rim light, shallow depth of field, 85mm lens"),
    ("landscape", "A lone lighthouse on a basalt cliff during a storm, huge waves breaking below, "
                  "dramatic overcast sky, cold blue tones, long exposure"),
    ("zh_scene", "江南水乡的清晨，石桥上有人挑担走过，河面薄雾，白墙黑瓦，远处传来评弹声，写实摄影"),
    ("text_en", "A minimalist poster with the word \"TURBO\" in bold white sans-serif letters, "
                "centered on a deep teal background, subtle film grain"),
    ("text_zh", "一张极简风格的海报，中间写着四个白色粗体汉字「极速生成」，背景是深红色渐变，胶片颗粒感"),
]

BY_NAME = {name: p for name, p in PROMPTS}
DEFAULT = PROMPTS[0][1]


def main() -> None:  # 方便 `python t2i/prompts.py` 检查
    for name, p in PROMPTS:
        print(f"{name:10s} {p}")


if __name__ == "__main__":
    main()
