"""模型适配器：新增一个模型 = 新增一个文件，pass 层不用动。"""

from __future__ import annotations

from pathlib import Path

from .base import GenerationConditions, ModelAdapter, WeightCacheSpec
from .generic import GenericAdapter
from .sdxl import SDXLAdapter
from .zimage import ZImageAdapter

REGISTRY: dict[str, type[ModelAdapter]] = {
    "z-image-turbo": ZImageAdapter,
    "zimage": ZImageAdapter,
    "sdxl-turbo": SDXLAdapter,
    "sdxl": SDXLAdapter,
}

_BY_CLASS = {
    "ZImagePipeline": ZImageAdapter,
    "StableDiffusionXLPipeline": SDXLAdapter,
    "StableDiffusionPipeline": GenericAdapter,
}


def get_adapter(model: str | ModelAdapter | None) -> ModelAdapter:
    """按名字 / 路径 / 已构造实例拿适配器（路径会读 model_index.json 的 _class_name）。"""
    if isinstance(model, ModelAdapter):
        return model
    if not model:
        return ZImageAdapter()
    key = str(model).lower()
    if key in REGISTRY:
        return REGISTRY[key]()
    path = Path(str(model))
    index = path / "model_index.json"
    if index.exists():
        import json
        cls = json.loads(index.read_text()).get("_class_name", "")
        if cls in _BY_CLASS:
            return _BY_CLASS[cls]()
        return GenericAdapter()
    raise KeyError(f"未知模型 {model!r}；可用 {sorted(REGISTRY)}，或传 model_index.json 所在目录")


def detect_adapter(pipe) -> ModelAdapter:
    """从已构造的 pipeline 反查适配器（`optimize(pipe)` 走这条路）。"""
    cls_name = type(pipe).__name__
    if cls_name in _BY_CLASS:
        return _BY_CLASS[cls_name]()
    if hasattr(pipe, "vae") and (hasattr(pipe, "transformer") or hasattr(pipe, "unet")):
        return GenericAdapter()
    raise TypeError(f"无法为 {cls_name} 推断适配器；请显式传 adapter=")


__all__ = ["ModelAdapter", "WeightCacheSpec", "GenerationConditions", "GenericAdapter",
           "SDXLAdapter", "ZImageAdapter", "REGISTRY", "get_adapter", "detect_adapter"]
