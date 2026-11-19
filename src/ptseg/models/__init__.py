from typing import Any

import torch.nn as nn

from .segformer3d import SegFormer3D
from .swin_unetr import SwinUNETR3D
from .unetr import UNETR3D

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "unetr": UNETR3D,
    "swin_unetr": SwinUNETR3D,
    "segformer3d": SegFormer3D,
}

def available_models() -> list[str]:
    return sorted(MODEL_REGISTRY)

def build_model(name: str, **kwargs: Any) -> nn.Module:
    key = str(name).lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {', '.join(available_models())}.")
    return MODEL_REGISTRY[key](**kwargs)

def describe_model(model: nn.Module) -> dict:
    describe = getattr(model, "describe", None)
    return describe() if callable(describe) else {}
