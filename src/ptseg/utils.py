import copy
import hashlib
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml

LOGGER_NAME = "ptseg"

def set_determinism(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass
    else:
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def resolve_device(policy: str = "auto", require_cuda: bool = False) -> torch.device:
    policy = (policy or "auto").lower()
    cuda_available = torch.cuda.is_available()

    if policy == "cuda":
        if not cuda_available:
            raise RuntimeError("run.device=cuda but CUDA is not available on this machine.")
        return torch.device("cuda")

    if policy == "cpu":
        if require_cuda:
            raise RuntimeError("run.require_cuda=true but run.device=cpu.")
        return torch.device("cpu")

    if policy != "auto":
        raise ValueError(f"Unknown device policy: {policy!r} (expected auto|cpu|cuda).")

    if require_cuda and not cuda_available:
        raise RuntimeError("run.require_cuda=true but CUDA is not available on this machine.")
    return torch.device("cuda" if cuda_available else "cpu")

def device_summary(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"cuda ({name}, {total:.1f} GiB)"
    return "cpu"

def get_logger(name: str, log_file: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in logger.handlers):
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
        logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        existing = {getattr(h, "baseFilename", None) for h in logger.handlers}
        if str(log_file.resolve()) not in existing:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
            logger.addHandler(handler)

    return logger

class Timer:

    def __init__(self) -> None:
        self.elapsed = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start

def format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m {secs:02d}s"

def deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged

def load_yaml(path: str | Path) -> dict:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")

    parent = data.pop("extends", None)
    if parent is None:
        return data

    parent_path = (path.parent / parent).resolve()
    return deep_merge(load_yaml(parent_path), data)

def parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text

def apply_overrides(config: dict, overrides: Iterable[str]) -> dict:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        dotted, raw = item.split("=", 1)
        keys = [part for part in dotted.strip().split(".") if part]
        if not keys:
            raise ValueError(f"Override without a key: {item!r}")

        cursor = config
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = parse_value(raw.strip())
    return config

def short_id(payload: Any, length: int = 8) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]

def dump_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

def dump_yaml(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def as_tuple3(value: int | Sequence[int], name: str = "value") -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    items = tuple(int(v) for v in value)
    if len(items) != 3:
        raise ValueError(f"{name} must be an int or a sequence of 3 ints, got {value!r}")
    return items

def divisible_by(value: int, divisor: int) -> bool:
    return divisor > 0 and value % divisor == 0

def apply_intensity_window(array: np.ndarray, low: float, high: float,
                           out_range: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    clipped = np.clip(array, low, high)
    if high > low:
        scaled = (clipped - low) / (high - low)
    else:
        scaled = np.zeros_like(clipped, dtype=np.float32)
    low_out, high_out = out_range
    return scaled * (high_out - low_out) + low_out


def pad_to_multiple(x: torch.Tensor, multiple: int | tuple[int, int, int],
                    mode: str = "constant") -> tuple[torch.Tensor, tuple[int, int, int]]:
    dims = x.shape[-3:]
    step = as_tuple3(multiple, "multiple") if not isinstance(multiple, int) else (multiple,) * 3
    original = (int(dims[0]), int(dims[1]), int(dims[2]))
    pad = [max(0, -(-original[i] // step[i]) * step[i] - original[i]) for i in range(3)]

    if any(p > 0 for p in pad):
        x = torch.nn.functional.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]), mode=mode)
    return x, original
