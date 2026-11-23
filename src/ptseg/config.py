from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .utils import as_tuple3, divisible_by, load_yaml

@dataclass
class RunConfig:
    seed: int = 42
    device: str = "auto"
    require_cuda: bool = False
    deterministic: bool = True
    output_dir: str = "outputs/runs"
    run_name: str | None = None

    @classmethod
    def from_dict(cls, raw: dict | None) -> "RunConfig":
        return cls(**(raw or {}))

@dataclass
class DataConfig:
    root: str
    dataset_json: str | None = None
    valid_ratio: float = 0.2
    roi_size: tuple[int, int, int] = (96, 96, 96)
    pixdim: tuple[float, float, float] = (1.5, 1.5, 1.5)
    hu_window: tuple[float, float] = (-175.0, 250.0)
    intensity_range: tuple[float, float] = (0.0, 1.0)
    num_samples: int = 4
    batch_size: int = 2
    num_workers: int = 4
    cache_rate: float = 0.5
    augment: bool = True
    labels: dict[str, int] = field(default_factory=lambda: {"background": 0, "pancreas": 1, "tumor": 2})

    @classmethod
    def from_dict(cls, raw: dict) -> "DataConfig":
        raw = dict(raw or {})
        for key in ("roi_size", "pixdim", "hu_window", "intensity_range"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    @property
    def num_classes(self) -> int:
        return len(self.labels)

    @property
    def foreground_classes(self) -> list[int]:
        return [index for name, index in self.labels.items() if index != 0]

@dataclass
class EarlyStoppingConfig:
    enabled: bool = True
    patience: int = 40
    min_delta: float = 0.001
    mode: str = "max"

    @classmethod
    def from_dict(cls, raw: dict | None) -> "EarlyStoppingConfig":
        return cls(**(raw or {}))

@dataclass
class LossConfig:
    name: str = "dice_ce"
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    class_weights: Sequence[float] | None = None
    include_background: bool = False
    tversky_alpha: float = 0.3
    tversky_beta: float = 0.7
    focal_gamma: float = 2.0

    @classmethod
    def from_dict(cls, raw: dict | None) -> "LossConfig":
        raw = dict(raw or {})
        weights = raw.get("class_weights")
        if weights is not None:
            raw["class_weights"] = tuple(float(w) for w in weights)
        return cls(**raw)

@dataclass
class TrainConfig:
    epochs: int = 200
    amp: bool = True
    grad_accum_steps: int = 1
    grad_clip_norm: float = 0.0
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 10
    scheduler: str = "cosine"
    eta_min: float = 1e-6
    log_every_steps: int = 20
    val_interval: int = 1
    max_steps: int = 0
    save_every_epochs: int = 0
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TrainConfig":
        raw = dict(raw or {})
        raw["early_stopping"] = EarlyStoppingConfig.from_dict(raw.get("early_stopping"))
        raw["loss"] = LossConfig.from_dict(raw.get("loss"))
        return cls(**raw)

@dataclass
class EvalConfig:
    sw_batch_size: int = 4
    overlap: float = 0.5
    mode: str = "gaussian"
    include_background: bool = False
    hd95_percentile: float = 95.0
    metrics: Sequence[str] = ("dice", "iou", "precision", "recall", "hd95")
    selection_metric: str = "tumor_dice"
    max_cases: int = 0

    @classmethod
    def from_dict(cls, raw: dict | None) -> "EvalConfig":
        raw = dict(raw or {})
        if "metrics" in raw:
            raw["metrics"] = tuple(raw["metrics"])
        return cls(**raw)

@dataclass
class ModelConfig:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "ModelConfig":
        raw = dict(raw or {})
        if "name" not in raw:
            raise ValueError("model.name is required in every configuration.")
        name = raw.pop("name")
        return cls(name=str(name), kwargs=raw)

@dataclass
class Config:
    run: RunConfig
    data: DataConfig
    train: TrainConfig
    eval: EvalConfig
    model: ModelConfig
    source: str | None = None

    @classmethod
    def from_dict(cls, raw: dict, source: str | None = None) -> "Config":
        for key in ("data", "model"):
            if key not in raw:
                raise ValueError(f"Missing required configuration section: {key!r}")
        return cls(
            run=RunConfig.from_dict(raw.get("run")),
            data=DataConfig.from_dict(raw["data"]),
            train=TrainConfig.from_dict(raw.get("train")),
            eval=EvalConfig.from_dict(raw.get("eval")),
            model=ModelConfig.from_dict(raw["model"]),
            source=source,
        )

    def to_dict(self) -> dict:
        from dataclasses import asdict

        payload = asdict(self)
        payload.pop("source", None)
        return payload

def load_config(path: str | Path, overrides: Sequence[str] | None = None) -> Config:
    from .utils import apply_overrides

    raw = load_yaml(path)
    raw = apply_overrides(raw, overrides or [])
    return Config.from_dict(raw, source=str(Path(path).resolve()))

def load_configs(paths: Sequence[str | Path], overrides: Sequence[str] | None = None) -> Config:
    from .utils import apply_overrides, deep_merge

    merged: dict = {}
    resolved: list[str] = []
    for path in paths:
        merged = deep_merge(merged, load_yaml(path))
        resolved.append(str(Path(path).resolve()))
    merged = apply_overrides(merged, overrides or [])
    return Config.from_dict(merged, source=";".join(resolved))

def validate_config(config: Config) -> list[str]:
    notes: list[str] = []

    data, model = config.data, config.model
    roi = as_tuple3(data.roi_size, "data.roi_size")
    image_size = model.kwargs.get("img_size")

    if data.batch_size < 1 or data.num_samples < 1:
        raise ValueError("data.batch_size and data.num_samples must be >= 1.")
    if not 0.0 < data.valid_ratio < 1.0:
        raise ValueError(f"data.valid_ratio must be in (0, 1), got {data.valid_ratio}.")
    if data.num_classes != int(model.kwargs.get("out_channels", data.num_classes)):
        raise ValueError(
            "model.out_channels must equal the number of entries in data.labels "
            f"({data.num_classes}).")
    if config.train.grad_accum_steps < 1:
        raise ValueError("train.grad_accum_steps must be >= 1.")
    if config.train.scheduler not in {"cosine", "none"}:
        raise ValueError(f"Unsupported train.scheduler: {config.train.scheduler!r}.")
    if config.train.optimizer not in {"adamw", "adam", "sgd"}:
        raise ValueError(f"Unsupported train.optimizer: {config.train.optimizer!r}.")

    if model.name == "unetr":
        patch = int(model.kwargs.get("patch_size", 16))
        if image_size is not None and tuple(image_size) != roi:
            raise ValueError(
                f"unetr: model.img_size {tuple(image_size)} must equal data.roi_size {roi}; "
                "the ViT position embedding and the token grid depend on it.")
        if any(not divisible_by(value, patch) for value in roi):
            raise ValueError(
                f"unetr: data.roi_size {roi} must be divisible by patch_size={patch}.")
        tokens = 1
        for value in roi:
            tokens *= value // patch
        notes.append(f"unetr: token grid {tokens} tokens per sample.")

    elif model.name == "swin_unetr":
        if image_size is not None and tuple(image_size) != roi:
            notes.append(
                f"swin_unetr: model.img_size {tuple(image_size)} differs from data.roi_size {roi}; "
                "the model is fully convolutional, only the patch embedding stride matters.")
        levels = len(model.kwargs.get("depths", [2, 2, 2, 2])) - 1
        stride = int(model.kwargs.get("patch_size", 2)) * 2**levels
        if levels >= 0 and any(not divisible_by(value, stride) for value in roi):
            raise ValueError(
                f"swin_unetr: data.roi_size {roi} must be divisible by {stride} "
                f"(patch_size * 2**{levels}).")
        window = int(model.kwargs.get("window_size", 7))
        notes.append(f"swin_unetr: window {window}^3 with pad-and-crop, downsample stride {stride}.")

    elif model.name == "segformer3d":
        stride = 1
        for step in model.kwargs.get("patch_strides", [4, 2, 2, 2]):
            stride *= int(step)
        if any(value < 2 * stride for value in roi):
            raise ValueError(
                f"segformer3d: data.roi_size {roi} is too small; this encoder has stride "
                f"{stride} and needs at least {2 * stride} voxels per axis.")
        notes.append(f"segformer3d: encoder output stride {stride}, minimum roi {2 * stride}.")

    else:
        raise ValueError(f"Unknown model name: {model.name!r}")

    if model.kwargs.get("in_channels", 1) != 1:
        notes.append("in_channels != 1: make sure the input pipeline provides that many channels.")

    if config.train.loss.name not in {"dice_ce", "dicece", "dice", "tversky", "focal"}:
        raise ValueError(f"Unknown loss: {config.train.loss.name!r}")

    return notes
