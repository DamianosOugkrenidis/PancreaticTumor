import json
import random
from pathlib import Path
from typing import Any

import torch
from monai.data import CacheDataset, DataLoader, Dataset, list_data_collate

from ..config import DataConfig
from ..utils import seed_worker
from .transforms import IMAGE_KEY, LABEL_KEY, train_transforms, val_transforms

def discover_files(root: str | Path, dataset_json: str | Path | None = None
                   ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root not found: {root}. Run scripts/download_data.py and "
            "scripts/prepare_data.py, then point data.root at datasets/prepared.")

    if dataset_json:
        json_path = Path(dataset_json)
        if not json_path.is_absolute():
            json_path = root / json_path
        if not json_path.is_file():
            raise FileNotFoundError(f"dataset.json not found: {json_path}")
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("training") or []
        files = []
        for entry in entries:
            image = entry.get("image")
            label = entry.get("label")
            if not image:
                continue
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = root / image_path
            if label:
                label_path = Path(label)
                label_path = label_path if label_path.is_absolute() else root / label_path
            else:
                label_path = root / "labelsTr" / Path(image).name.replace("_0000", "")
            files.append({"image": str(image_path), "label": str(label_path)})
        if not files:
            raise ValueError(f"No training entries found in {json_path}")
        return files, []

    images = sorted((root / "imagesTr").glob("*.nii.gz"))
    labels_dir = root / "labelsTr"
    files = []
    for image in images:
        label = labels_dir / image.name
        if not label.is_file():
            candidates = sorted(labels_dir.glob(image.name.split("_0000")[0] + "*"))
            if not candidates:
                continue
            label = candidates[0]
        files.append({"image": str(image), "label": str(label)})
    if not files:
        raise FileNotFoundError(
            f"No image/label pairs found under {root} (expected imagesTr/*.nii.gz and "
            "labelsTr/*.nii.gz).")
    return files, []

def split_files(files: list[dict[str, str]], valid_ratio: float, seed: int
                ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError(f"valid_ratio must be in (0, 1), got {valid_ratio}.")
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    num_valid = max(1, int(round(len(shuffled) * valid_ratio)))
    return shuffled[num_valid:], shuffled[:num_valid]

def build_dataloaders(config: DataConfig, seed: int, logger: Any = None
                      ) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    files, split_from_json = discover_files(config.root, config.dataset_json)
    if split_from_json:
        train_files = files
        _, val_files = split_files(files, config.valid_ratio, seed)
    else:
        train_files, val_files = split_files(files, config.valid_ratio, seed)

    if not train_files or not val_files:
        raise ValueError(
            f"Split produced {len(train_files)} training and {len(val_files)} validation "
            "volumes. Adjust data.valid_ratio or provide more data.")

    if logger is not None:
        logger.info(f"Dataset root: {Path(config.root).resolve()}")
        logger.info(f"Volumes: {len(train_files)} training / {len(val_files)} validation "
                    f"(split seed {seed})")
        if len(train_files) < 10:
            logger.warning("Fewer than 10 training volumes: results will be unreliable.")

    train_dataset = Dataset(data=train_files, transform=train_transforms(config))
    if config.cache_rate > 0:
        val_dataset: Dataset = CacheDataset(data=val_files, transform=val_transforms(config),
                                            cache_rate=float(config.cache_rate),
                                            num_workers=0, progress=False)
    else:
        val_dataset = Dataset(data=val_files, transform=val_transforms(config))

    loader_kwargs = dict(num_workers=int(config.num_workers),
                         pin_memory=torch.cuda.is_available(),
                         worker_init_fn=seed_worker, collate_fn=list_data_collate)
    train_loader = DataLoader(train_dataset, batch_size=int(config.batch_size),
                             shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)

    metadata = {
        "num_train_volumes": len(train_files),
        "num_valid_volumes": len(val_files),
        "roi_size": list(config.roi_size),
        "batch_size": int(config.batch_size),
        "num_samples": int(config.num_samples),
        "cache_rate": float(config.cache_rate),
        "train_files": [Path(entry[IMAGE_KEY]).name for entry in train_files],
        "valid_files": [Path(entry[IMAGE_KEY]).name for entry in val_files],
    }
    return train_loader, val_loader, metadata
