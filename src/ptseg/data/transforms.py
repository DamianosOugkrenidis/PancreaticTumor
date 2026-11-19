from typing import Any

from monai.transforms import (Compose, CropForegroundd, EnsureChannelFirstd, LoadImaged,
                              Orientationd, RandFlipd, RandGaussianNoised, RandGaussianSmoothd,
                              RandCropByPosNegLabeld, RandRotate90d, RandScaleIntensityd,
                              RandShiftIntensityd, ScaleIntensityRanged, Spacingd, ToTensord)

from ..config import DataConfig

IMAGE_KEY = "image"
LABEL_KEY = "label"

def deterministic_transforms(config: DataConfig) -> list[Any]:
    return [
        LoadImaged(keys=[IMAGE_KEY, LABEL_KEY], image_only=False),
        EnsureChannelFirstd(keys=[IMAGE_KEY, LABEL_KEY]),
        Orientationd(keys=[IMAGE_KEY, LABEL_KEY], axcodes="RAS",
                     labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        Spacingd(keys=[IMAGE_KEY, LABEL_KEY], pixdim=tuple(config.pixdim),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=[IMAGE_KEY], a_min=float(config.hu_window[0]),
                             a_max=float(config.hu_window[1]),
                             b_min=float(config.intensity_range[0]),
                             b_max=float(config.intensity_range[1]), clip=True),
        CropForegroundd(keys=[IMAGE_KEY, LABEL_KEY], source_key=IMAGE_KEY,
                        allow_smaller=True),
        ToTensord(keys=[IMAGE_KEY, LABEL_KEY]),
    ]

def augmentation_transforms(config: DataConfig) -> list[Any]:
    return [
        RandFlipd(keys=[IMAGE_KEY, LABEL_KEY], prob=0.5, spatial_axis=0),
        RandFlipd(keys=[IMAGE_KEY, LABEL_KEY], prob=0.5, spatial_axis=1),
        RandFlipd(keys=[IMAGE_KEY, LABEL_KEY], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=[IMAGE_KEY, LABEL_KEY], prob=0.3, max_k=3),
        RandShiftIntensityd(keys=[IMAGE_KEY], offsets=0.10, prob=0.5),
        RandScaleIntensityd(keys=[IMAGE_KEY], factors=0.10, prob=0.3),
        RandGaussianNoised(keys=[IMAGE_KEY], prob=0.2, mean=0.0, std=0.05, sample_std=True),
        RandGaussianSmoothd(keys=[IMAGE_KEY], prob=0.2,
                            sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
    ]

def train_transforms(config: DataConfig) -> Compose:
    transforms = deterministic_transforms(config)
    transforms.append(
        RandCropByPosNegLabeld(keys=[IMAGE_KEY, LABEL_KEY], label_key=LABEL_KEY,
                               spatial_size=tuple(config.roi_size),
                               pos=1.0, neg=1.0,
                               num_samples=int(config.num_samples),
                               image_key=IMAGE_KEY, image_threshold=0.0))
    if config.augment:
        transforms.extend(augmentation_transforms(config))
    return Compose(transforms)

def val_transforms(config: DataConfig) -> Compose:
    return Compose(deterministic_transforms(config))
