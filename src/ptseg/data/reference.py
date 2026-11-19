from pathlib import Path

import numpy as np
import torch

SEARCH_ROOTS = (
    "datasets/prepared/imagesTr",
    "datasets/prepared/labelsTr",
    "datasets/tcia_pda/extracted/Pancreas_segmentations",
    "datasets/tcia_pda/extracted/Tumor_segmentations",
)

def find_reference_volume(root: Path | None = None) -> Path:
    root = Path(root or Path(__file__).resolve().parents[3])
    for candidate in SEARCH_ROOTS:
        folder = root / candidate
        if folder.is_dir():
            files = sorted(folder.glob("*.nii.gz"))
            if files:
                return files[0]
    raise FileNotFoundError(
        "No reference volume found under datasets/. Run scripts/download_data.py and "
        "scripts.prepare_data.py first.")

def load_array(path: Path) -> np.ndarray:
    import nibabel as nib

    data = np.asanyarray(nib.load(str(path)).dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"{path}: expected a 3D volume, got shape {data.shape}")
    return data

def crop_around(array: np.ndarray, size: tuple[int, int, int]) -> np.ndarray:
    pad_width = [(0, max(0, target - array.shape[axis])) for axis, target in enumerate(size)]
    padded = np.pad(array, pad_width)

    if array.any():
        coordinates = np.argwhere(array > 0)
        center = [int(np.median(coordinates[:, axis])) for axis in range(3)]
    else:
        center = [padded.shape[axis] // 2 for axis in range(3)]

    slices = []
    for axis, target in enumerate(size):
        start = min(max(0, center[axis] - target // 2), padded.shape[axis] - target)
        slices.append(slice(start, start + target))
    return padded[tuple(slices)]

def load_patch(size: tuple[int, int, int] = (32, 32, 32), channels: int = 1,
               path: Path | None = None) -> torch.Tensor:
    array = load_array(Path(path) if path else find_reference_volume()).astype(np.float32)
    low, high = float(array.min()), float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    patch = crop_around(array, tuple(int(v) for v in size))
    tensor = torch.from_numpy(np.ascontiguousarray(patch))[None, None]
    if channels > 1:
        tensor = tensor.repeat(1, channels, 1, 1, 1)
    return tensor

def load_mask(size: tuple[int, int, int] = (64, 64, 64), path: Path | None = None) -> np.ndarray:
    array = load_array(Path(path) if path else find_reference_volume())
    patch = crop_around(array, tuple(int(v) for v in size))
    return (patch > 0).astype(np.uint8)
