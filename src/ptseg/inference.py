import itertools
from typing import Sequence

import torch
import torch.nn.functional as F

from .utils import as_tuple3

def _window_starts(size: int, roi: int, overlap: float) -> list[int]:
    if roi >= size:
        return [0]
    step = max(1, int(round(roi * (1.0 - overlap))))
    starts = list(range(0, size - roi + 1, step))
    if starts[-1] != size - roi:
        starts.append(size - roi)
    return starts

def gaussian_window(roi_size: Sequence[int], sigma_scale: float = 0.125,
                    min_fraction: float = 0.25, device: torch.device | str = "cpu",
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if not 0.0 <= min_fraction < 1.0:
        raise ValueError(f"min_fraction must be in [0, 1), got {min_fraction}.")
    roi = as_tuple3(roi_size, "roi_size")
    axes = []
    for size in roi:
        sigma = size * sigma_scale
        coordinates = torch.arange(size, dtype=torch.float32)
        center = (size - 1) / 2.0
        axes.append(torch.exp(-((coordinates - center) ** 2) / (2 * sigma ** 2)))
    window = (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :])
    window = window / window.max()
    if min_fraction > 0:
        window = window.clamp_min(min_fraction)
    return window.to(device=device, dtype=dtype)[None, None]

@torch.no_grad()
def sliding_window_inference(inputs: torch.Tensor, roi_size: Sequence[int],
                             model: torch.nn.Module, sw_batch_size: int = 4,
                             overlap: float = 0.5, mode: str = "gaussian",
                             progress: bool = False) -> torch.Tensor:
    if inputs.dim() != 5:
        raise ValueError(f"Expected (B, C, D, H, W) input, got {tuple(inputs.shape)}.")
    if inputs.shape[0] != 1:
        raise ValueError(
            f"sliding_window_inference merges the windows of a single volume; got batch "
            f"size {inputs.shape[0]}. Evaluate volumes one by one.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}.")

    roi = as_tuple3(roi_size, "roi_size")
    model.eval()
    device = next(model.parameters()).device

    original_size = tuple(int(v) for v in inputs.shape[-3:])
    padding = [max(0, roi[axis] - original_size[axis]) for axis in range(3)]
    if any(padding):
        inputs = F.pad(inputs, (0, padding[2], 0, padding[1], 0, padding[0]))

    spatial = tuple(int(v) for v in inputs.shape[-3:])
    starts = [_window_starts(spatial[axis], roi[axis], overlap) for axis in range(3)]
    positions = list(itertools.product(*starts))

    if mode == "gaussian":
        weight = gaussian_window(roi, device=device, dtype=inputs.dtype)
    elif mode == "constant":
        weight = torch.ones((1, 1, *roi), device=device, dtype=inputs.dtype)
    else:
        raise ValueError(f"Unknown merging mode: {mode!r} (expected gaussian|constant).")

    output: torch.Tensor | None = None
    normaliser: torch.Tensor | None = None

    iterator = range(0, len(positions), sw_batch_size)
    for start in iterator:
        batch_positions = positions[start:start + sw_batch_size]
        crops = torch.stack([
            inputs[0, :, d:d + roi[0], h:h + roi[1], w:w + roi[2]] for d, h, w in batch_positions])
        crops = crops.to(device, non_blocking=True)

        logits = model(crops)
        if output is None:
            num_classes = logits.shape[1]
            output = torch.zeros((1, num_classes, *spatial), device=device, dtype=logits.dtype)
            normaliser = torch.zeros((1, 1, *spatial), device=device, dtype=logits.dtype)

        for index, (d, h, w) in enumerate(batch_positions):
            output[..., d:d + roi[0], h:h + roi[1], w:w + roi[2]] += logits[index:index + 1] * weight
            normaliser[..., d:d + roi[0], h:h + roi[1], w:w + roi[2]] += weight

    if output is None or normaliser is None:
        raise RuntimeError("Sliding window produced no windows; check roi_size and input size.")

    output = output / normaliser.clamp_min(1e-8)
    return output[..., :original_size[0], :original_size[1], :original_size[2]]

def stats_positions(roi_size: Sequence[int], input_size: Sequence[int], overlap: float) -> int:
    roi = as_tuple3(roi_size, "roi_size")
    size = as_tuple3(input_size, "input_size")
    total = 1
    for axis in range(3):
        total *= len(_window_starts(max(size[axis], roi[axis]), roi[axis], overlap))
    return total
