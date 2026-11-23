import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ptseg.data.reference import load_mask, load_patch
from src.ptseg.inference import gaussian_window, sliding_window_inference
from src.ptseg.losses import DiceCELoss, soft_dice_loss
from src.ptseg.metrics import evaluate_case, hausdorff_distance_95
from src.ptseg.models import build_model
from src.ptseg.models.blocks import (PatchMerging3D, SwinBlock3D, WindowAttention3D,
                                     window_partition3d, window_reverse3d)
from src.ptseg.utils import pad_to_multiple

SMALL_ROI = (32, 32, 32)
SPACING = (1.5, 1.5, 1.5)

def small_model_kwargs() -> dict[str, dict]:
    return {
        "unetr": dict(in_channels=1, out_channels=3, img_size=SMALL_ROI, patch_size=16,
                      feature_size=8, hidden_size=64, mlp_dim=128, num_heads=4, num_layers=2,
                      skip_indices=(1, 1, 1, 2), decoder_channels=(32, 16, 8, 8)),
        "swin_unetr": dict(in_channels=1, out_channels=3, img_size=SMALL_ROI, patch_size=2,
                           feature_size=8, depths=(1, 1, 2, 2), num_heads=(2, 2, 4, 4),
                           window_size=4, window_shift=2),
        "segformer3d": dict(in_channels=1, out_channels=3, img_size=SMALL_ROI,
                            embed_dims=(16, 32, 64), depths=(1, 1, 1),
                            num_heads=(1, 2, 4), sr_ratios=(8, 4, 2),
                            patch_sizes=(7, 3, 3), patch_strides=(4, 2, 2),
                            patch_paddings=(3, 1, 1)),
    }

def one_hot_logits(labels: torch.Tensor, num_classes: int = 3, scale: float = 16.0,
                   requires_grad: bool = False) -> torch.Tensor:
    squeezed = labels[:, 0].long()
    one_hot = F.one_hot(squeezed, num_classes)
    logits = one_hot.permute(0, 4, 1, 2, 3).float() * scale - scale / 2.0
    return logits.requires_grad_(requires_grad)

def test_model_forward_shapes() -> None:
    inputs = load_patch(SMALL_ROI)
    for name, kwargs in small_model_kwargs().items():
        model = build_model(name, **kwargs).eval()
        with torch.no_grad():
            outputs = model(inputs)
        assert outputs.shape == (1, 3, *SMALL_ROI), f"{name}: {tuple(outputs.shape)}"
        assert torch.isfinite(outputs).all(), f"{name}: non-finite output"

def test_segformer_default_config_runs() -> None:
    model = build_model("segformer3d", in_channels=1, out_channels=3, img_size=(64, 64, 64))
    with torch.no_grad():
        outputs = model(load_patch((64, 64, 64)))
    assert outputs.shape == (1, 3, 64, 64, 64), tuple(outputs.shape)

def test_unetr_token_grid_and_guard() -> None:
    model = build_model("unetr", **small_model_kwargs()["unetr"])
    assert model.describe()["tokens"] == 8, model.describe()
    try:
        build_model("unetr", img_size=(30, 32, 32), patch_size=16, hidden_size=64, mlp_dim=128,
                    num_heads=4, num_layers=1, feature_size=8, decoder_channels=(16, 8, 8, 8),
                    skip_indices=(1, 1, 1, 1), out_channels=3)
    except ValueError:
        return
    raise AssertionError("UNETR accepted a non-divisible image size")

def test_swin_padding_keeps_odd_sizes_working() -> None:
    model = build_model("swin_unetr", img_size=(48, 48, 48), patch_size=2, feature_size=8,
                        depths=(1, 1, 2, 2), num_heads=(2, 2, 4, 4), window_size=7,
                        window_shift=3, out_channels=3).eval()
    with torch.no_grad():
        outputs = model(load_patch((48, 48, 48)))
    assert outputs.shape == (1, 3, 48, 48, 48), tuple(outputs.shape)

def test_swin_rejects_indivisible_sizes() -> None:
    try:
        build_model("swin_unetr", img_size=(20, 20, 20), patch_size=2, feature_size=8,
                    depths=(1, 1, 2, 2), num_heads=(2, 2, 4, 4), out_channels=3)
    except ValueError:
        return
    raise AssertionError("Swin UNETR accepted an input size its stride cannot divide")

def reference_window_attention(module: WindowAttention3D, windows: torch.Tensor) -> torch.Tensor:
    num_windows, tokens, channels = windows.shape
    heads, head_dim = module.num_heads, module.head_dim

    qkv = F.linear(windows, module.qkv.weight, module.qkv.bias)
    qkv = qkv.reshape(num_windows, tokens, 3, heads, head_dim)
    q = qkv[:, :, 0].permute(0, 2, 1, 3)
    k = qkv[:, :, 1].permute(0, 2, 1, 3)
    v = qkv[:, :, 2].permute(0, 2, 1, 3)

    bias = module.relative_position_bias_table[module.relative_position_index.flatten()]
    bias = bias.reshape(tokens, tokens, heads).permute(2, 0, 1)

    outputs = []
    for window in range(num_windows):
        heads_out = []
        for head in range(heads):
            scores = torch.einsum("nd,md->nm", q[window, head], k[window, head]) * module.scale
            weights = torch.softmax(scores + bias[head], dim=-1)
            heads_out.append(torch.einsum("nm,md->nd", weights, v[window, head]))
        joined = torch.cat(heads_out, dim=-1)
        outputs.append(F.linear(joined, module.proj.weight, module.proj.bias))
    return torch.stack(outputs)

def test_window_attention_matches_reference() -> None:
    module = WindowAttention3D(dim=16, window_size=(4, 4, 4), num_heads=2).eval()
    patch = load_patch((4, 4, 4)).reshape(-1, 1)
    windows = torch.cat([patch * (index + 1) for index in range(16)], dim=1).repeat(3, 1, 1)
    with torch.no_grad():
        fast = module(windows)
        reference = reference_window_attention(module, windows)
    assert torch.allclose(fast, reference, atol=1e-5), \
        f"max diff {(fast - reference).abs().max().item()}"

def test_window_partition_roundtrip() -> None:
    x = torch.arange(2 * 4 * 8 * 8 * 3, dtype=torch.float32).reshape(2, 4, 8, 8, 3)
    windows = window_partition3d(x, (2, 4, 4))
    restored = window_reverse3d(windows, (2, 4, 4), (4, 8, 8))
    assert restored.shape == x.shape
    assert torch.allclose(restored, x)

def test_swin_block_shift_changes_output_and_keeps_shape() -> None:
    block = SwinBlock3D(dim=8, num_heads=2, window_size=4, shift_size=0).eval()
    shifted = SwinBlock3D(dim=8, num_heads=2, window_size=4, shift_size=2).eval()
    shifted.load_state_dict(block.state_dict())

    base = load_patch((8, 8, 8))[0, 0]
    x = torch.stack([torch.roll(base, shifts=index, dims=0) for index in range(8)],
                    dim=-1).unsqueeze(0)
    with torch.no_grad():
        base = block(x)
        moved = shifted(x)
    assert base.shape == x.shape and moved.shape == x.shape
    assert not torch.allclose(base, moved), "shifting had no effect"

def test_patch_merging_halves_resolution() -> None:
    merging = PatchMerging3D(dim=8)
    out = merging(load_patch((4, 6, 6), channels=8).permute(0, 2, 3, 4, 1))
    assert out.shape == (1, 2, 3, 3, 16), tuple(out.shape)

def test_pad_to_multiple() -> None:
    padded, original = pad_to_multiple(load_patch((5, 9, 3), channels=2), 4)
    assert original == (5, 9, 3)
    assert padded.shape[-3:] == (8, 12, 4), tuple(padded.shape)

def test_dice_loss_limits() -> None:
    labels = torch.from_numpy(load_mask((16, 16, 16)))[None, None].long()
    assert labels.max() > 0, "reference mask patch is empty"

    perfect = one_hot_logits(labels)
    loss_perfect = float(soft_dice_loss(perfect, labels, num_classes=3))
    assert loss_perfect < 0.05, loss_perfect

    inverted = perfect.clone()
    inverted[:, 1] = -perfect[:, 1]
    loss_bad = float(soft_dice_loss(inverted, labels, num_classes=3))
    assert loss_bad > 0.4, loss_bad

def test_dice_loss_ignores_absent_classes() -> None:
    labels = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    logits = one_hot_logits(labels)
    loss = float(soft_dice_loss(logits, labels, num_classes=3))
    assert loss == 0.0, loss

def test_dice_ce_is_finite_and_differentiable() -> None:
    criterion = DiceCELoss(num_classes=3, class_weights=(0.1, 1.0, 4.0))
    labels = torch.from_numpy(load_mask((16, 16, 16)))[None, None].long()
    logits = one_hot_logits(labels, requires_grad=True)
    loss = criterion(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

def test_metrics_perfect_and_disjoint() -> None:
    reference = load_mask((64, 64, 64))
    assert reference.sum() > 0, "reference mask patch is empty"

    perfect = evaluate_case(reference.copy(), reference, ("background", "pancreas"),
                            spacing=SPACING, include_background=False)
    assert set(perfect["dice"]) == {"pancreas"}, set(perfect["dice"])
    assert abs(perfect["dice"]["pancreas"] - 1.0) < 1e-9
    assert perfect["hd95"]["pancreas"] == 0.0

    shifted = np.roll(reference, 2, axis=0)
    disjoint = evaluate_case(shifted, reference, ("background", "pancreas"), spacing=SPACING,
                             include_background=False)
    assert 0.0 < disjoint["dice"]["pancreas"] < 1.0, disjoint["dice"]
    assert disjoint["hd95"]["pancreas"] > 0.0, disjoint["hd95"]

def test_hd95_is_reported_in_millimetres() -> None:
    reference = load_mask((64, 64, 64))
    shifted = np.roll(reference, 2, axis=0)
    at_one_mm = hausdorff_distance_95(shifted > 0, reference > 0, spacing=(1.0, 1.0, 1.0))
    at_spacing = hausdorff_distance_95(shifted > 0, reference > 0, spacing=SPACING)
    assert at_one_mm > 0.0
    assert abs(at_spacing - at_one_mm * 1.5) < 1e-6, (at_one_mm, at_spacing)

def test_hd95_undefined_for_empty_prediction() -> None:
    reference = load_mask((32, 32, 32))
    empty = np.zeros_like(reference)
    assert np.isnan(hausdorff_distance_95(empty, reference))
    assert hausdorff_distance_95(empty, empty) == 0.0

class ConstantModel(nn.Module):
    def __init__(self, out_channels: int = 3, value: float = 0.5) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.full((1, out_channels, 1, 1, 1), value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(x.shape[0], -1, *x.shape[-3:]).contiguous()

class ShapeGuardModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.out_channels = 3
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise AssertionError(f"window batch must be (B, C, D, H, W), got {tuple(x.shape)}")
        return torch.ones(x.shape[0], self.out_channels, *x.shape[-3:], device=x.device) * self.scale

def test_sliding_window_normalises_weights() -> None:
    model = ConstantModel(value=0.25)
    inputs = load_patch((32, 32, 32))
    output = sliding_window_inference(inputs, (16, 16, 16), model, sw_batch_size=4, overlap=0.5)
    assert output.shape == (1, 3, 32, 32, 32), tuple(output.shape)
    assert torch.allclose(output, torch.full_like(output, 0.25), atol=1e-5), \
        f"min={float(output.min())}, max={float(output.max())}"

def test_sliding_window_handles_small_inputs() -> None:
    model = ConstantModel(value=1.0)
    inputs = load_patch((8, 12, 6))
    output = sliding_window_inference(inputs, (16, 16, 16), model, overlap=0.0)
    assert output.shape == (1, 3, 8, 12, 6), tuple(output.shape)
    assert torch.allclose(output, torch.ones_like(output), atol=1e-5)

def test_sliding_window_window_batch_is_five_dimensional() -> None:
    model = ShapeGuardModel()
    output = sliding_window_inference(load_patch((24, 24, 24)), (16, 16, 16), model,
                                      sw_batch_size=4, overlap=0.5)
    assert output.shape == (1, 3, 24, 24, 24), tuple(output.shape)

def test_gaussian_window_range() -> None:
    window = gaussian_window((16, 16, 16))
    assert window.shape == (1, 1, 16, 16, 16)
    assert float(window.max()) == 1.0
    assert float(window.min()) > 0.0

def test_flip_tta_returns_probabilities() -> None:
    from src.ptseg.inference import flip_tta_inference

    model = ConstantModel(value=0.5)
    inputs = load_patch((16, 16, 16))
    probabilities = flip_tta_inference(inputs, (16, 16, 16), model, overlap=0.0)
    assert probabilities.shape == (1, 3, 16, 16, 16), tuple(probabilities.shape)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(1, 16, 16, 16), atol=1e-5)


def main() -> int:
    torch.manual_seed(0)
    tests = [(name, value) for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    failures = []
    for name, test in tests:
        try:
            test()
            print(f"PASS  {name}")
        except Exception as error:
            failures.append((name, error))
            print(f"FAIL  {name}: {type(error).__name__}: {error}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} checks passed")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
