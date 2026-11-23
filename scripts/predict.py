import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from src.ptseg.config import load_configs
from src.ptseg.inference import flip_tta_inference, sliding_window_inference
from src.ptseg.models import build_model
from src.ptseg.postprocess import keep_largest_components
from src.ptseg.utils import apply_intensity_window, resolve_device, set_determinism


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment a single CT volume and save the mask.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by scripts/train.py.")
    parser.add_argument("--config", "-c", nargs="*", default=None,
                        help="Optional config; defaults to <run-dir>/artifacts/config.yml.")
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint file inside run-dir.")
    parser.add_argument("--input", required=True, help="Input CT volume (.nii.gz).")
    parser.add_argument("--output", default=None, help="Output mask path.")
    parser.add_argument("--device", default=None, help="Override run.device (auto|cpu|cuda).")
    parser.add_argument("--postprocess", action="store_true",
                        help="Keep only the largest connected component per foreground class.")
    parser.add_argument("--tta", action="store_true",
                        help="Average the logits over the eight flip combinations (slower).")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    run_dir = Path(arguments.run_dir).resolve()
    overrides = [f"run.device={arguments.device}"] if arguments.device else []
    config = load_configs(arguments.config or [str(run_dir / "artifacts" / "config.yml")], overrides)

    set_determinism(config.run.seed, config.run.deterministic)
    device = resolve_device(config.run.device)

    checkpoint_path = run_dir / "checkpoints" / arguments.checkpoint
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(config.model.name, **config.model.kwargs).to(device)
    model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
    model.eval()

    image = nib.as_closest_canonical(nib.load(arguments.input))
    array = np.asanyarray(image.dataobj, dtype=np.float32)
    original_shape = array.shape
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])

    windowed = apply_intensity_window(array, float(config.data.hu_window[0]),
                                      float(config.data.hu_window[1]),
                                      tuple(config.data.intensity_range))
    pixdim = tuple(float(v) for v in config.data.pixdim)
    zoom = [spacing[axis] / pixdim[axis] for axis in range(3)]
    if any(abs(factor - 1.0) > 1e-3 for factor in zoom):
        windowed = ndimage.zoom(windowed, zoom, order=1)

    inputs = torch.from_numpy(np.ascontiguousarray(windowed))[None, None].float().to(device)
    if arguments.tta:
        probabilities = flip_tta_inference(inputs, config.data.roi_size, model,
                                           sw_batch_size=int(config.eval.sw_batch_size),
                                           overlap=float(config.eval.overlap),
                                           mode=config.eval.mode)
        prediction = probabilities.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    else:
        logits = sliding_window_inference(inputs, config.data.roi_size, model,
                                          sw_batch_size=int(config.eval.sw_batch_size),
                                          overlap=float(config.eval.overlap),
                                          mode=config.eval.mode)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

    if arguments.postprocess:
        foreground = [index for index in config.data.labels.values() if index != 0]
        prediction = keep_largest_components(prediction, tuple(foreground))

    inverse_zoom = [original_shape[axis] / prediction.shape[axis] for axis in range(3)]
    restored = ndimage.zoom(prediction, inverse_zoom, order=0)[:original_shape[0],
                                                               :original_shape[1],
                                                               :original_shape[2]]
    restored = restored.astype(np.uint8)

    output_path = Path(arguments.output) if arguments.output else Path(
        arguments.input).with_name(Path(arguments.input).name.replace(".nii.gz", "") + "_pred.nii.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(restored, image.affine), str(output_path))
    print(f"Saved mask: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
