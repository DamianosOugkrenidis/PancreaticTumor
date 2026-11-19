import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.ptseg.config import load_configs, validate_config
from src.ptseg.data.reference import load_patch
from src.ptseg.models import available_models, build_model, describe_model
from src.ptseg.utils import count_parameters, dump_json, resolve_device, set_determinism

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the model implementations.")
    parser.add_argument("--config-dir", default="configs", help="Directory holding model YAML files.")
    parser.add_argument("--models", nargs="*", default=available_models(),
                        help=f"Subset of {available_models()}.")
    parser.add_argument("--roi", type=int, nargs=3, default=[64, 64, 64],
                        metavar=("D", "H", "W"),
                        help="Input size for the forward-pass check (64^3 runs every shipped "
                             "configuration).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="auto|cpu|cuda.")
    parser.add_argument("--input", default=None,
                        help="Reference NIfTI used to build the input patch (default: first file "
                             "under datasets/).")
    parser.add_argument("--output", default="outputs/verification/architecture_report.json")
    return parser.parse_args()

def check_shape_guards(config_dir: Path, model_names: list[str], roi: tuple[int, int, int]
                       ) -> list[dict]:
    results = []
    bad_size = tuple(value - 1 for value in roi)
    for name in model_names:
        config_path = config_dir / f"{name}.yml"
        if not config_path.is_file():
            continue
        override = f"model.img_size=[{bad_size[0]},{bad_size[1]},{bad_size[2]}]"
        try:
            config = load_configs([config_path], [override])
            config.data.roi_size = bad_size
            validate_config(config)
            rejected = False
            message = "accepted an indivisible input size"
        except ValueError as error:
            rejected = True
            message = str(error)
        results.append({"model": name, "invalid_roi": list(bad_size), "rejected": rejected,
                        "message": message})
    return results

def main() -> int:
    arguments = parse_arguments()
    roi = tuple(int(v) for v in arguments.roi)
    device = resolve_device(arguments.device)
    set_determinism(0)

    config_dir = Path(arguments.config_dir)
    report: dict = {"roi": list(roi), "device": str(device), "models": [], "guard_checks": []}
    failures: list[str] = []

    print(f"{'model':<13}{'params':>12}{'out shape':>22}{'forward (ms)':>14}  status")
    for name in arguments.models:
        config_path = config_dir / f"{name}.yml"
        if not config_path.is_file():
            failures.append(f"{name}: missing config {config_path}")
            continue

        config = load_configs([config_path], [f"model.img_size=[{roi[0]},{roi[1]},{roi[2]}]"])
        try:
            model = build_model(config.model.name, **config.model.kwargs).to(device)
            model.eval()
            inputs = load_patch(roi, channels=int(config.model.kwargs.get("in_channels", 1)),
                                path=Path(arguments.input) if arguments.input else None)
            inputs = inputs.to(device).repeat(arguments.batch_size, 1, 1, 1, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            import time
            start = time.perf_counter()
            with torch.no_grad():
                outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            expected = (arguments.batch_size, config.model.kwargs.get("out_channels", 3), *roi)
            shape_ok = tuple(outputs.shape) == expected
            finite = bool(torch.isfinite(outputs).all().item())
            total, trainable = count_parameters(model)
            status = "ok" if (shape_ok and finite) else "FAILED"
            if not shape_ok:
                failures.append(f"{name}: output {tuple(outputs.shape)} != expected {expected}")
            if not finite:
                failures.append(f"{name}: output contains non-finite values")

            report["models"].append({
                "model": name,
                "parameters": total,
                "trainable": trainable,
                "output_shape": list(outputs.shape),
                "forward_ms": elapsed_ms,
                "finite": finite,
                "architecture": describe_model(model),
            })
            print(f"{name:<13}{total:>12,}{str(tuple(outputs.shape)):>22}{elapsed_ms:>14.1f}  {status}")
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
            print(f"{name:<13}{'-':>12}{'-':>22}{'-':>14}  FAILED ({error})")

    report["guard_checks"] = check_shape_guards(config_dir, arguments.models, roi)
    for guard in report["guard_checks"]:
        if not guard["rejected"]:
            failures.append(f"{guard['model']}: {guard['message']}")
        print(f"guard {guard['model']:<13} roi {guard['invalid_roi']} -> "
              f"{'rejected' if guard['rejected'] else 'ACCEPTED (bug)'}")

    dump_json(arguments.output, report)
    print(f"\nReport written to {arguments.output}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All architecture checks passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
