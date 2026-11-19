import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.ptseg.config import load_configs, validate_config
from src.ptseg.inference import sliding_window_inference, stats_positions
from src.ptseg.models import available_models, build_model, describe_model
from src.ptseg.utils import count_parameters, dump_json, resolve_device, set_determinism


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure the computational cost of every model (parameters, FLOPs, "
                    "forward/backward time, peak memory, sliding-window inference).")
    parser.add_argument("--config-dir", default="configs", help="Directory holding model YAML files.")
    parser.add_argument("--models", nargs="*", default=available_models(),
                        help=f"Subset of {available_models()}.")
    parser.add_argument("--roi", type=int, nargs=3, default=[64, 64, 64], metavar=("D", "H", "W"),
                        help="Patch size for the forward/backward and FLOP measurements.")
    parser.add_argument("--sw-volume", type=int, nargs=3, default=None, metavar=("D", "H", "W"),
                        help="Full-volume size for the sliding-window benchmark "
                             "(default: roi with extra 48 voxels per axis).")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions.")
    parser.add_argument("--device", default="cpu", help="auto|cpu|cuda.")
    parser.add_argument("--no-backward", action="store_true",
                        help="Skip the backward-pass timing (forward-only).")
    parser.add_argument("--no-sliding-window", action="store_true",
                        help="Skip the sliding-window inference benchmark.")
    parser.add_argument("--output", default="outputs/benchmark/benchmark.json",
                        help="Where to write the JSON report.")
    return parser.parse_args()


def make_input(roi: tuple[int, int, int], channels: int, device: torch.device) -> torch.Tensor:
    try:
        from src.ptseg.data.reference import load_patch

        return load_patch(roi, channels=channels).to(device)
    except Exception:
        return torch.randn(1, channels, *roi, device=device)


def measure_flops(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device):
    try:
        from torch.utils.flop_counter import FlopCounterMode

        counter = FlopCounterMode(display=False)
        with counter:
            model(inputs)
        return int(counter.get_total_flops())
    except Exception:
        return None


def time_forward(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device,
                 repeats: int) -> float:
    with torch.no_grad():
        model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / max(repeats, 1)


def time_backward(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device,
                  repeats: int) -> float:
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    for _ in range(repeats):
        step = torch.autograd.grad(model(inputs).float().pow(2).mean(), parameters,
                                   retain_graph=False, allow_unused=True)
        model.zero_grad(set_to_none=True)
        del step
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        step = torch.autograd.grad(model(inputs).float().pow(2).mean(), parameters,
                                   retain_graph=False, allow_unused=True)
        model.zero_grad(set_to_none=True)
        del step
    if device.type == "cuda":
        torch.cuda.synchronize()
    model.eval()
    return (time.perf_counter() - start) * 1000.0 / max(repeats, 1)


def benchmark_model(name: str, config_dir: Path, arguments: argparse.Namespace,
                    device: torch.device, sw_volume: tuple[int, int, int]) -> dict:
    config = load_configs([config_dir / f"{name}.yml"],
                          [f"model.img_size=[{arguments.roi[0]},{arguments.roi[1]},"
                           f"{arguments.roi[2]}]",
                           f"data.roi_size=[{arguments.roi[0]},{arguments.roi[1]},"
                           f"{arguments.roi[2]}]"])
    validate_config(config)
    channels = int(config.model.kwargs.get("in_channels", 1))
    model = build_model(config.model.name, **config.model.kwargs).to(device).eval()
    total, trainable = count_parameters(model)
    inputs = make_input(tuple(arguments.roi), channels, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    flops = measure_flops(model, inputs, device)
    forward_ms = time_forward(model, inputs, device, arguments.repeats)
    backward_ms = None if arguments.no_backward else \
        time_backward(model, inputs, device, arguments.repeats)
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024 ** 2) if device.type == "cuda" \
        else None

    sw_seconds = None
    sw_windows = None
    if not arguments.no_sliding_window:
        volume = torch.randn(1, channels, *sw_volume, device=device)
        model.eval()
        with torch.no_grad():
            sliding_window_inference(volume, tuple(arguments.roi), model, sw_batch_size=4,
                                     overlap=0.5, mode="gaussian")
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            sliding_window_inference(volume, tuple(arguments.roi), model, sw_batch_size=4,
                                     overlap=0.5, mode="gaussian")
            if device.type == "cuda":
                torch.cuda.synchronize()
        sw_seconds = time.perf_counter() - start
        sw_windows = stats_positions(tuple(arguments.roi), sw_volume, 0.5)

    gflops = None if flops is None else flops / 1e9
    return {
        "model": name,
        "parameters": total,
        "trainable": trainable,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "gflops": gflops,
        "peak_memory_mb": peak_mb,
        "sliding_window_seconds": sw_seconds,
        "sliding_window_windows": sw_windows,
        "architecture": describe_model(model),
    }


def main() -> int:
    arguments = parse_arguments()
    set_determinism(0)
    device = resolve_device(arguments.device)
    config_dir = Path(arguments.config_dir)
    roi = tuple(int(v) for v in arguments.roi)
    sw_volume = tuple(int(v) for v in (arguments.sw_volume or
                                       [value + 48 for value in roi]))

    print(f"device: {device} | roi: {roi} | sliding-window volume: {sw_volume} | "
          f"repeats: {arguments.repeats}")
    header = f"{'model':<14}{'params':>12}{'GFLOPs':>10}{'fwd ms':>10}{'bwd ms':>10}" \
             f"{'peak MB':>10}{'sw s':>9}{'windows':>9}"
    print(header)
    print("-" * len(header))

    report = {"device": str(device), "roi": list(roi), "sw_volume": list(sw_volume),
              "repeats": arguments.repeats, "models": []}
    for name in arguments.models:
        if not (config_dir / f"{name}.yml").is_file():
            print(f"{name:<14}  missing config {config_dir / f'{name}.yml'}")
            continue
        try:
            result = benchmark_model(name, config_dir, arguments, device, sw_volume)
        except Exception as error:
            print(f"{name:<14}  skipped ({type(error).__name__}: {error})")
            continue
        report["models"].append(result)

        gflops = "-" if result["gflops"] is None else f"{result['gflops']:.1f}"
        backward = "-" if result["backward_ms"] is None else f"{result['backward_ms']:.1f}"
        peak = "-" if result["peak_memory_mb"] is None else f"{result['peak_memory_mb']:.0f}"
        sw = "-" if result["sliding_window_seconds"] is None \
            else f"{result['sliding_window_seconds']:.2f}"
        windows = "-" if result["sliding_window_windows"] is None \
            else str(result["sliding_window_windows"])
        print(f"{name:<14}{result['parameters']:>12,}{gflops:>10}{result['forward_ms']:>10.1f}"
              f"{backward:>10}{peak:>10}{sw:>9}{windows:>9}")

    dump_json(arguments.output, report)
    print(f"\nReport written to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
