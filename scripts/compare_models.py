import argparse
import csv
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.ptseg.config import load_configs, validate_config
from src.ptseg.models import available_models, build_model
from src.ptseg.utils import count_parameters, dump_json, resolve_device, set_determinism


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click comparison of every model: segmentation metrics gathered from the "
                    "run artifacts plus computational cost measured on the fly.")
    parser.add_argument("--config-dir", default="configs")
    parser.add_argument("--models", nargs="*", default=available_models())
    parser.add_argument("--roi", type=int, nargs=3, default=[64, 64, 64], metavar=("D", "H", "W"))
    parser.add_argument("--device", default="cpu", help="auto|cpu|cuda.")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--no-flops", action="store_true")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--output-dir", default="outputs/comparison")
    parser.add_argument("--train", action="store_true",
                        help="First train and evaluate every model (long), then report.")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _model_of(config_path: Path):
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    return (payload.get("model") or {}).get("name")


def find_run(runs_dir: Path, model_name: str):
    best = None
    if runs_dir.is_dir():
        for run in runs_dir.iterdir():
            config_path = run / "artifacts" / "config.yml"
            eval_path = run / "artifacts" / "evaluation_metrics.json"
            if not (config_path.is_file() and eval_path.is_file()):
                continue
            if _model_of(config_path) != model_name:
                continue
            stamp = eval_path.stat().st_mtime
            if best is None or stamp > best[0]:
                best = (stamp, run)
    return best[1] if best else None


def _number(value):
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    return None


def gather_metrics(run: Path) -> dict:
    payload = _read_json(run / "artifacts" / "evaluation_metrics.json")
    summary_path = run / "artifacts" / "summary.json"
    train_seconds = None
    if summary_path.is_file():
        train_seconds = _number(_read_json(summary_path).get("total_seconds"))
    keys = ("pancreas_dice", "tumor_dice", "tumor_iou", "tumor_hd95", "mean_dice",
            "mean_hd95", "num_cases")
    record = {key: _number(payload.get(key)) for key in keys}
    record["train_seconds"] = train_seconds
    record["run"] = run.name
    return record


def measure_cost(name: str, config_dir: Path, arguments, device: torch.device) -> dict:
    config = load_configs(
        [config_dir / f"{name}.yml"],
        [f"model.img_size=[{arguments.roi[0]},{arguments.roi[1]},{arguments.roi[2]}]",
         f"data.roi_size=[{arguments.roi[0]},{arguments.roi[1]},{arguments.roi[2]}]"])
    validate_config(config)
    model = build_model(config.model.name, **config.model.kwargs).to(device).eval()
    parameters = count_parameters(model)[0]
    inputs = torch.randn(1, int(config.model.kwargs.get("in_channels", 1)), *arguments.roi,
                         device=device)

    flops = None
    if not arguments.no_flops:
        try:
            from torch.utils.flop_counter import FlopCounterMode

            counter = FlopCounterMode(display=False)
            with counter:
                model(inputs)
            flops = int(counter.get_total_flops()) / 1e9
        except Exception:
            flops = None

    with torch.no_grad():
        model(inputs)
        start = time.perf_counter()
        for _ in range(arguments.repeats):
            model(inputs)
        forward_ms = (time.perf_counter() - start) * 1000.0 / max(arguments.repeats, 1)
    return {"parameters": parameters, "gflops": flops, "forward_ms": forward_ms}


def train_and_evaluate(name: str, arguments) -> None:
    import subprocess

    run_name = f"cmp_{name}"
    subprocess.run([sys.executable, "scripts/train.py", "-c", f"configs/{name}.yml",
                    "--run-name", run_name, "--device", arguments.device],
                   cwd=PROJECT_ROOT, check=True)
    subprocess.run([sys.executable, "scripts/evaluate.py", "--run-dir",
                    f"outputs/runs/{run_name}", "--device", arguments.device],
                   cwd=PROJECT_ROOT, check=True)


def _fmt(value, pattern="{:.3f}"):
    return "-" if value is None else pattern.format(value)


def main() -> int:
    arguments = parse_arguments()
    set_determinism(0)
    device = resolve_device(arguments.device)
    config_dir = Path(arguments.config_dir)
    runs_dir = Path(arguments.runs_dir)
    output_dir = Path(arguments.output_dir)

    rows = []
    for name in arguments.models:
        if not (config_dir / f"{name}.yml").is_file():
            print(f"{name}: missing config")
            continue
        if arguments.train:
            train_and_evaluate(name, arguments)
        record = {"model": name}
        try:
            record.update(measure_cost(name, config_dir, arguments, device))
        except Exception as error:
            record.update({"parameters": None, "gflops": None, "forward_ms": None})
            print(f"{name}: cost measurement failed ({type(error).__name__}: {error})")
        run = find_run(runs_dir, name)
        if run is not None:
            record.update(gather_metrics(run))
        else:
            record.update({key: None for key in ("pancreas_dice", "tumor_dice", "tumor_iou",
                                                 "tumor_hd95", "mean_dice", "mean_hd95",
                                                 "num_cases", "train_seconds")})
            record["run"] = None
        rows.append(record)

    rows.sort(key=lambda row: (row.get("tumor_dice") is None, -(row.get("tumor_dice") or 0)))

    header = (f"{'model':<14}{'params(M)':>10}{'GFLOPs':>9}{'fwd ms':>9}{'tumor Dice':>11}"
              f"{'tumor IoU':>10}{'HD95(mm)':>10}{'panc Dice':>10}{'mean Dice':>10}"
              f"{'train(h)':>9}")
    print(f"device: {device} | roi: {tuple(arguments.roi)} | runs: {runs_dir}")
    print(header)
    print("-" * len(header))
    for row in rows:
        params = "-" if row.get("parameters") is None else f"{row['parameters'] / 1e6:.2f}"
        train_hours = None if row.get("train_seconds") is None else row["train_seconds"] / 3600.0
        print(f"{row['model']:<14}{params:>10}{_fmt(row.get('gflops'), '{:.1f}'):>9}"
              f"{_fmt(row.get('forward_ms'), '{:.0f}'):>9}"
              f"{_fmt(row.get('tumor_dice')):>11}{_fmt(row.get('tumor_iou')):>10}"
              f"{_fmt(row.get('tumor_hd95'), '{:.1f}'):>10}"
              f"{_fmt(row.get('pancreas_dice')):>10}{_fmt(row.get('mean_dice')):>10}"
              f"{_fmt(train_hours, '{:.1f}'):>9}")

    if not any(row.get("tumor_dice") is not None for row in rows):
        print("\nNo evaluation metrics found. Run scripts/train.py + scripts/evaluate.py, "
              "or pass --train to do both before building the table.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "parameters", "gflops", "forward_ms", "pancreas_dice", "tumor_dice",
              "tumor_iou", "tumor_hd95", "mean_dice", "mean_hd95", "num_cases",
              "train_seconds", "run"]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    dump_json(output_dir / "comparison.json", {"device": str(device), "roi": arguments.roi,
                                               "rows": rows})
    print(f"\nReport written to {output_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
