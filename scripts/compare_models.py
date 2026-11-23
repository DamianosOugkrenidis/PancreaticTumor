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
    keys = ("pancreas_dice", "tumor_dice", "tumor_iou", "tumor_hd95", "tumor_nsd",
            "mean_dice", "mean_hd95", "mean_nsd", "num_cases")
    record = {key: _number(payload.get(key)) for key in keys}
    record["train_seconds"] = train_seconds
    size_path = run / "artifacts" / "tumor_size_analysis.json"
    small_dice = large_dice = None
    if size_path.is_file():
        groups = _read_json(size_path).get("groups") or {}
        small_dice = _number((groups.get("small") or {}).get("mean_tumor_dice"))
        large_dice = _number((groups.get("large") or {}).get("mean_tumor_dice"))
    record["small_tumor_dice"] = small_dice
    record["large_tumor_dice"] = large_dice
    record["run"] = run.name
    return record


def load_per_case(run: Path) -> dict:
    path = run / "artifacts" / "evaluation_per_case.json"
    if not path.is_file():
        return {}
    payload = _read_json(path)
    cases = payload.get("cases") or []
    values = ((payload.get("series") or {}).get("dice") or {}).get("tumor") or []
    return {str(name): float(value) for name, value in zip(cases, values)
            if value is not None and not (isinstance(value, float) and math.isnan(value))}


def pairwise_report(rows) -> dict:
    usable = [(row["model"], row.get("_per_case") or {}) for row in rows if row.get("_per_case")]
    report = {}
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            name_a, first = usable[i]
            name_b, second = usable[j]
            common = sorted(set(first) & set(second))
            if len(common) < 2:
                continue
            diffs = [first[case] - second[case] for case in common]
            p_value = None
            if len(common) >= 6 and any(abs(value) > 1e-9 for value in diffs):
                try:
                    from scipy.stats import wilcoxon

                    p_value = float(wilcoxon(diffs).pvalue)
                except Exception:
                    p_value = None
            report[f"{name_a} vs {name_b}"] = {
                "cases": len(common),
                "mean_tumor_dice_diff": sum(diffs) / len(diffs),
                "wilcoxon_p": p_value,
            }
    return report


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
            record["_per_case"] = load_per_case(run)
        else:
            record.update({key: None for key in ("pancreas_dice", "tumor_dice", "tumor_iou",
                                                 "tumor_hd95", "tumor_nsd", "mean_dice",
                                                 "mean_hd95", "mean_nsd", "num_cases",
                                                 "train_seconds", "small_tumor_dice",
                                                 "large_tumor_dice")})
            record["run"] = None
        rows.append(record)

    rows.sort(key=lambda row: (row.get("tumor_dice") is None, -(row.get("tumor_dice") or 0)))

    header = (f"{'model':<14}{'params(M)':>10}{'GFLOPs':>9}{'fwd ms':>9}{'tumor Dice':>11}"
              f"{'tumor IoU':>10}{'HD95(mm)':>10}{'panc Dice':>10}{'mean Dice':>10}"
              f"{'small Dice':>11}{'large Dice':>11}{'train(h)':>9}")
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
              f"{_fmt(row.get('small_tumor_dice')):>11}{_fmt(row.get('large_tumor_dice')):>11}"
              f"{_fmt(train_hours, '{:.1f}'):>9}")

    if not any(row.get("tumor_dice") is not None for row in rows):
        print("\nNo evaluation metrics found. Run scripts/train.py + scripts/evaluate.py, "
              "or pass --train to do both before building the table.")

    pairwise = pairwise_report(rows)
    if pairwise:
        print("\npaired tumor-Dice comparison (Wilcoxon signed-rank):")
        for label, stats in pairwise.items():
            p_value = "-" if stats["wilcoxon_p"] is None else f"{stats['wilcoxon_p']:.3f}"
            print(f"  {label:<34} n={stats['cases']:<3} "
                  f"mean diff {stats['mean_tumor_dice_diff']:+.3f}  p={p_value}")

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["model", "parameters", "gflops", "forward_ms", "pancreas_dice", "tumor_dice",
              "tumor_iou", "tumor_hd95", "tumor_nsd", "small_tumor_dice", "large_tumor_dice",
              "mean_dice", "mean_hd95", "mean_nsd", "num_cases", "train_seconds", "run"]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    clean_rows = [{key: value for key, value in row.items() if key != "_per_case"} for row in rows]
    dump_json(output_dir / "comparison.json", {"device": str(device), "roi": arguments.roi,
                                               "rows": clean_rows, "pairwise": pairwise})
    dump_json(output_dir / "pairwise_stats.json", pairwise)
    print(f"\nReport written to {output_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
