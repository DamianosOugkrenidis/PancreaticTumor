import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.ptseg.config import Config, load_configs
from src.ptseg.data.msd import build_dataloaders
from src.ptseg.inference import sliding_window_inference, stats_positions
from src.ptseg.metrics import MetricAggregator, evaluate_case
from src.ptseg.models import build_model, describe_model
from src.ptseg.utils import (count_parameters, device_summary, dump_json, get_logger,
                             resolve_device, set_determinism)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--run-dir", required=True, help="Run directory created by scripts/train.py.")
    parser.add_argument("--config", "-c", nargs="*", default=None,
                        help="Optional config override; defaults to <run-dir>/artifacts/config.yml.")
    parser.add_argument("--set", nargs="*", default=[], help="Dotted configuration overrides.")
    parser.add_argument("--checkpoint", default="best.pt", help="Checkpoint file inside run-dir.")
    parser.add_argument("--device", default=None, help="Override run.device (auto|cpu|cuda).")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Store per-case predicted label maps as .npy files.")
    return parser.parse_args()

def _write_summary_csv(path: Path, summary: dict[str, float], class_names: list[str],
                       metric_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["metric"] + class_names + ["mean"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for metric in metric_names:
            row = [metric]
            row += [f"{summary.get(f'{name}_{metric}', float('nan')):.4f}" for name in class_names]
            row += [f"{summary.get(f'mean_{metric}', float('nan')):.4f}"]
            writer.writerow(row)

def main() -> int:
    arguments = parse_arguments()
    run_dir = Path(arguments.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    config_paths = arguments.config or [str(run_dir / "artifacts" / "config.yml")]
    overrides = list(arguments.set)
    if arguments.device:
        overrides.append(f"run.device={arguments.device}")
    config: Config = load_configs(config_paths, overrides)

    set_determinism(config.run.seed, config.run.deterministic)
    device = resolve_device(config.run.device, config.run.require_cuda)
    logger = get_logger("ptseg.evaluate", run_dir / "evaluate.log")
    logger.info(f"Evaluating {config.model.name} from {run_dir} on {device_summary(device)}")

    checkpoint_path = run_dir / "checkpoints" / arguments.checkpoint
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = build_model(config.model.name, **config.model.kwargs).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    total, _ = count_parameters(model)
    logger.info(f"Checkpoint epoch {checkpoint.get('epoch')} | {total:,} parameters | "
                f"{describe_model(model)}")

    _, val_loader, metadata = build_dataloaders(config.data, config.run.seed, logger)
    class_names = list(config.data.labels.keys())
    report_classes = class_names if config.eval.include_background else class_names[1:]

    aggregator = MetricAggregator(class_names=report_classes, metric_names=config.eval.metrics)
    prediction_dir = run_dir / "predictions"
    case_names: list[str] = []
    window_counts: list[int] = []

    with torch.no_grad():
        for index, batch in enumerate(val_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"]
            name = Path(str(batch.get("image_meta_dict", {}).get("filename_or_obj", [f"case_{index:03d}"])[0])).name

            logits = sliding_window_inference(images, config.data.roi_size, model,
                                              sw_batch_size=int(config.eval.sw_batch_size),
                                              overlap=float(config.eval.overlap),
                                              mode=config.eval.mode)
            prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            ground_truth = labels[0]
            if ground_truth.dim() == 4:
                ground_truth = ground_truth[0]
            ground_truth = ground_truth.cpu().numpy()

            case_metrics = evaluate_case(
                prediction, ground_truth, class_names,
                spacing=tuple(float(v) for v in config.data.pixdim),
                include_background=config.eval.include_background,
                hd95_percentile=float(config.eval.hd95_percentile), metrics=config.eval.metrics)
            aggregator.update(case_metrics)
            case_names.append(name)
            window_counts.append(stats_positions(
                config.data.roi_size, tuple(int(v) for v in images.shape[-3:]),
                float(config.eval.overlap)))

            summary_line = " | ".join(
                f"{cls}_dice {case_metrics['dice'].get(cls, float('nan')):.4f}"
                for cls in report_classes)
            logger.info(f"[{index + 1}/{len(val_loader)}] {name} | {summary_line}")

            if arguments.save_predictions:
                prediction_dir.mkdir(parents=True, exist_ok=True)
                np.save(prediction_dir / f"{name}_pred.npy", prediction)

    summary = aggregator.summary()
    summary["cases"] = case_names
    summary.update({
        "model": config.model.name,
        "checkpoint": str(checkpoint_path),
        "parameters": total,
        "windows_per_case": window_counts,
        "mean_windows_per_case": float(np.mean(window_counts)) if window_counts else 0.0,
    })

    dump_json(run_dir / "artifacts" / "evaluation_metrics.json", summary)
    dump_json(run_dir / "artifacts" / "evaluation_per_case.json", {
        "cases": case_names,
        "series": aggregator.per_case_series(),
    })
    _write_summary_csv(run_dir / "artifacts" / "evaluation_table.csv", summary, report_classes,
                       list(config.eval.metrics))

    logger.info("--- mean metrics (validation split) ---")
    for metric in config.eval.metrics:
        per_class = " | ".join(
            f"{cls} {summary.get(f'{cls}_{metric}', float('nan')):.4f}" for cls in report_classes)
        logger.info(f"{metric:<10} {per_class} | mean "
                    f"{summary.get(f'mean_{metric}', float('nan')):.4f}")
    if summary.get("undefined_hd95"):
        logger.warning(f"HD95 was undefined in {int(summary['undefined_hd95'])} class/case pairs "
                       "(empty prediction or empty reference) and excluded from the mean.")
    logger.info(f"Validation volumes: {metadata['num_valid_volumes']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
