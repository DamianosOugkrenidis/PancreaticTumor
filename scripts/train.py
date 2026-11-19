import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ptseg.config import load_configs, validate_config
from src.ptseg.data.msd import build_dataloaders
from src.ptseg.trainer import Trainer
from src.ptseg.utils import (device_summary, dump_json, dump_yaml, get_logger,
                             resolve_device, set_determinism, short_id)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3D segmentation model.")
    parser.add_argument("--config", "-c", nargs="+", required=True,
                        help="YAML configuration file(s); later files override earlier ones.")
    parser.add_argument("--set", nargs="*", default=[],
                        help="Dotted overrides, e.g. train.epochs=300 data.roi_size=[64,64,64].")
    parser.add_argument("--device", default=None, help="Override run.device (auto|cpu|cuda).")
    parser.add_argument("--run-name", default=None, help="Override run.run_name.")
    parser.add_argument("--output-dir", default=None, help="Override run.output_dir.")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run for pipeline validation: 2 epochs, no AMP, no workers.")
    return parser.parse_args()

def main() -> int:
    arguments = parse_arguments()
    overrides = list(arguments.set)

    if arguments.device:
        overrides.append(f"run.device={arguments.device}")
    if arguments.run_name:
        overrides.append(f"run.run_name={arguments.run_name}")
    if arguments.output_dir:
        overrides.append(f"run.output_dir={arguments.output_dir}")
    if arguments.smoke:
        overrides += ["train.epochs=2", "train.amp=false", "train.val_interval=1",
                      "train.log_every_steps=1", "train.early_stopping.enabled=false",
                      "data.num_workers=0"]

    config = load_configs(arguments.config, overrides)
    notes = validate_config(config)

    set_determinism(config.run.seed, config.run.deterministic)
    device = resolve_device(config.run.device, config.run.require_cuda)

    run_name = config.run.run_name or \
        f"{config.model.name}_{short_id(config.to_dict())}"
    run_dir = Path(config.run.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("ptseg.train", run_dir / "train.log")
    logger.info(f"Configuration: {config.source}")
    logger.info(f"Device: {device_summary(device)} | run dir: {run_dir}")
    for note in notes:
        logger.info(f"Note: {note}")

    dump_yaml(run_dir / "artifacts" / "config.yml", config.to_dict())
    dump_json(run_dir / "artifacts" / "environment.json", {
        "device": str(device),
        "device_summary": device_summary(device),
        "torch": __import__("torch").__version__,
        "cuda_available": __import__("torch").cuda.is_available(),
        "config_source": config.source,
        "run_name": run_name,
    })

    train_loader, val_loader, data_metadata = build_dataloaders(config.data, config.run.seed,
                                                               logger)
    dump_json(run_dir / "artifacts" / "data_split.json", data_metadata)

    trainer = Trainer(config, train_loader, val_loader, run_dir, device, logger)
    if arguments.resume:
        trainer.load_checkpoint(arguments.resume)

    result = trainer.fit()
    dump_json(run_dir / "artifacts" / "summary.json", result["summary"])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
