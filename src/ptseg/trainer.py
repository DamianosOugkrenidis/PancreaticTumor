import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .ema import ModelEMA
from .inference import sliding_window_inference
from .losses import build_loss, dice_score
from .metrics import MetricAggregator, evaluate_case
from .models import build_model, describe_model
from .utils import Timer, count_parameters, dump_json, format_duration

class Trainer:

    def __init__(self, config: Config, train_loader: Any, val_loader: Any, run_dir: str | Path,
                 device: torch.device, logger: Any) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.logger = logger

        self.class_names = list(config.data.labels.keys())
        self.selection_metric = config.eval.selection_metric
        self.epochs = int(config.train.epochs)
        self.accumulation_steps = int(config.train.grad_accum_steps)

        self.model = build_model(config.model.name, **config.model.kwargs).to(device)
        self.loss = build_loss(config.train.loss, config.data.num_classes).to(device)

        total, trainable = count_parameters(self.model)
        self.logger.info(f"Model {config.model.name}: {total:,} parameters ({trainable:,} trainable)")
        summary = describe_model(self.model)
        if summary:
            self.logger.info(f"Architecture: {summary}")

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = self._build_scaler()

        ema_decay = float(config.train.ema_decay)
        self.ema = ModelEMA(self.model, ema_decay) if 0.0 < ema_decay < 1.0 else None

        self.history: list[dict[str, Any]] = []
        self.best_metric = -math.inf
        self.best_epoch = 0
        self.start_epoch = 1
        self.patience_counter = 0

        self.writer = self._build_writer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        train_cfg = self.config.train
        name = train_cfg.optimizer.lower()
        kwargs = {"lr": float(train_cfg.lr), "weight_decay": float(train_cfg.weight_decay)}
        if name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), **kwargs)
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), **kwargs)
        if name == "sgd":
            kwargs.pop("weight_decay")
            return torch.optim.SGD(self.model.parameters(), momentum=0.9, nesterov=True,
                                   weight_decay=float(train_cfg.weight_decay), **kwargs)
        raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer!r}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        train_cfg = self.config.train
        if train_cfg.scheduler == "none":
            return None
        warmup = max(0, int(train_cfg.warmup_epochs))
        total = max(1, self.epochs)
        eta_ratio = float(train_cfg.eta_min) / max(float(train_cfg.lr), 1e-12)

        def schedule(epoch: int) -> float:
            if warmup and epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return eta_ratio + (1.0 - eta_ratio) * cosine

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=schedule)

    def _build_scaler(self) -> Any | None:
        if not self.config.train.amp or self.device.type != "cuda":
            return None
        try:
            return torch.amp.GradScaler("cuda", enabled=True)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=True)

    def _build_writer(self) -> Any | None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return None
        return SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))

    def _autocast(self) -> Any:
        if self.scaler is None:
            return nullcontext()
        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except (AttributeError, TypeError):
            return torch.cuda.amp.autocast(enabled=True)

    def train_epoch(self, epoch: int) -> dict[str, Any]:
        self.model.train()
        running_loss = 0.0
        dice_sums = torch.zeros(len(self.class_names) - 1, device=self.device)
        steps = 0

        self.optimizer.zero_grad(set_to_none=True)
        with Timer() as timer:
            for step, batch in enumerate(self.train_loader):
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)

                with self._autocast():
                    logits = self.model(images)
                    loss = self.loss(logits, labels) / self.accumulation_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.accumulation_steps == 0:
                    if self.config.train.grad_clip_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 float(self.config.train.grad_clip_norm))
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.model)

                running_loss += float(loss.detach()) * self.accumulation_steps
                with torch.no_grad():
                    dice_sums += dice_score(logits.detach(), labels, len(self.class_names))
                steps += 1

                if self.config.train.log_every_steps and \
                        (step + 1) % int(self.config.train.log_every_steps) == 0:
                    self.logger.info(
                        f"epoch {epoch}/{self.epochs} | step {step + 1}/{len(self.train_loader)} "
                        f"| loss {running_loss / steps:.4f}")

                if int(self.config.train.max_steps) and (step + 1) >= int(self.config.train.max_steps):
                    self.logger.info(f"Stopping the epoch after {step + 1} steps "
                                     "(train.max_steps).")
                    break

        if steps % self.accumulation_steps:
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)

        dice = (dice_sums / max(steps, 1)).detach().cpu().numpy()
        metrics: dict[str, Any] = {
            "train_loss": running_loss / max(steps, 1),
            "train_seconds": timer.elapsed,
            "train_steps": steps,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        for index, class_name in enumerate(self.class_names[1:]):
            metrics[f"train_dice_{class_name}"] = float(dice[index])
        return metrics

    def validate(self, epoch: int) -> dict[str, Any]:
        report_classes = list(self.class_names) if self.config.eval.include_background \
            else list(self.class_names[1:])
        aggregator = MetricAggregator(class_names=report_classes,
                                      metric_names=self.config.eval.metrics)
        if self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
        self.model.eval()
        max_cases = int(self.config.eval.max_cases)
        with Timer() as timer:
            with torch.no_grad():
                for index, batch in enumerate(self.val_loader):
                    if max_cases and index >= max_cases:
                        break
                    images = batch["image"].to(self.device, non_blocking=True)
                    labels = batch["label"]

                    logits = sliding_window_inference(
                        images, self.config.data.roi_size, self.model,
                        sw_batch_size=int(self.config.eval.sw_batch_size),
                        overlap=float(self.config.eval.overlap), mode=self.config.eval.mode)
                    prediction = logits.argmax(dim=1)[0].detach().cpu().numpy()

                    ground_truth = labels[0]
                    if ground_truth.dim() == 4:
                        ground_truth = ground_truth[0]
                    ground_truth = ground_truth.detach().cpu().numpy()

                    case_metrics = evaluate_case(
                        prediction, ground_truth, self.class_names,
                        spacing=tuple(float(v) for v in self.config.data.pixdim),
                        include_background=self.config.eval.include_background,
                        hd95_percentile=float(self.config.eval.hd95_percentile),
                        metrics=self.config.eval.metrics)
                    aggregator.update(case_metrics)

        summary = aggregator.summary()
        if self.ema is not None:
            self.ema.restore(self.model)
        summary["val_seconds"] = timer.elapsed
        summary["epoch"] = epoch
        return summary

    def fit(self) -> dict[str, Any]:
        started = time.perf_counter()
        self.logger.info(f"Training on {self.device} for {self.epochs} epochs "
                         f"(selection metric: {self.selection_metric})")

        for epoch in range(self.start_epoch, self.epochs + 1):
            record: dict[str, Any] = {"epoch": epoch}
            record.update(self.train_epoch(epoch))

            interval = int(self.config.train.val_interval)
            run_validation = interval > 0 and (epoch % interval == 0 or epoch == self.epochs)
            if run_validation:
                validation = self.validate(epoch)
                record["val"] = validation
                metric = float(validation.get(self.selection_metric, float("nan")))
                self._track_metric(epoch, metric)

            self.history.append(record)
            self._log_epoch(record)
            self._write_tensorboard(record, epoch)

            self._save_checkpoint(self.run_dir / "checkpoints" / "last.pt", epoch)
            if self.config.train.save_every_epochs and \
                    epoch % int(self.config.train.save_every_epochs) == 0:
                self._save_checkpoint(self.run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt", epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            if self._should_stop(epoch):
                self.logger.info(f"Early stopping at epoch {epoch} "
                                 f"(best {self.selection_metric}={self.best_metric:.4f} "
                                 f"at epoch {self.best_epoch})")
                break

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        summary = {
            "epochs_completed": len(self.history),
            "configured_epochs": self.epochs,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "selection_metric": self.selection_metric,
            "total_seconds": time.perf_counter() - started,
            "seconds_per_epoch": (time.perf_counter() - started) / max(len(self.history), 1),
            "parameters": count_parameters(self.model)[0],
        }
        dump_json(self.run_dir / "artifacts" / "training_history.json",
                  {"history": self.history, "summary": summary})
        self.logger.info(f"Training finished in {format_duration(summary['total_seconds'])} | "
                         f"best {self.selection_metric}: {self.best_metric:.4f}")
        return {"history": self.history, "summary": summary}

    def _track_metric(self, epoch: int, metric: float) -> None:
        if math.isnan(metric):
            self.logger.warning(f"Epoch {epoch}: {self.selection_metric} is NaN; "
                                "skipping model selection for this epoch.")
            return
        improvement = metric - self.best_metric
        if improvement > float(self.config.train.early_stopping.min_delta):
            self.best_metric = metric
            self.best_epoch = epoch
            self.patience_counter = 0
            self._save_checkpoint(self.run_dir / "checkpoints" / "best.pt", epoch)
            self.logger.info(f"New best {self.selection_metric}={metric:.4f} (epoch {epoch})")
        else:
            self.patience_counter += 1

    def _should_stop(self, epoch: int) -> bool:
        early = self.config.train.early_stopping
        return bool(early.enabled and epoch < self.epochs and
                    self.patience_counter >= int(early.patience))

    def _log_epoch(self, record: dict[str, Any]) -> None:
        parts = [f"epoch {record['epoch']}/{self.epochs}",
                 f"loss {record['train_loss']:.4f}",
                 f"lr {record['lr']:.2e}",
                 f"{record['train_seconds']:.0f}s train"]
        validation = record.get("val")
        if validation:
            for class_name in self.class_names[1:]:
                parts.append(f"{class_name}_dice {validation.get(f'{class_name}_dice', float('nan')):.4f}")
            if "mean_hd95" in validation:
                parts.append(f"mean_hd95 {validation['mean_hd95']:.2f}mm")
            parts.append(f"{validation['val_seconds']:.0f}s val")
        self.logger.info(" | ".join(parts))

    def _write_tensorboard(self, record: dict[str, Any], epoch: int) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", record["train_loss"], epoch)
        self.writer.add_scalar("train/lr", record["lr"], epoch)
        for key, value in record.items():
            if key.startswith("train_dice_"):
                self.writer.add_scalar(f"train/{key}", value, epoch)
        validation = record.get("val") or {}
        for key, value in validation.items():
            if isinstance(value, float) and key not in {"epoch", "val_seconds"}:
                self.writer.add_scalar(f"val/{key}", value, epoch)

    def _save_checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "scaler": self.scaler.state_dict() if self.scaler else None,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "model_name": self.config.model.name,
            "model_kwargs": self.config.model.kwargs,
            "selection_metric": self.selection_metric,
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }, path)

    def load_checkpoint(self, path: str | Path, resume_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        if self.ema is not None and checkpoint.get("ema"):
            self.ema.module.load_state_dict(checkpoint["ema"])
        if resume_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            if self.scheduler is not None and checkpoint.get("scheduler"):
                self.scheduler.load_state_dict(checkpoint["scheduler"])
            if self.scaler is not None and checkpoint.get("scaler"):
                self.scaler.load_state_dict(checkpoint["scaler"])
            self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_metric = float(checkpoint.get("best_metric", -math.inf))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.logger.info(f"Resumed from {path} (epoch {checkpoint['epoch']}, "
                         f"best {self.selection_metric}={self.best_metric:.4f})")
        return checkpoint

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        if self.ema is not None:
            self.ema.store(self.model)
            self.ema.copy_to(self.model)
        try:
            return sliding_window_inference(images, self.config.data.roi_size, self.model,
                                            sw_batch_size=int(self.config.eval.sw_batch_size),
                                            overlap=float(self.config.eval.overlap),
                                            mode=self.config.eval.mode)
        finally:
            if self.ema is not None:
                self.ema.restore(self.model)
