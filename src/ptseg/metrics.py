from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import ndimage

NEIGHBOURHOOD = ndimage.generate_binary_structure(3, 1)

def surface_voxels(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndimage.binary_erosion(mask, structure=NEIGHBOURHOOD, iterations=1, border_value=0)
    return mask & ~eroded

def hausdorff_distance_95(prediction: np.ndarray, ground_truth: np.ndarray,
                          spacing: Sequence[float] = (1.0, 1.0, 1.0),
                          percentile: float = 95.0) -> float:
    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    if not prediction.any() and not ground_truth.any():
        return 0.0
    if not prediction.any() or not ground_truth.any():
        return float("nan")

    surface_pred = surface_voxels(prediction)
    surface_gt = surface_voxels(ground_truth)

    distance_to_gt = ndimage.distance_transform_edt(~surface_gt, sampling=spacing)
    distance_to_pred = ndimage.distance_transform_edt(~surface_pred, sampling=spacing)

    pred_to_gt = distance_to_gt[surface_pred]
    gt_to_pred = distance_to_pred[surface_gt]
    return float(max(np.percentile(pred_to_gt, percentile),
                     np.percentile(gt_to_pred, percentile)))

def overlap_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    pred = prediction.astype(bool)
    gt = ground_truth.astype(bool)

    true_positive = int(np.count_nonzero(pred & gt))
    false_positive = int(np.count_nonzero(pred & ~gt))
    false_negative = int(np.count_nonzero(~pred & gt))

    dice_denominator = 2 * true_positive + false_positive + false_negative
    iou_denominator = true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    return {
        "dice": 2.0 * true_positive / dice_denominator if dice_denominator else float("nan"),
        "iou": true_positive / iou_denominator if iou_denominator else float("nan"),
        "precision": true_positive / precision_denominator if precision_denominator
        else float("nan"),
        "recall": true_positive / recall_denominator if recall_denominator else float("nan"),
        "tp": float(true_positive),
        "fp": float(false_positive),
        "fn": float(false_negative),
    }

def evaluate_case(prediction: np.ndarray, ground_truth: np.ndarray,
                  class_names: Sequence[str], spacing: Sequence[float] = (1.0, 1.0, 1.0),
                  include_background: bool = False, hd95_percentile: float = 95.0,
                  metrics: Iterable[str] = ("dice", "iou", "precision", "recall", "hd95")
                  ) -> dict[str, dict[str, float]]:
    metrics = tuple(metrics)
    results: dict[str, dict[str, float]] = {name: {} for name in metrics}

    for index, class_name in enumerate(class_names):
        if index == 0 and not include_background:
            continue
        pred_mask = prediction == index
        gt_mask = ground_truth == index

        overlap = overlap_metrics(pred_mask, gt_mask)
        for name in ("dice", "iou", "precision", "recall"):
            if name in metrics:
                results[name][class_name] = overlap[name]
        if "hd95" in metrics:
            results["hd95"][class_name] = hausdorff_distance_95(
                pred_mask, gt_mask, spacing, hd95_percentile)

    return results

@dataclass
class MetricAggregator:

    class_names: Sequence[str]
    metric_names: Sequence[str] = ("dice", "iou", "precision", "recall", "hd95")
    values: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    undefined: dict[str, int] = field(default_factory=dict)
    num_cases: int = 0

    def __post_init__(self) -> None:
        if not self.values:
            self.values = {name: {cls: [] for cls in self.class_names}
                           for name in self.metric_names}
            self.undefined = {name: 0 for name in self.metric_names}

    def update(self, case_metrics: dict[str, dict[str, float]]) -> None:
        self.num_cases += 1
        for name, per_class in case_metrics.items():
            for class_name, value in per_class.items():
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    self.undefined[name] = self.undefined.get(name, 0) + 1
                    continue
                self.values.setdefault(name, {}).setdefault(class_name, []).append(float(value))

    def mean(self, metric: str, class_name: str | None = None) -> float:
        if class_name is not None:
            series = self.values.get(metric, {}).get(class_name, [])
            return float(np.mean(series)) if series else float("nan")
        series = [value for per_class in self.values.get(metric, {}).values() for value in per_class]
        return float(np.mean(series)) if series else float("nan")

    def summary(self) -> dict[str, float]:
        payload: dict[str, float] = {"num_cases": float(self.num_cases)}
        for metric in self.metric_names:
            for class_name in self.class_names:
                payload[f"{class_name}_{metric}"] = self.mean(metric, class_name)
            payload[f"mean_{metric}"] = self.mean(metric)
        for metric, count in self.undefined.items():
            payload[f"undefined_{metric}"] = float(count)
        return payload

    def per_case_series(self) -> dict[str, dict[str, list[float]]]:
        return {metric: {cls: list(series) for cls, series in per_class.items()}
                for metric, per_class in self.values.items()}

def stratify_by_volume(volumes, n_groups: int = 3) -> np.ndarray:
    values = np.asarray(volumes, dtype=float)
    groups = np.full(values.shape, -1, dtype=int)
    valid = np.isfinite(values) & (values > 0)
    if not valid.any() or n_groups < 1:
        return groups
    edges = np.quantile(values[valid], np.linspace(0.0, 1.0, n_groups + 1)[1:-1])
    groups[valid] = np.searchsorted(edges, values[valid], side="right")
    return groups
