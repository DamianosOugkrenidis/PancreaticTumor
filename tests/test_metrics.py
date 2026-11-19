import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.ptseg.metrics import (MetricAggregator, evaluate_case, hausdorff_distance_95,
                               overlap_metrics, stratify_by_volume)


def test_overlap_metrics_counts_tp_fp_fn() -> None:
    prediction = np.zeros((4, 4, 4), dtype=np.uint8)
    ground_truth = np.zeros((4, 4, 4), dtype=np.uint8)
    prediction[0, 0, 0] = 1
    ground_truth[0, 0, 0] = 1
    prediction[1, 1, 1] = 1
    ground_truth[2, 2, 2] = 1

    scores = overlap_metrics(prediction == 1, ground_truth == 1)
    assert (scores["tp"], scores["fp"], scores["fn"]) == (1.0, 1.0, 1.0)
    assert abs(scores["dice"] - 0.5) < 1e-9
    assert abs(scores["iou"] - 1.0 / 3.0) < 1e-9
    assert abs(scores["precision"] - 0.5) < 1e-9
    assert abs(scores["recall"] - 0.5) < 1e-9


def test_overlap_metrics_empty_is_nan() -> None:
    empty = np.zeros((2, 2, 2), dtype=bool)
    scores = overlap_metrics(empty, empty)
    assert np.isnan(scores["dice"]) and np.isnan(scores["iou"])


def test_evaluate_case_skips_background() -> None:
    volume = np.zeros((5, 5, 5), dtype=np.uint8)
    volume[1:3, 1:3, 1:3] = 1
    volume[3, 3, 3] = 2
    result = evaluate_case(volume, volume, ("background", "pancreas", "tumor"),
                           include_background=False, metrics=("dice", "hd95"))
    assert set(result["dice"]) == {"pancreas", "tumor"}, result["dice"]
    assert abs(result["dice"]["pancreas"] - 1.0) < 1e-9
    assert result["hd95"]["tumor"] == 0.0


def test_hd95_zero_and_undefined() -> None:
    blob = np.zeros((6, 6, 6), dtype=bool)
    blob[2:4, 2:4, 2:4] = True
    assert hausdorff_distance_95(blob, blob) == 0.0
    assert np.isnan(hausdorff_distance_95(blob, np.zeros_like(blob)))


def test_aggregator_excludes_undefined_and_reports_counts() -> None:
    aggregator = MetricAggregator(class_names=("pancreas",), metric_names=("dice", "hd95"))
    aggregator.update({"dice": {"pancreas": 0.5}, "hd95": {"pancreas": float("nan")}})
    aggregator.update({"dice": {"pancreas": 1.0}, "hd95": {"pancreas": 2.0}})
    summary = aggregator.summary()
    assert abs(summary["pancreas_dice"] - 0.75) < 1e-9
    assert abs(summary["pancreas_hd95"] - 2.0) < 1e-9
    assert summary["undefined_hd95"] == 1.0


def test_stratify_by_volume_terciles() -> None:
    groups = stratify_by_volume([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    assert list(groups) == [0, 0, 0, 1, 1, 1, 2, 2, 2], list(groups)
    missing = stratify_by_volume([0.0, 0.0], 3)
    assert (missing == -1).all()


def main() -> int:
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
