import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.ptseg.data.msd import kfold_splits
from src.ptseg.utils import apply_intensity_window


def test_apply_intensity_window_scales_to_range() -> None:
    values = np.array([-500.0, -175.0, 37.5, 250.0, 900.0], dtype=np.float32)
    scaled = apply_intensity_window(values, -175.0, 250.0, (0.0, 1.0))
    assert abs(scaled[0]) < 1e-6
    assert abs(scaled[1]) < 1e-6
    assert abs(scaled[-1] - 1.0) < 1e-6
    assert abs(scaled[2] - 0.5) < 1e-6


def test_apply_intensity_window_handles_constant_range() -> None:
    values = np.array([1.0, 2.0], dtype=np.float32)
    scaled = apply_intensity_window(values, 5.0, 5.0, (0.0, 1.0))
    assert np.allclose(scaled, 0.0)


def test_kfold_partitions_every_entry_once() -> None:
    files = [{"image": f"case_{index:02d}"} for index in range(11)]
    splits = kfold_splits(files, num_folds=5, seed=0)
    assert len(splits) == 5
    for training, validation in splits:
        assert len(training) + len(validation) == 11
        assert not ({entry["image"] for entry in training}
                    & {entry["image"] for entry in validation})
    seen = sorted(entry["image"] for _, validation in splits for entry in validation)
    assert seen == sorted(entry["image"] for entry in files)


def test_kfold_rejects_single_fold() -> None:
    try:
        kfold_splits([{"image": "a"}], num_folds=1)
    except ValueError:
        return
    raise AssertionError("num_folds=1 should be rejected")


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
