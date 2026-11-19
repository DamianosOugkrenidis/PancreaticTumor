import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ptseg.config import load_configs, validate_config

CONFIGS = PROJECT_ROOT / "configs"


def expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def test_unetr_default_is_valid() -> None:
    config = load_configs([CONFIGS / "unetr.yml"])
    notes = validate_config(config)
    assert any("token grid" in note for note in notes), notes


def test_swin_unetr_default_is_valid() -> None:
    config = load_configs([CONFIGS / "swin_unetr.yml"])
    notes = validate_config(config)
    assert any("window" in note for note in notes), notes


def test_segformer3d_default_is_valid() -> None:
    config = load_configs([CONFIGS / "segformer3d.yml"])
    notes = validate_config(config)
    assert any("stride" in note for note in notes), notes


def test_unetr_rejects_indivisible_roi() -> None:
    config = load_configs([CONFIGS / "unetr.yml"],
                          ["data.roi_size=[95,96,96]", "model.img_size=[95,96,96]"])
    expect_value_error(lambda: validate_config(config))


def test_unetr_requires_img_size_equal_roi() -> None:
    config = load_configs([CONFIGS / "unetr.yml"], ["model.img_size=[64,64,64]"])
    expect_value_error(lambda: validate_config(config))


def test_segformer3d_rejects_small_roi() -> None:
    config = load_configs([CONFIGS / "segformer3d.yml"],
                          ["data.roi_size=[48,48,48]", "model.img_size=[48,48,48]"])
    expect_value_error(lambda: validate_config(config))


def test_out_channels_must_match_labels() -> None:
    config = load_configs([CONFIGS / "unetr.yml"], ["model.out_channels=2"])
    expect_value_error(lambda: validate_config(config))


def test_unsupported_scheduler_is_rejected() -> None:
    config = load_configs([CONFIGS / "unetr.yml"], ["train.scheduler=linear"])
    expect_value_error(lambda: validate_config(config))


def test_invalid_valid_ratio_is_rejected() -> None:
    config = load_configs([CONFIGS / "unetr.yml"], ["data.valid_ratio=1.5"])
    expect_value_error(lambda: validate_config(config))


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
