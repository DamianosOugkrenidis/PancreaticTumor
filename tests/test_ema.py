import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.ptseg.ema import ModelEMA


def make_model() -> nn.Module:
    return nn.Linear(3, 3)


def test_update_moves_halfway_with_decay_half() -> None:
    model = make_model()
    ema = ModelEMA(model, decay=0.5)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    for averaged, current in zip(ema.module.parameters(), model.parameters()):
        difference = current.detach() - averaged.detach()
        assert torch.allclose(difference, torch.full_like(difference, 0.5), atol=1e-5), difference


def test_non_float_entries_are_copied() -> None:
    class WithBuffer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)
            self.register_buffer("steps", torch.tensor(0, dtype=torch.long))

    model = WithBuffer()
    ema = ModelEMA(model, decay=0.9)
    with torch.no_grad():
        model.steps.add_(5)
    ema.update(model)
    assert int(ema.module.steps) == 5


def test_store_and_restore_roundtrip() -> None:
    model = make_model()
    ema = ModelEMA(model, decay=0.9)
    original = [parameter.detach().clone() for parameter in model.parameters()]
    ema.store(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)
    ema.copy_to(model)
    ema.restore(model)
    for parameter, saved in zip(model.parameters(), original):
        assert torch.allclose(parameter, saved)


def test_invalid_decay_is_rejected() -> None:
    try:
        ModelEMA(make_model(), decay=1.0)
    except ValueError:
        return
    raise AssertionError("decay=1.0 should be rejected")


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
