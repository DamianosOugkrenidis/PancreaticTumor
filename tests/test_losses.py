import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from src.ptseg.config import LossConfig
from src.ptseg.losses import build_loss, soft_focal_loss, soft_tversky_loss


def make_labels() -> torch.Tensor:
    labels = torch.zeros((1, 1, 8, 8, 8), dtype=torch.long)
    labels[..., 1:4, 1:4] = 1
    labels[..., 5:7, 5:7] = 2
    return labels


def perfect_logits(labels: torch.Tensor, num_classes: int = 3, scale: float = 12.0):
    one_hot = F.one_hot(labels[:, 0].long(), num_classes).permute(0, 4, 1, 2, 3).float()
    return one_hot * scale - scale / 2.0


def test_tversky_perfect_is_low() -> None:
    labels = make_labels()
    loss = float(soft_tversky_loss(perfect_logits(labels), labels, num_classes=3))
    assert loss < 0.05, loss


def test_tversky_penalises_errors() -> None:
    labels = make_labels()
    damaged = perfect_logits(labels)
    damaged[:, 2] = -damaged[:, 2]
    assert float(soft_tversky_loss(damaged, labels, num_classes=3)) > 0.3


def test_focal_is_finite_and_differentiable() -> None:
    labels = make_labels()
    logits = perfect_logits(labels).requires_grad_(True)
    loss = soft_focal_loss(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_build_loss_selects_names() -> None:
    labels = make_labels()
    for name in ("dice_ce", "dice", "tversky", "focal"):
        module = build_loss(LossConfig(name=name), num_classes=3)
        value = module(perfect_logits(labels), labels)
        assert torch.isfinite(value), name


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
