from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig

def to_one_hot(targets: torch.Tensor, num_classes: int, logits: torch.Tensor) -> torch.Tensor:
    if targets.dim() == logits.dim() and targets.shape[1] == logits.shape[1]:
        return targets.float()
    if targets.dim() == logits.dim() - 1:
        targets = targets.unsqueeze(1)
    if targets.dim() != logits.dim():
        raise ValueError(f"Cannot interpret targets of shape {tuple(targets.shape)} against "
                         f"logits of shape {tuple(logits.shape)}.")
    if targets.shape[1] != 1:
        raise ValueError(f"Expected a label map with one channel, got {targets.shape[1]}.")
    squeezed = targets[:, 0].long()
    if squeezed.min() < 0 or squeezed.max() >= num_classes:
        raise ValueError(f"Labels out of range [0, {num_classes - 1}]: "
                         f"min={int(squeezed.min())}, max={int(squeezed.max())}.")
    one_hot = F.one_hot(squeezed, num_classes=num_classes)
    return one_hot.permute(0, 4, 1, 2, 3).float()

def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int | None = None,
                   include_background: bool = False, class_weights: Sequence[float] | None = None,
                   eps: float = 1e-5) -> torch.Tensor:
    num_classes = num_classes or logits.shape[1]
    one_hot = to_one_hot(targets, num_classes, logits)
    probabilities = logits.softmax(dim=1)

    dims = (0, 2, 3, 4)
    intersection = (probabilities * one_hot).sum(dim=dims)
    cardinality = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
    dice_per_class = (2.0 * intersection + eps) / (cardinality + eps)
    present = one_hot.sum(dim=dims) > 0

    selected = slice(1, num_classes) if not include_background else slice(0, num_classes)
    dice = dice_per_class[selected]
    presence = present[selected].to(dice.dtype)

    if float(presence.sum()) == 0.0:
        return logits.new_zeros(())

    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=dice.dtype, device=dice.device)
        weights = (weights[selected] if weights.numel() == num_classes else weights)[: dice.numel()]
        weights = weights * presence
    else:
        weights = presence

    return 1.0 - (dice * weights).sum() / weights.sum().clamp_min(eps)

def soft_tversky_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int | None = None,
                      alpha: float = 0.3, beta: float = 0.7, include_background: bool = False,
                      class_weights: Sequence[float] | None = None,
                      eps: float = 1e-5) -> torch.Tensor:
    num_classes = num_classes or logits.shape[1]
    one_hot = to_one_hot(targets, num_classes, logits)
    probabilities = logits.softmax(dim=1)

    dims = (0, 2, 3, 4)
    true_positive = (probabilities * one_hot).sum(dim=dims)
    false_positive = (probabilities * (1.0 - one_hot)).sum(dim=dims)
    false_negative = ((1.0 - probabilities) * one_hot).sum(dim=dims)
    tversky = (true_positive + eps) / (true_positive + alpha * false_positive
                                       + beta * false_negative + eps)

    present = one_hot.sum(dim=dims) > 0
    selected = slice(1, num_classes) if not include_background else slice(0, num_classes)
    tversky = tversky[selected]
    presence = present[selected].to(tversky.dtype)
    if float(presence.sum()) == 0.0:
        return logits.new_zeros(())

    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=tversky.dtype, device=tversky.device)
        weights = (weights[selected] if weights.numel() == num_classes else weights)[: tversky.numel()]
        weights = weights * presence
    else:
        weights = presence
    return 1.0 - (tversky * weights).sum() / weights.sum().clamp_min(eps)

def soft_focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
                    class_weights: Sequence[float] | None = None,
                    include_background: bool = False) -> torch.Tensor:
    if targets.dim() == logits.dim() and targets.shape[1] == logits.shape[1]:
        targets = targets.argmax(dim=1, keepdim=True)
    if targets.dim() == logits.dim() - 1:
        targets = targets.unsqueeze(1)
    labels = targets[:, 0].long()

    log_probabilities = F.log_softmax(logits, dim=1)
    log_probability = log_probabilities.gather(1, labels.unsqueeze(1)).squeeze(1)
    probability = log_probability.exp()
    focal = (1.0 - probability) ** gamma * (-log_probability)

    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=focal.dtype, device=focal.device)
        focal = focal * weights[labels]
    if not include_background:
        foreground = labels != 0
        if foreground.any():
            focal = focal[foreground]
    return focal.mean()

def dice_score(logits: torch.Tensor, targets: torch.Tensor, num_classes: int | None = None,
               include_background: bool = False) -> torch.Tensor:
    num_classes = num_classes or logits.shape[1]
    one_hot = to_one_hot(targets, num_classes, logits)
    probabilities = logits.argmax(dim=1, keepdim=True)
    probabilities = F.one_hot(probabilities[:, 0], num_classes).permute(0, 4, 1, 2, 3).float()

    dims = (0, 2, 3, 4)
    intersection = (probabilities * one_hot).sum(dim=dims)
    cardinality = probabilities.sum(dim=dims) + one_hot.sum(dim=dims)
    dice = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
    return dice if include_background else dice[1:]

class DiceOnlyLoss(nn.Module):
    def __init__(self, num_classes: int, include_background: bool = False,
                 class_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.include_background = bool(include_background)
        self.class_weights = tuple(class_weights) if class_weights is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_dice_loss(logits, targets, self.num_classes, self.include_background,
                              self.class_weights)

class TverskyLoss(nn.Module):

    def __init__(self, num_classes: int, alpha: float = 0.3, beta: float = 0.7,
                 include_background: bool = False,
                 class_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.include_background = bool(include_background)
        self.class_weights = tuple(class_weights) if class_weights is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_tversky_loss(logits, targets, self.num_classes, self.alpha, self.beta,
                                 self.include_background, self.class_weights)

class FocalLoss(nn.Module):

    def __init__(self, num_classes: int, gamma: float = 2.0, include_background: bool = False,
                 class_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.gamma = float(gamma)
        self.include_background = bool(include_background)
        self.class_weights = tuple(class_weights) if class_weights is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return soft_focal_loss(logits, targets, self.gamma, self.class_weights,
                               self.include_background)

class DiceCELoss(nn.Module):

    def __init__(self, num_classes: int, dice_weight: float = 1.0, ce_weight: float = 1.0,
                 include_background: bool = False,
                 class_weights: Sequence[float] | None = None) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.dice_weight = float(dice_weight)
        self.ce_weight = float(ce_weight)
        self.include_background = bool(include_background)
        self.register_buffer(
            "ce_class_weights",
            torch.as_tensor(class_weights, dtype=torch.float32) if class_weights is not None
            else torch.ones(num_classes, dtype=torch.float32),
            persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == logits.dim() and targets.shape[1] == logits.shape[1]:
            targets = targets.argmax(dim=1, keepdim=True)
        if targets.dim() == logits.dim() - 1:
            targets = targets.unsqueeze(1)
        labels = targets[:, 0].long()

        loss = logits.new_zeros(())
        if self.dice_weight:
            loss = loss + self.dice_weight * soft_dice_loss(
                logits, labels.unsqueeze(1), self.num_classes, self.include_background)
        if self.ce_weight:
            loss = loss + self.ce_weight * F.cross_entropy(
                logits, labels, weight=self.ce_class_weights.to(logits.device))
        return loss

def build_loss(config: LossConfig, num_classes: int) -> nn.Module:
    name = (config.name or "dice_ce").lower()
    if name in {"dice_ce", "dicece"}:
        return DiceCELoss(num_classes, config.dice_weight, config.ce_weight,
                          config.include_background, config.class_weights)
    if name == "dice":
        return DiceOnlyLoss(num_classes, config.include_background, config.class_weights)
    if name == "tversky":
        return TverskyLoss(num_classes, config.tversky_alpha, config.tversky_beta,
                           config.include_background, config.class_weights)
    if name == "focal":
        return FocalLoss(num_classes, config.focal_gamma, config.include_background,
                         config.class_weights)
    raise ValueError(f"Unknown loss name: {config.name!r} "
                     "(expected dice_ce, dice, tversky or focal).")
