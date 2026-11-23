import copy

import torch
import torch.nn as nn


class ModelEMA:

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}.")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        self._backup: dict | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            averaged = self.module.state_dict()
            current = model.state_dict()
            for key, value in averaged.items():
                source = current[key].detach()
                if value.dtype.is_floating_point:
                    value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
                else:
                    value.copy_(source)

    def store(self, model: nn.Module) -> None:
        self._backup = copy.deepcopy(model.state_dict())

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.module.state_dict())

    def restore(self, model: nn.Module) -> None:
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None

    def state_dict(self) -> dict:
        return self.module.state_dict()
