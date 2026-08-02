from __future__ import annotations

import numpy as np

from .base import (
    BaseFitter,
)


class TemplateAmplitudeFitter(BaseFitter):
    """
    Fits y_{i} = N * t_{i} by weighted least squares (t_{i} fixed, N free).

        N           = \sum(w_{i} t_{i} y_{i}) / \sum(w_{i} t_{i}^{2}),   w_{i} = 1/\sigma_{i}^{2}
        \sigma(N)   = 1 / \sqrt{\sum(w_{i} t_{i}^{2})}
    """
    n_params: int = 1

    def __init__(self, template: np.ndarray, bins: np.ndarray, y: np.ndarray, yerr: np.ndarray) -> None:
        super().__init__(bins, y, yerr)
        valid              = yerr > 0
        self.template      = np.asarray(template)[valid]
        self.template_full = template

    def model(self, x: np.ndarray, N: float) -> np.ndarray:
        return N * self.template
    
    def model_full(self, N: float) -> np.ndarray:
        return N * self.template_full
    
    def _initial_params(self) -> list[float]:
        N = 1
        return [N]