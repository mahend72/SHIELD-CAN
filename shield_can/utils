# shield_can/utils.py
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class EWMAStat:
    half_life_s: float
    sigma_min: float = 0.05

    mean: float = 0.0
    second_moment: float = 0.0
    sigma: float = 0.0
    last_t: Optional[float] = None
    initialised: bool = False

    def update(self, t: float, x: float) -> float:
        """
        Time-based EMA + variance.
        Returns |x - mean| / sigma_eff (z-score-like).
        """
        if not self.initialised:
            self.mean = x
            self.second_moment = x * x
            self.sigma = 0.0
            self.last_t = t
            self.initialised = True
            return 0.0

        dt = max(t - self.last_t, 1e-6)
        tau = self.half_life_s / math.log(2.0)
        alpha = 1.0 - math.exp(-dt / tau)

        self.last_t = t

        self.mean = (1.0 - alpha) * self.mean + alpha * x
        self.second_moment = (1.0 - alpha) * self.second_moment + alpha * (x * x)

        var = max(self.second_moment - self.mean * self.mean, 0.0)
        self.sigma = math.sqrt(var)
        sigma_eff = max(self.sigma, self.sigma_min)

        z = abs(x - self.mean) / sigma_eff
        return z
