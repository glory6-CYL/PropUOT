from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

try:
    from geomloss import SamplesLoss
except ImportError:  # pragma: no cover - exercised by the dependency check
    SamplesLoss = None


def propensity_odds(
    propensity: torch.Tensor,
    gamma: float,
    eps: float = 1e-6,
    bound: float = 0.05,
    maximum: float = 10.0,
) -> torch.Tensor:
    """Stable relaxed likelihood ratio Ω=(π/(1-π+ε₀))^γ."""
    propensity = propensity.clamp(min=float(bound), max=1.0 - float(bound))
    odds = propensity / (1.0 - propensity + float(eps))
    return odds.pow(float(gamma)).clamp(max=float(maximum))


class PropensityLoss(nn.Module):
    def __init__(self, entropy_weight: float = 0.005, entropy_sign: float = -1.0):
        super().__init__()
        self.entropy_weight = float(entropy_weight)
        self.entropy_sign = float(entropy_sign)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float().reshape(-1).clamp(-50.0, 50.0)
        target = target.to(device=logits.device, dtype=torch.float32).reshape(-1)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        probability = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)
        entropy = -(
            probability * probability.log()
            + (1.0 - probability) * (1.0 - probability).log()
        ).mean()
        return bce + self.entropy_sign * self.entropy_weight * entropy


class UnbalancedOTLoss(nn.Module):
    def __init__(self, blur: float = 0.10, reach: float = 0.10) -> None:
        super().__init__()
        if SamplesLoss is None:
            raise ImportError(
                "PropUOT requires geomloss for UOT. Install the provided environment first."
            )
        self.blur = float(blur)
        self.reach = float(reach)

    @staticmethod
    def _diameter(source: torch.Tensor, target: torch.Tensor) -> float:
        joined_min = torch.minimum(source.amin(dim=0), target.amin(dim=0))
        joined_max = torch.maximum(source.amax(dim=0), target.amax(dim=0))
        value = float((joined_max - joined_min).norm().detach().cpu())
        return value if math.isfinite(value) else 0.0

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        target_present: torch.Tensor,
    ) -> torch.Tensor:
        target_present = target_present.to(device=target.device, dtype=torch.bool)
        observed_target = target[target_present]
        if source.shape[0] == 0 or observed_target.shape[0] == 0:
            return source.new_zeros(())

        with torch.no_grad():
            diameter = self._diameter(source, observed_target)
        if diameter <= 0.0:
            return source.new_zeros(())

        # Keep the Sinkhorn blur below the empirical diameter and above
        # machine scale so that GeomLoss can build a valid scale schedule.
        minimum_blur = max(diameter / (2.0**40), 1e-20)
        blur = max(min(self.blur, diameter * 0.49), minimum_blur)
        sinkhorn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=blur,
            reach=self.reach,
            scaling=0.5,
            debias=True,
        )
        value = sinkhorn(source, observed_target)
        return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
