from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TACCConfig:
    """Inference configuration used by the LagerNVS TACC checkpoint."""

    grid_size: int = 37
    token_dim: int = 768
    hidden_dim: int = 192
    moment_distance_weight: float = 0.1
    residual_temperature: float = 0.8

    @property
    def tokens_per_view(self) -> int:
        return self.grid_size * self.grid_size

    def budget(self, equivalent_views: int) -> int:
        if equivalent_views < 1:
            raise ValueError("equivalent_views must be positive")
        return int(equivalent_views) * self.tokens_per_view

    def to_dict(self) -> dict:
        return asdict(self)
