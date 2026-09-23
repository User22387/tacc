from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn

from .allocation import target_aware_allocation
from .carrier import ProgressiveTreeCarrier
from .config import TACCConfig
from .hierarchy import ProgressiveTokenTree, build_progressive_token_tree


@dataclass
class TACCSourceCache:
    hierarchy: ProgressiveTokenTree

    @property
    def num_views(self) -> int:
        return self.hierarchy.num_views


class LagerNVSTACC(nn.Module):
    """TACC inference adapter for a frozen LagerNVS model."""

    def __init__(self, config: TACCConfig | None = None):
        super().__init__()
        self.config = config or TACCConfig()
        self.carrier = ProgressiveTreeCarrier(
            token_dim=self.config.token_dim,
            hidden_dim=self.config.hidden_dim,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: torch.device | str = "cpu",
    ) -> "LagerNVSTACC":
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = TACCConfig(**state.get("config", {}))
        module = cls(config)
        module.carrier.load_state_dict(state["model_state_dict"], strict=True)
        module.to(device).eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return module

    @torch.no_grad()
    def build_source_cache(self, source_tokens: torch.Tensor) -> TACCSourceCache:
        hierarchy = build_progressive_token_tree(
            source_tokens, grid_size=self.config.grid_size
        )
        hierarchy.precompute_frontier_tables()
        hierarchy.precompute_carriers(self.carrier)
        return TACCSourceCache(hierarchy=hierarchy)

    @torch.no_grad()
    def compress(
        self,
        cache: TACCSourceCache,
        source_rays: torch.Tensor,
        target_rays: torch.Tensor,
        *,
        equivalent_views: int = 10,
    ) -> tuple[torch.Tensor, dict]:
        if source_rays.shape[1] != cache.num_views:
            raise ValueError("source_rays and cached source-token views do not match")
        budget = self.config.budget(equivalent_views)
        if budget > cache.num_views * self.config.tokens_per_view:
            raise ValueError("Requested budget exceeds the available source tokens")
        quotas, relevance, allocation_statistics = target_aware_allocation(
            source_rays,
            target_rays,
            budget,
            self.config.tokens_per_view,
            moment_distance_weight=self.config.moment_distance_weight,
            residual_temperature=self.config.residual_temperature,
        )
        compressed_tokens, readout_statistics = cache.hierarchy.gather(quotas)
        return compressed_tokens, {
            "quotas": quotas,
            "relevance": relevance,
            "allocation": allocation_statistics,
            "readout": readout_statistics,
        }

    @torch.no_grad()
    def render_targets(
        self,
        renderer: nn.Module,
        cache: TACCSourceCache,
        source_rays: torch.Tensor,
        target_rays: torch.Tensor,
        *,
        equivalent_views: int = 10,
        target_batch_size: int = 4,
    ) -> torch.Tensor:
        predictions = []
        for start in range(0, target_rays.shape[1], int(target_batch_size)):
            rays = target_rays[:, start : start + int(target_batch_size)]
            tokens, _metadata = self.compress(
                cache,
                source_rays,
                rays,
                equivalent_views=equivalent_views,
            )
            predictions.append(renderer(tokens, rays, timeit=False))
        return torch.cat(predictions, dim=1)
