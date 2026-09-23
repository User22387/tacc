from __future__ import annotations

import torch
import torch.nn as nn


class ProgressiveTreeCarrier(nn.Module):
    """Encode one rectangular source-token region as one Carrier token."""

    def __init__(self, token_dim: int = 768, hidden_dim: int = 192):
        super().__init__()
        self.token_dim = int(token_dim)
        self.delta_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.delta_proj = nn.Linear(token_dim, hidden_dim)
        self.condition = nn.Sequential(
            nn.Linear(5, 32),
            nn.SiLU(),
            nn.Linear(32, hidden_dim),
        )
        self.hidden_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_dim, token_dim)
        self.gate = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_region(
        self,
        anchor: torch.Tensor,
        region_mean: torch.Tensor,
        region_variance: torch.Tensor,
        region_mass: torch.Tensor,
        region_height: torch.Tensor,
        region_width: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if anchor.shape != region_mean.shape or anchor.shape[-1] != self.token_dim:
            raise ValueError(
                f"Expected matching [..., {self.token_dim}] anchor/mean tensors, got "
                f"{tuple(anchor.shape)} and {tuple(region_mean.shape)}"
            )
        expected = anchor.shape[:-1]
        for name, value in (
            ("region_variance", region_variance),
            ("region_mass", region_mass),
            ("region_height", region_height),
            ("region_width", region_width),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")

        mass = region_mass.float().clamp_min(1.0)
        variance = region_variance.float().clamp_min(0.0)
        descriptor = torch.stack(
            [
                mass.log() / 8.0,
                variance.sqrt().clamp_max(10.0) / 10.0,
                region_height.float().clamp(1.0, 37.0) / 37.0,
                region_width.float().clamp(1.0, 37.0) / 37.0,
                (region_height.float() * region_width.float())
                .clamp_min(1.0)
                .log()
                / 8.0,
            ],
            dim=-1,
        )
        delta_input = self.delta_norm((region_mean - anchor).float())
        hidden = self.delta_proj(delta_input) + self.condition(descriptor)
        return self.activation(self.hidden_norm(hidden)), mass

    def forward(
        self,
        anchor: torch.Tensor,
        region_mean: torch.Tensor,
        region_variance: torch.Tensor,
        region_mass: torch.Tensor,
        region_height: torch.Tensor,
        region_width: torch.Tensor,
    ) -> torch.Tensor:
        encoded, mass = self.encode_region(
            anchor,
            region_mean,
            region_variance,
            region_mass,
            region_height,
            region_width,
        )
        residual = torch.sigmoid(self.gate) * self.output(encoded)
        non_leaf = (mass > 1.0).unsqueeze(-1).to(residual.dtype)
        return anchor + (non_leaf * residual).to(anchor.dtype)


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
