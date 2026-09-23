"""Minimal TACC insertion point for an already loaded LagerNVS model."""

from pathlib import Path

import torch

from tacc import LagerNVSTACC


@torch.no_grad()
def render_with_tacc(
    lagernvs_model,
    source_images: torch.Tensor,
    source_camera_tokens: torch.Tensor,
    source_plucker_rays: torch.Tensor,
    target_plucker_rays: torch.Tensor,
    checkpoint: str | Path,
) -> torch.Tensor:
    device = source_images.device
    tacc = LagerNVSTACC.from_checkpoint(checkpoint, device=device)

    # LagerNVS source encoding is executed once for a fixed source set.
    source_tokens = lagernvs_model.reconstructor(
        source_images, source_camera_tokens
    )  # [1, V, 1369, 768]

    # TACC source-side hierarchy and all Carrier nodes are also cached once.
    source_cache = tacc.build_source_cache(source_tokens)

    # Allocation and frontier readout are target-conditioned. Target rays keep
    # the original LagerNVS path and are passed unchanged to the renderer.
    return tacc.render_targets(
        lagernvs_model.renderer,
        source_cache,
        source_plucker_rays,
        target_plucker_rays,
        equivalent_views=10,
        target_batch_size=4,
    )
