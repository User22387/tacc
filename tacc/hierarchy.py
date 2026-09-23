from __future__ import annotations

import heapq
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn.functional as F

from .carrier import ProgressiveTreeCarrier


@dataclass(frozen=True)
class TreeGeometry:
    y0: torch.Tensor
    y1: torch.Tensor
    x0: torch.Tensor
    x1: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    anchor_index: torch.Tensor
    mass: torch.Tensor

    @property
    def node_count(self) -> int:
        return int(self.mass.numel())


@lru_cache(maxsize=4)
def tree_geometry_cpu(grid_size: int) -> TreeGeometry:
    tokens_per_view = int(grid_size) ** 2
    nodes: list[dict] = []

    def build(y0: int, y1: int, x0: int, x1: int) -> int:
        node_id = len(nodes)
        nodes.append({})
        height, width = y1 - y0, x1 - x0
        left = right = -1
        if height * width > 1:
            if height >= width and height > 1:
                midpoint = y0 + height // 2
                left = build(y0, midpoint, x0, x1)
                right = build(midpoint, y1, x0, x1)
            else:
                midpoint = x0 + width // 2
                left = build(y0, y1, x0, midpoint)
                right = build(y0, y1, midpoint, x1)
        anchor_y = (y0 + y1 - 1) // 2
        anchor_x = (x0 + x1 - 1) // 2
        nodes[node_id] = {
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
            "left": left,
            "right": right,
            "anchor": anchor_y * grid_size + anchor_x,
            "mass": height * width,
        }
        return node_id

    if build(0, grid_size, 0, grid_size) != 0 or len(nodes) != 2 * tokens_per_view - 1:
        raise RuntimeError("Invalid progressive-tree geometry")

    def tensor(key: str) -> torch.Tensor:
        return torch.tensor([node[key] for node in nodes], dtype=torch.long)

    return TreeGeometry(
        y0=tensor("y0"),
        y1=tensor("y1"),
        x0=tensor("x0"),
        x1=tensor("x1"),
        left=tensor("left"),
        right=tensor("right"),
        anchor_index=tensor("anchor"),
        mass=tensor("mass"),
    )


def _integral_image(values: torch.Tensor) -> torch.Tensor:
    prefix = values.cumsum(dim=1).cumsum(dim=2)
    return F.pad(prefix, (0, 0, 1, 0, 1, 0))


def _rect_sum(prefix: torch.Tensor, geometry: TreeGeometry) -> torch.Tensor:
    y0 = geometry.y0.to(prefix.device)
    y1 = geometry.y1.to(prefix.device)
    x0 = geometry.x0.to(prefix.device)
    x1 = geometry.x1.to(prefix.device)
    return (
        prefix[:, y1, x1]
        - prefix[:, y0, x1]
        - prefix[:, y1, x0]
        + prefix[:, y0, x0]
    )


@dataclass
class ProgressiveTokenTree:
    view_tokens: torch.Tensor
    node_means: torch.Tensor
    node_variance: torch.Tensor
    split_order: torch.Tensor
    geometry: TreeGeometry
    grid_size: int
    node_carriers: torch.Tensor | None = None
    frontier_node_order: torch.Tensor | None = None
    frontier_birth: torch.Tensor | None = None
    frontier_death: torch.Tensor | None = None

    @property
    def num_views(self) -> int:
        return int(self.view_tokens.shape[0])

    @property
    def tokens_per_view(self) -> int:
        return int(self.view_tokens.shape[1])

    @torch.no_grad()
    def precompute_frontier_tables(self) -> None:
        terminal_quota = self.tokens_per_view + 1
        births = torch.full(
            (self.num_views, self.geometry.node_count),
            terminal_quota,
            dtype=torch.int16,
        )
        deaths = torch.full_like(births, terminal_quota)
        births[:, 0] = 1
        for view_index in range(self.num_views):
            for step, node_tensor in enumerate(self.split_order[view_index].cpu()):
                node = int(node_tensor)
                next_quota = step + 2
                left = int(self.geometry.left[node])
                right = int(self.geometry.right[node])
                deaths[view_index, node] = next_quota
                births[view_index, left] = next_quota
                births[view_index, right] = next_quota

        order = torch.argsort(self.geometry.anchor_index, stable=True)
        device = self.view_tokens.device
        self.frontier_node_order = order.to(device)
        self.frontier_birth = births.index_select(1, order).to(device)
        self.frontier_death = deaths.index_select(1, order).to(device)

    @torch.no_grad()
    def precompute_carriers(
        self, carrier: ProgressiveTreeCarrier, chunk_size: int = 4096
    ) -> None:
        geometry = self.geometry
        device = self.view_tokens.device
        node_ids = torch.arange(geometry.node_count, device=device)
        anchor_indices = geometry.anchor_index.to(device)
        masses = geometry.mass.to(device).float()
        heights = (geometry.y1 - geometry.y0).to(device).float()
        widths = (geometry.x1 - geometry.x0).to(device).float()
        per_view = []
        for view_index in range(self.num_views):
            chunks = []
            for start in range(0, geometry.node_count, int(chunk_size)):
                selected = node_ids[start : start + int(chunk_size)]
                anchors = self.view_tokens[view_index].index_select(
                    0, anchor_indices.index_select(0, selected)
                )
                chunks.append(
                    carrier(
                        anchors,
                        self.node_means[view_index].index_select(0, selected),
                        self.node_variance[view_index].index_select(0, selected),
                        masses.index_select(0, selected),
                        heights.index_select(0, selected),
                        widths.index_select(0, selected),
                    )
                )
            per_view.append(torch.cat(chunks, dim=0))
        self.node_carriers = torch.stack(per_view).detach()

    @torch.no_grad()
    def gather(self, quotas: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if quotas.ndim != 3 or quotas.shape[0] != 1 or quotas.shape[2] != self.num_views:
            raise ValueError(
                f"Expected quotas [1, target, {self.num_views}], got {tuple(quotas.shape)}"
            )
        if self.node_carriers is None:
            raise RuntimeError("Call precompute_carriers before target-time readout")
        if bool((quotas < 0).any()) or bool((quotas > self.tokens_per_view).any()):
            raise ValueError("Quota is outside the per-view capacity")
        totals = quotas.sum(dim=-1)
        if not bool((totals == totals[:, :1]).all()):
            raise ValueError("Every target must have the same total budget")
        if self.frontier_node_order is None:
            self.precompute_frontier_tables()

        quota_rows = quotas[0].to(self.view_tokens.device, dtype=self.frontier_birth.dtype)
        active = (self.frontier_birth.unsqueeze(0) <= quota_rows.unsqueeze(-1)) & (
            quota_rows.unsqueeze(-1) < self.frontier_death.unsqueeze(0)
        )
        if not bool((active.sum(dim=-1) == quota_rows).all()):
            raise RuntimeError("Vectorized frontier does not match requested quotas")

        budget = int(totals[0, 0])
        masses = self.geometry.mass.to(self.view_tokens.device)
        output = []
        selected_masses = []
        for target_index in range(quotas.shape[1]):
            selected = torch.nonzero(active[target_index], as_tuple=False)
            view_indices = selected[:, 0]
            node_indices = self.frontier_node_order.index_select(0, selected[:, 1])
            row = self.node_carriers[view_indices, node_indices]
            if row.shape[0] != budget:
                raise RuntimeError(f"Expected {budget} tokens, got {row.shape[0]}")
            output.append(row)
            selected_masses.append(masses.index_select(0, node_indices).float())

        all_masses = torch.cat(selected_masses)
        return torch.stack(output), {
            "tokens_after": budget,
            "carrier_non_leaf_count": int((all_masses > 1).sum().item()),
            "carrier_mass_mean": float(all_masses.mean().item()),
            "carrier_mass_max": float(all_masses.max().item()),
        }


@torch.no_grad()
def build_progressive_token_tree(
    source_tokens: torch.Tensor, grid_size: int = 37
) -> ProgressiveTokenTree:
    """Build the source-only hierarchy for one fixed source set."""
    if source_tokens.ndim == 4:
        if source_tokens.shape[0] != 1:
            raise ValueError("TACC source-cache construction currently expects batch size 1")
        _, num_views, tokens_per_view, channels = source_tokens.shape
        flat = source_tokens.reshape(1, num_views * tokens_per_view, channels)
    elif source_tokens.ndim == 3:
        if source_tokens.shape[0] != 1:
            raise ValueError("TACC source-cache construction currently expects batch size 1")
        tokens_per_view = int(grid_size) ** 2
        if source_tokens.shape[1] % tokens_per_view:
            raise ValueError("Flat source-token count is not divisible by grid_size**2")
        num_views = source_tokens.shape[1] // tokens_per_view
        flat = source_tokens
    else:
        raise ValueError("source_tokens must be [1,V,P,C] or [1,V*P,C]")

    expected = int(grid_size) ** 2
    if tokens_per_view != expected:
        raise ValueError(f"Expected {expected} tokens per view, got {tokens_per_view}")
    geometry = tree_geometry_cpu(int(grid_size))
    views = flat[0].reshape(num_views, grid_size, grid_size, -1)
    values = views.float()
    vector_prefix = _integral_image(values)
    scalar_prefix = _integral_image(values.square().sum(dim=-1, keepdim=True))
    region_sum = _rect_sum(vector_prefix, geometry)
    region_square_sum = _rect_sum(scalar_prefix, geometry).squeeze(-1)
    mass = geometry.mass.to(values.device).float()
    means = region_sum / mass[None, :, None]
    distortion = (
        region_square_sum - region_sum.square().sum(dim=-1) / mass[None, :]
    ).clamp_min(0.0)
    variance = distortion / (mass[None, :] * values.shape[-1]).clamp_min(1.0)

    left, right = geometry.left, geometry.right
    internal = torch.nonzero(left >= 0, as_tuple=False).flatten()
    gains = torch.zeros_like(distortion)
    internal_device = internal.to(values.device)
    gains[:, internal_device] = (
        distortion[:, internal_device]
        - distortion[:, left[internal].to(values.device)]
        - distortion[:, right[internal].to(values.device)]
    ).clamp_min(0.0)
    gains_cpu = gains.cpu()

    split_orders = []
    for view_index in range(num_views):
        heap = [(-int(geometry.mass[0]), -float(gains_cpu[view_index, 0]), 0)]
        order = []
        while heap:
            _negative_mass, _negative_gain, node = heapq.heappop(heap)
            if int(left[node]) < 0:
                continue
            order.append(node)
            for child in (int(left[node]), int(right[node])):
                if int(left[child]) >= 0:
                    heapq.heappush(
                        heap,
                        (
                            -int(geometry.mass[child]),
                            -float(gains_cpu[view_index, child]),
                            child,
                        ),
                    )
        if len(order) != tokens_per_view - 1:
            raise RuntimeError("Progressive split order is incomplete")
        split_orders.append(order)

    return ProgressiveTokenTree(
        view_tokens=views.reshape(num_views, tokens_per_view, -1).detach(),
        node_means=means.to(flat.dtype).detach(),
        node_variance=variance.detach(),
        split_order=torch.tensor(split_orders, dtype=torch.long),
        geometry=geometry,
        grid_size=int(grid_size),
    )
