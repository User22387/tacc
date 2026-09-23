from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _center_rays(plucker_rays: torch.Tensor) -> torch.Tensor:
    if plucker_rays.ndim != 5 or plucker_rays.shape[2] != 6:
        raise ValueError(
            "Expected Plucker rays [batch, view, 6, height, width], got "
            f"{tuple(plucker_rays.shape)}"
        )
    height, width = plucker_rays.shape[-2:]
    return plucker_rays[:, :, :, height // 2, width // 2].float()


def _minmax(values: torch.Tensor, eps: float) -> torch.Tensor:
    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    return (values - minimum) / (maximum - minimum + eps)


def compute_relevance(
    source_rays: torch.Tensor,
    target_rays: torch.Tensor,
    *,
    moment_distance_weight: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute target-to-source relevance from central Plucker rays.

    The first three channels are Plucker moments and the final three channels
    are ray directions, matching LagerNVS camera conditioning.
    """
    source = _center_rays(source_rays)
    target = _center_rays(target_rays)
    if source.shape[0] != target.shape[0]:
        raise ValueError("Source and target ray batches must match")

    source_direction = F.normalize(source[:, :, 3:6], dim=-1, eps=eps)
    target_direction = F.normalize(target[:, :, 3:6], dim=-1, eps=eps)
    direction_similarity = torch.einsum(
        "bvc,btc->btv", source_direction, target_direction
    )
    moment_distance = torch.linalg.norm(
        source[:, None, :, 0:3] - target[:, :, None, 0:3], dim=-1
    )
    normalized_distance = _minmax(moment_distance, eps)
    return direction_similarity - float(moment_distance_weight) * normalized_distance


def _integerize_residual(
    soft_quota: torch.Tensor,
    scores: torch.Tensor,
    target_total: int,
    max_quota: int,
    eps: float,
) -> torch.Tensor:
    """Match the deterministic capacity-aware rounding used in the experiments."""
    soft_quota = torch.nan_to_num(soft_quota.float(), nan=0.0)
    quotas = torch.floor(soft_quota).long().clamp(0, int(max_quota))
    remaining = int(target_total - int(quotas.sum().item()))
    fractional = soft_quota - torch.floor(soft_quota)
    priority = fractional + 1e-6 * _minmax(scores.view(1, -1), 1e-6).view(-1)

    if remaining > 0:
        for index in torch.argsort(priority, descending=True, stable=True):
            if remaining == 0:
                break
            capacity = int(max_quota) - int(quotas[index].item())
            if capacity <= 0:
                continue
            addition = min(remaining, capacity)
            quotas[index] += addition
            remaining -= addition
    elif remaining < 0:
        for index in torch.argsort(priority, descending=False, stable=True):
            if remaining == 0:
                break
            capacity = int(quotas[index].item())
            if capacity <= 0:
                continue
            subtraction = min(-remaining, capacity)
            quotas[index] -= subtraction
            remaining += subtraction

    if remaining != 0:
        raise RuntimeError(f"Could not integerize quota; remaining={remaining}")
    return quotas


@torch.no_grad()
def allocate_quotas(
    relevance: torch.Tensor,
    budget: int,
    tokens_per_view: int,
    *,
    residual_temperature: float = 0.8,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Allocate an exact decoder-visible budget for each target."""
    if relevance.ndim != 3:
        raise ValueError(f"Expected relevance [batch, target, view], got {relevance.shape}")
    batch_size, target_count, view_count = relevance.shape
    if not 0 <= budget <= view_count * tokens_per_view:
        raise ValueError("Budget exceeds source-token capacity")
    if residual_temperature <= 0:
        raise ValueError("residual_temperature must be positive")

    quotas = torch.zeros_like(relevance, dtype=torch.long)
    entropy_values = []
    floor_values = []
    for batch_index in range(batch_size):
        for target_index in range(target_count):
            scores = relevance[batch_index, target_index]
            order = scores.argsort(descending=True)
            if budget <= tokens_per_view or view_count == 1:
                quotas[batch_index, target_index, order[0]] = budget
                entropy_values.append(0.0)
                floor_values.append(0.0)
                continue

            quotas[batch_index, target_index, order[0]] = tokens_per_view
            remaining_budget = int(budget - tokens_per_view)
            support = order[1:]
            support_scores = scores[support]

            if budget < 2 * tokens_per_view:
                quotas[batch_index, target_index, support[0]] = remaining_budget
                entropy_values.append(0.0)
                floor_values.append(0.0)
                continue

            standard_deviation = support_scores.std(unbiased=False)
            if float(standard_deviation.item()) <= eps:
                standardized = torch.zeros_like(support_scores)
            else:
                standardized = (
                    support_scores - support_scores.mean()
                ) / standard_deviation.clamp_min(eps)

            probability = torch.softmax(standardized, dim=0)
            entropy = -(probability * probability.clamp_min(eps).log()).sum()
            entropy = (entropy / math.log(max(view_count - 1, 2))).clamp(0.0, 1.0)
            floor_quota = int(
                math.floor(float(entropy.item()) * remaining_budget / (view_count - 1))
            )
            floor_quota = max(0, min(tokens_per_view, floor_quota))
            quotas[batch_index, target_index, support] = floor_quota

            residual_budget = remaining_budget - floor_quota * (view_count - 1)
            if residual_budget > 0:
                weights = torch.softmax(
                    standardized / float(residual_temperature), dim=0
                )
                additions = _integerize_residual(
                    weights * float(residual_budget),
                    support_scores,
                    residual_budget,
                    tokens_per_view - floor_quota,
                    eps,
                )
                quotas[batch_index, target_index, support] += additions

            entropy_values.append(float(entropy.item()))
            floor_values.append(float(floor_quota))

    if not bool((quotas.sum(dim=-1) == int(budget)).all()):
        raise RuntimeError("TACC allocation violated the exact budget")
    if bool((quotas < 0).any()) or bool((quotas > tokens_per_view).any()):
        raise RuntimeError("TACC allocation violated a per-view capacity")
    return quotas, {
        "entropy_mean": sum(entropy_values) / max(len(entropy_values), 1),
        "floor_quota_mean": sum(floor_values) / max(len(floor_values), 1),
    }


@torch.no_grad()
def target_aware_allocation(
    source_rays: torch.Tensor,
    target_rays: torch.Tensor,
    budget: int,
    tokens_per_view: int,
    *,
    moment_distance_weight: float = 0.1,
    residual_temperature: float = 0.8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    relevance = compute_relevance(
        source_rays,
        target_rays,
        moment_distance_weight=moment_distance_weight,
    )
    quotas, statistics = allocate_quotas(
        relevance,
        budget,
        tokens_per_view,
        residual_temperature=residual_temperature,
    )
    return quotas, relevance, statistics
