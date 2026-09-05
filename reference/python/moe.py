"""Unquantized DeepSeek-style routing, dispatch, grouped GEMM and weighted combine.

Expert weights use [E,K,N]; packed tokens use [sum(M_e),K]. No capacity drops.
Routing bias affects selection only. Hard top-k is not differentiable in its indices.
"""

import torch

from .attention import compute


def route(logits, bias, topk, scale=1.0, groups=1, selected_groups=None):
    """Sigmoid routing, optionally group-limited by the sum of each group's top two.

    Stable sorting resolves equal expert scores by the lower expert index.
    Returned weights normalize the unbiased selected sigmoid scores.
    """
    affinity = compute(logits).sigmoid()
    scores = affinity + compute(bias)
    if groups > 1:
        per_group = scores.shape[-1] // groups
        if scores.shape[-1] % groups or per_group < 2:
            raise ValueError(
                "groups must divide experts, with at least two experts per group"
            )
        selected_groups = groups if selected_groups is None else selected_groups
        if topk > selected_groups * per_group:
            raise ValueError("topk exceeds the selected groups capacity")
        grouped = scores.unflatten(-1, (groups, per_group))
        group_scores = grouped.topk(2, dim=-1).values.sum(-1)
        group_ids = group_scores.argsort(dim=-1, descending=True, stable=True)[
            ..., :selected_groups
        ]
        keep = torch.zeros_like(group_scores, dtype=torch.bool).scatter(
            -1, group_ids, True
        )
        scores = grouped.masked_fill(~keep[..., None], -torch.inf).flatten(-2)
    ids = scores.argsort(dim=-1, descending=True, stable=True)[..., :topk]
    weights = affinity.gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True) * scale
    return ids, weights


def dispatch(x, expert_ids, experts):
    """Return packed tokens, expert prefix offsets and route->packed-slot mapping."""
    flat = expert_ids.flatten()
    order = flat.argsort(stable=True)
    route_to_slot = torch.empty_like(order).scatter(
        0, order, torch.arange(order.numel(), device=x.device)
    )
    counts = torch.zeros(experts, device=flat.device, dtype=flat.dtype)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    packed = x[order // expert_ids.shape[-1]]
    return packed, offsets, route_to_slot.reshape_as(expert_ids)


def grouped_gemm(x, weights, offsets):
    """One FP32/FP64 matmul per expert, with autograd for both operands.

    offsets is a host tuple prepared outside timing; there is no CUDA->CPU sync
    in this operation. BF16 input conversion is inside this reference call.
    """
    if isinstance(offsets, torch.Tensor):
        raise TypeError("prepare offsets as a host tuple outside the measured call")
    x, weights = compute(x), compute(weights)
    return torch.cat(
        [
            x[begin:end] @ weights[e]
            for e, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:]))
        ],
        dim=0,
    )


def combine(packed, route_to_slot, weights, shared=None):
    out = (compute(packed)[route_to_slot] * compute(weights)[..., None]).sum(1)
    return out if shared is None else out + compute(shared)
