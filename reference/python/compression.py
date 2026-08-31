"""V4 learned compression from already projected values/gates; no FP8/FP4 QAT.

HCA uses ratio=128 without overlap; CSA uses ratio=4 with overlap. Ratios remain
arguments so small CPU examples can verify exactly the same equations.
"""

import torch

from .attention import compute, rotary


def compress(
    values,
    gates,
    position_bias,
    ratio,
    norm_weight,
    *,
    overlap=False,
    epsilon=1e-6,
    angles=None,
    storage_dtype=None,
):
    """Pool completed blocks, RMS-normalize, then apply partial RoPE.

    values/gates: [B,T,D] (HCA) or [B,T,2D] (CSA); bias: [ratio,D or 2D].
    For CSA, previous block's first D and current block's last D share ONE
    per-channel softmax over 2*ratio positions. The first previous block is masked.
    angles contains per-token frequencies; a compressed block uses its START position.
    """
    batch, tokens, channels = values.shape
    blocks = tokens // ratio
    width = channels // (2 if overlap else 1)
    dtype = values.dtype if storage_dtype is None else storage_dtype
    if blocks == 0:
        return values[:, :0, :width].to(dtype)
    x = compute(values[:, : blocks * ratio]).reshape(batch, blocks, ratio, channels)
    g = compute(gates[:, : blocks * ratio]).reshape_as(x) + compute(position_bias)
    if overlap:
        previous_x = torch.cat(
            (torch.zeros_like(x[:, :1, :, :width]), x[:, :-1, :, :width]), 1
        )
        previous_g = torch.cat(
            (torch.full_like(g[:, :1, :, :width], -torch.inf), g[:, :-1, :, :width]), 1
        )
        x = torch.cat((previous_x, x[..., width:]), 2)
        g = torch.cat((previous_g, g[..., width:]), 2)
    pooled = (x * g.softmax(2)).sum(2).to(dtype)
    full = compute(pooled)
    pooled = (
        full
        * torch.rsqrt(full.square().mean(-1, keepdim=True) + epsilon)
        * compute(norm_weight)
    ).to(dtype)
    if angles is not None:
        pooled = rotary(pooled, angles[: blocks * ratio : ratio])
    return pooled


def local_indices(positions, window):
    ids = positions[:, None] - torch.arange(window - 1, -1, -1, device=positions.device)
    return ids.masked_fill(ids < 0, -1)


def compressed_indices(positions, blocks, ratio):
    ids = torch.arange(blocks, device=positions.device).expand(
        positions.numel(), blocks
    )
    return ids.masked_fill((ids + 1) * ratio > positions[:, None] + 1, -1)
