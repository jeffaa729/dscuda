"""Sparse-attention primitives used by the DSA mathematical reference.

BF16 inputs are upcast for FP32 arithmetic; FP64 stays FP64 for gradcheck.
"""

import torch


def compute(x):
    return x if x.dtype == torch.float64 else x.float()


def probabilities(scores, sink=None):
    """Normalize once over all entries; an optional per-head sink has zero value.

    scores is [B,Q,H,L]. Returned LSE includes the sink. Empty/masked rows
    produce zero probabilities and -inf LSE, with zero rather than NaN gradients.
    """
    width = scores.shape[-1]
    if sink is not None:
        extra = compute(sink)[None, None, :, None].expand(*scores.shape[:-1], 1)
        scores = torch.cat((scores, extra), dim=-1)
    if scores.shape[-1] == 0:
        return scores, scores.new_full(scores.shape[:-1], -torch.inf)
    empty = torch.isneginf(scores).all(-1, keepdim=True)
    infinite_sink = torch.isposinf(scores).any(-1, keepdim=True)
    safe = torch.where(empty | infinite_sink, torch.zeros_like(scores), scores)
    p = safe.softmax(-1)
    p = torch.where(empty | infinite_sink, 0.0, p)[..., :width]
    lse = safe.logsumexp(-1)
    lse = torch.where(empty.squeeze(-1), -torch.inf, lse)
    lse = torch.where(infinite_sink.squeeze(-1), torch.inf, lse)
    return p, lse


def sparse_attention(q, k, v, indices, scale=None, sink=None):
    """Shared-KV attention: q[B,Q,H,D], k[B,N,D], v[B,N,V], indices[B,Q,L].

    -1 denotes padding. Callers supply unique valid indices and enforce causality.
    Gathering selected KV is intentional: this is the sparse-core boundary.
    """
    q, k, v = map(compute, (q, k, v))
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    valid = (indices >= 0) & (indices < k.shape[1])
    # A zero sentinel also makes N=0 and completely masked rows well defined.
    k = torch.cat((k, k.new_zeros(k.shape[0], 1, k.shape[-1])), dim=1)
    v = torch.cat((v, v.new_zeros(v.shape[0], 1, v.shape[-1])), dim=1)
    selected = torch.where(valid, indices.long(), k.shape[1] - 1)
    batch = torch.arange(q.shape[0], device=q.device)[:, None, None]
    keys, values = k[batch, selected], v[batch, selected]
    scores = torch.einsum("bqhd,bqld->bqhl", q, keys) * scale
    scores = scores.masked_fill(~valid[:, :, None], -torch.inf)
    p, lse = probabilities(scores, sink)
    return torch.einsum("bqhl,bqlv->bqhv", p, values), lse.transpose(1, 2)
