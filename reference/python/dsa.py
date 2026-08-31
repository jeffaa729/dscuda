"""DSA lightning indexer plus sparse MLA on original (uncompressed) KV tokens.

Indexer q/k are already projected and position encoded. weights include the model's
head/dimension scaling. No FP8 simulation: this is an unquantized math oracle.
"""

import torch

from .attention import compute, sparse_attention


def index_scores(q_index, k_index, weights, visible):
    """[B,Q,Hi,Di] x [B,N,Di] -> ReLU -> weighted head sum -> [B,Q,N]."""
    dots = torch.einsum("bqhd,bnd->bqhn", compute(q_index), compute(k_index))
    scores = (dots.relu() * compute(weights)[..., None]).sum(2)
    return scores.masked_fill(~visible, -torch.inf)


def select_indices(scores, topk):
    """Select at most topk entries, padding causally unavailable slots with -1.

    Ties use PyTorch topk's unspecified order; correctness compares score/set
    equivalence for ties rather than assuming official kernels break ties identically.
    """
    values, ids = scores.topk(min(topk, scores.shape[-1]), dim=-1)
    return ids.masked_fill(torch.isneginf(values), -1)


def dsa_forward(
    q, kv, q_index, k_index, weights, topk, *, value_dim=512, start_pos=0, scale=None
):
    positions = torch.arange(start_pos, start_pos + q.shape[1], device=q.device)
    visible = torch.arange(kv.shape[1], device=q.device) <= positions[:, None]
    ids = select_indices(index_scores(q_index, k_index, weights, visible), topk)
    output, lse = sparse_attention(q, kv, kv[..., :value_dim], ids, scale)
    return output, lse, ids
