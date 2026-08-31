"""HCA core: one joint softmax over a local window, compressed history and sink.

q/local_kv/compressed_kv have already received normalization and partial RoPE.
K=V is shared across heads. Projection weights, quantization and output projection
are outside this boundary; inverse output RoPE can be supplied via output_angles.
"""

import torch

from .attention import rotary, sparse_attention
from .compression import compressed_indices, local_indices


def hca_forward(
    q,
    local_kv,
    compressed_kv,
    *,
    ratio=128,
    window=128,
    start_pos=0,
    sink=None,
    scale=None,
    output_angles=None,
):
    positions = torch.arange(start_pos, start_pos + q.shape[1], device=q.device)
    local = local_indices(positions, window)
    compressed = compressed_indices(positions, compressed_kv.shape[1], ratio)
    compressed = torch.where(compressed >= 0, compressed + local_kv.shape[1], -1)
    ids = torch.cat((local, compressed), -1)[None].expand(q.shape[0], -1, -1)
    cache = torch.cat((local_kv, compressed_kv), 1)
    out, lse = sparse_attention(q, cache, cache, ids, scale, sink)
    if output_angles is not None:
        out = rotary(out, output_angles, inverse=True)
    return out, lse
