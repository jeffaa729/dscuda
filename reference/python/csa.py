"""CSA core: select compressed entries, union with local tokens, normalize jointly.

The indexer's compressed keys are SEPARATE from the main attention compressed KV.
Prepare each with its own learned compressor (ratio=4, overlap=True). Hadamard
rotation can be omitted in this unquantized reference if applied to neither q nor k:
the common orthogonal transform preserves their dot product.
"""

import torch

from .attention import rotary, sparse_attention
from .compression import compressed_indices, local_indices
from .dsa import index_scores, select_indices


def csa_forward(
    q,
    local_kv,
    compressed_kv,
    q_index,
    compressed_index_keys,
    weights,
    topk,
    *,
    ratio=4,
    window=128,
    start_pos=0,
    sink=None,
    scale=None,
    output_angles=None,
):
    positions = torch.arange(start_pos, start_pos + q.shape[1], device=q.device)
    valid = compressed_indices(positions, compressed_kv.shape[1], ratio) >= 0
    selected = select_indices(
        index_scores(q_index, compressed_index_keys, weights, valid), topk
    )
    shifted = torch.where(selected >= 0, selected + local_kv.shape[1], -1)
    local = local_indices(positions, window)[None].expand(q.shape[0], -1, -1)
    ids = torch.cat((local, shifted), -1)
    cache = torch.cat((local_kv, compressed_kv), 1)
    output, lse = sparse_attention(q, cache, cache, ids, scale, sink)
    if output_angles is not None:
        output = rotary(output, output_angles, inverse=True)
    return output, lse, selected
