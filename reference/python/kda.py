"""Kimi Delta Attention's recurrent mathematical oracle and FLA chunk adapter.

q/k[B,T,H,K], v[B,T,HV,V], log_decay[B,T,HV,K], beta[B,T,HV]. q/k normalization
and gate activation are outside both calls. State is [B,HV,K,V], FP32 (FP64 tests).
"""

import torch
from fla.ops.kda import chunk_kda

from .attention import compute


def kda_forward(q, k, v, log_decay, beta, initial_state=None, scale=None):
    dtype = v.dtype
    q, k, v, log_decay, beta = map(compute, (q, k, v, log_decay, beta))
    batch, tokens, heads, width = q.shape
    value_heads, value_width = v.shape[-2:]
    if value_heads % heads:
        raise ValueError("value heads must be a multiple of query/key heads")
    q = q.repeat_interleave(value_heads // heads, 2)
    k = k.repeat_interleave(value_heads // heads, 2)
    scale = width**-0.5 if scale is None else scale
    state = (
        q.new_zeros(batch, value_heads, width, value_width)
        if initial_state is None
        else compute(initial_state)
    )
    outputs = []
    for t in range(tokens):
        decayed = state * log_decay[:, t].exp()[..., None]
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, t], decayed)
        innovation = (v[:, t] - prediction) * beta[:, t, :, None]
        state = decayed + k[:, t, :, :, None] * innovation[..., None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, state))
    output = torch.stack(outputs, 1) if outputs else v[:, :0]
    return output.to(dtype), state


def fla_forward(q, k, v, log_decay, beta, initial_state=None, scale=None):
    """Official FLA chunk_kda with exactly the oracle's prepared-input contract.

    Prewarm before graph capture; outputs and final state participate in FLA's
    analytical backward.
    """
    return chunk_kda(
        q,
        k,
        v,
        log_decay,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
        use_beta_sigmoid_in_kernel=False,
    )
