"""Small forward-only MoE oracle and grouped-GEMM benchmark reference."""

import torch
import torch.nn.functional as F

from .attention import compute


def grouped_gemm(x, weights, offsets):
    """Reference for packed [M,K] @ [E,K,N] weights in expert order."""
    if isinstance(offsets, torch.Tensor):
        raise TypeError("prepare offsets as a host tuple outside the measured call")
    x, weights = compute(x), compute(weights)
    return torch.cat(
        [x[begin:end] @ weights[e]
         for e, (begin, end) in enumerate(zip(offsets[:-1], offsets[1:]))],
        dim=0,
    )


@torch.no_grad()
def moe_forward(x, expert_ids, route_weights, w1, w2, clamp=10.0):
    """BF16 routed experts; IDs are global and W1/W2 use [E,N,K].

    Only the small correctness case uses this per-expert PyTorch loop. GEMM,
    SwiGLU, and the second GEMM each round to BF16 like the unfused pipeline.
    """
    tokens, hidden = x.shape
    output = torch.zeros(tokens, hidden, device=x.device, dtype=torch.float32)
    intermediate = w2.shape[-1]
    for expert in range(w1.shape[0]):
        token, choice = torch.where(expert_ids == expert)
        if token.numel() == 0:
            continue
        gate_up = (x[token].float() @ w1[expert].float().T).bfloat16()
        gate = gate_up[:, :intermediate].float().clamp(max=clamp)
        up = gate_up[:, intermediate:].float().clamp(-clamp, clamp)
        activation = (F.silu(gate) * up * route_weights[token, choice, None]).bfloat16()
        expert_output = (activation.float() @ w2[expert].float().T).bfloat16()
        output.index_add_(0, token, expert_output.float())
    return output.bfloat16()
