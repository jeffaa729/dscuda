"""Optional DeepGEMM BF16 grouped-GEMM reference, using its masked layout.

This is a separate BF16-output library contract, not the native FP32-output
cuBLAS comparison. Valid rows are compared; padding is not part of the output.
See upstream tests/test_bf16.py and tests/generators.py for the masked ABI.
"""

import torch


def masked_grouped_reference(x, weights_nt, counts):
    """PyTorch oracle: [E,M,K] @ [E,N,K]^T -> [E,M,N], BF16 output."""
    from .attention import compute

    out = (compute(x) @ compute(weights_nt).transpose(-1, -2)).to(x.dtype)
    valid = torch.arange(x.shape[1], device=x.device)[None] < counts[:, None]
    return out.masked_fill(~valid[..., None], 0)


class DeepGEMMGrouped:
    def __init__(self, x, weights_nt, counts, expected_m):
        if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] not in (
            9,
            10,
        ):
            raise RuntimeError(
                "DeepGEMM requires SM90/SM100 and an upstream installation"
            )
        if x.dtype != torch.bfloat16 or weights_nt.dtype != torch.bfloat16:
            raise ValueError("BF16 inputs are required")
        import deep_gemm

        self.call = deep_gemm.m_grouped_bf16_gemm_nt_masked
        self.x, self.w = x.contiguous(), weights_nt.contiguous()
        self.counts = counts.to(device=x.device, dtype=torch.int32).contiguous()
        self.out = torch.zeros(
            (*x.shape[:2], weights_nt.shape[1]), device=x.device, dtype=x.dtype
        )
        self.expected_m = expected_m

    def __call__(self):
        self.call(self.x, self.w, self.out, self.counts, self.expected_m)
        return self.out
