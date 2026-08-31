"""Packed expert NN GEMMs: PyTorch correctness and cuBLAS-loop timing."""
import ctypes

from common import I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.moe import grouped_gemm


def expert_counts(rows, experts, distribution):
    if distribution == "uniform":
        counts = [rows // experts] * experts
    elif distribution == "hot":
        counts = [rows * 4 // 5] + [rows // (5 * (experts - 1))] * (experts - 1)
    else:
        counts = [0, rows // 3, 0] + [(rows - rows // 3) // (experts - 3)] * (experts - 3)
    counts[-1] += rows - sum(counts)
    return counts


def cases(args, family):
    if args.reference not in (None, "cublas"):
        raise ValueError("Grouped GEMM uses cuBLAS as its only benchmark reference")
    lib = library("operator")
    checked(lib, "operator", bind(lib, "dscuda_cublas_init", [])())
    gemm = bind(lib, "dscuda_grouped_gemm", [P] * 6 + [I] * 6 + [P])
    backward = bind(lib, "dscuda_grouped_backward", [P] * 7 + [I] * 5 + [P])
    shapes = ((64, 5, 32, 48), (129, 8, 37, 49)) if args.test else (
        ((512, 8, 256, 512),) if args.suite == "quick" else
        ((4096, 8, 512, 1536), (8192, 16, 512, 1536)))
    try:
        for m, e, k, n in shapes:
            for distribution in ("uniform", "hot", "empty"):
                counts = expert_counts(m, e, distribution)
                offsets = [0]
                for count in counts:
                    offsets.append(offsets[-1] + count)
                host_offsets = (I * len(offsets))(*offsets)
                device_offsets = torch.tensor(offsets, device="cuda", dtype=torch.int32)
                slots = torch.tensor([expert for expert, count in enumerate(counts) for _ in range(count)],
                                     device="cuda", dtype=torch.int32)
                for dtype in (torch.float32, torch.bfloat16) if args.test else (torch.bfloat16,):
                    x = (torch.randn(m, k, device="cuda", dtype=dtype) * .1).requires_grad_()
                    w = (torch.randn(e, k, n, device="cuda", dtype=dtype) * .1).requires_grad_()
                    outputs = [torch.empty(m, n, device="cuda") for _ in range(2)]

                    def pytorch():
                        return grouped_gemm(x, w, tuple(offsets))

                    expected = pytorch()

                    def call(ref=0):
                        checked(lib, "operator", gemm(*pointers((outputs[ref], x, w, device_offsets, slots)),
                                ctypes.cast(host_offsets, P), m, e, n, k,
                                int(dtype == torch.bfloat16), ref, stream()))
                        return outputs[ref]

                    size = f"M={m},N={n},K={k},E={e},{distribution}"
                    yield Operation(size, "bf16" if dtype == torch.bfloat16 else "fp32", "NN",
                                    {"custom": call, "cuBLAS loop": lambda: call(1)},
                                    (expected.detach(),))
                    if args.test and dtype == torch.float32:
                        dy = torch.randn_like(expected)
                        dx, dw = torch.empty_like(x), torch.empty_like(w)
                        def pytorch_backward():
                            return torch.autograd.grad(expected, (x, w), dy, retain_graph=True)
                        expected_gradients = pytorch_backward()
                        def custom_backward():
                            dx.zero_()
                            dw.zero_()
                            checked(lib, "operator", backward(*pointers((dx, dw, dy, x, w, device_offsets, slots)),
                                                               m, e, n, k, 0, stream()))
                            return dx, dw
                        yield Operation(size, "fp32", "backward",
                                        {"custom": custom_backward}, expected_gradients)
    finally:
        bind(lib, "dscuda_cublas_destroy", [], None)()
