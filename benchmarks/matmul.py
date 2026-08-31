"""Row-major NN GEMM: PyTorch correctness and cuBLAS timing."""
from common import I, P, Operation, bind, checked, library, pointers, stream, torch


def cases(args, family):
    if args.reference not in (None, "cublas"):
        raise ValueError("GEMM uses cuBLAS as its only benchmark reference")
    lib = library("operator")
    checked(lib, "operator", bind(lib, "dscuda_cublas_init", [])())
    gemm = bind(lib, "dscuda_gemm", [P] * 3 + [I] * 5 + [P])
    try:
        for dtype in (torch.float32, torch.bfloat16):
            shapes = ((32, 48, 64), (128, 256, 64), (640, 128, 96), (1152, 128, 32)) if args.test else tuple(
                (n, n, n) for n in ((2048,) if args.suite == "quick" else (2048, 4096, 8192)))
            if args.test and dtype == torch.float32:
                shapes += ((17, 33, 65),)
            for m, n, k in shapes:
                a = torch.randn((m, k), device="cuda", dtype=dtype) * .1
                b = torch.randn((k, n), device="cuda", dtype=dtype) * .1
                expected = a.float() @ b.float()
                outputs = [torch.empty((m, n), device="cuda") for _ in range(2)]

                def call(reference=0):
                    checked(lib, "operator", gemm(*pointers((outputs[reference], a, b)),
                            m, n, k, int(dtype == torch.bfloat16), reference, stream()))
                    return outputs[reference]

                yield Operation(f"M={m},N={n},K={k}",
                                "bf16" if dtype == torch.bfloat16 else "fp32", "NN",
                                {"custom": call, "cuBLAS": lambda: call(1)}, (expected,))
    finally:
        bind(lib, "dscuda_cublas_destroy", [], None)()
