"""Row-major GEMM: PyTorch correctness and same-precision cuBLAS timing."""
from common import I, P, Operation, bind, checked, library, pointers, stream, torch


def cases(args, family):
    lib = library("operator")
    checked(lib, "operator", bind(lib, "dscuda_cublas_init", [])())
    gemm = bind(lib, "dscuda_gemm", [P] * 3 + [I] * 8 + [P])
    try:
        for dtype in (torch.float32, torch.bfloat16):
            shapes = ((32, 48, 64), (128, 256, 64), (640, 128, 96), (1152, 128, 32)) if args.test else tuple(
                (n, n, n) for n in ((2048,) if args.suite == "quick" else (2048, 4096, 8192)))
            if args.test and dtype == torch.float32:
                shapes += ((17, 33, 65),)
            for m, n, k in shapes:
                for ta, tb in ((False, False), (False, True), (True, False), (True, True)):
                    if not args.test and ta and tb:
                        continue
                    a = torch.randn((k, m) if ta else (m, k), device="cuda", dtype=dtype) * .1
                    b = torch.randn((n, k) if tb else (k, n), device="cuda", dtype=dtype) * .1
                    expected = (a.float().T if ta else a.float()) @ (b.float().T if tb else b.float())
                    outputs = [torch.empty((m, n), device="cuda") for _ in range(2)]
                    def call(reference=0, accumulate=0):
                        checked(lib, "operator", gemm(*pointers((outputs[reference], a, b)),
                                m, n, k, int(dtype == torch.bfloat16), int(ta), int(tb),
                                accumulate, reference, stream()))
                        return outputs[reference]
                    reference = args.reference or ("pytorch" if args.test else "cublas")
                    if reference not in ("cublas", "pytorch"):
                        raise ValueError("GEMM references: cublas or pytorch")
                    def pytorch():
                        return (a.float().T if ta else a.float()) @ (b.float().T if tb else b.float())
                    name = ("T" if ta else "N") + ("T" if tb else "N")
                    yield Operation(f"M={m},N={n},K={k}", "bf16" if dtype == torch.bfloat16 else "fp32", name,
                                    {"custom": call, "cuBLAS" if reference == "cublas" else "PyTorch":
                                     (lambda: call(1)) if reference == "cublas" else pytorch}, (expected,))
                    if args.test:
                        outputs[0].fill_(1)
                        torch.testing.assert_close(call(accumulate=1), expected + 1, atol=2e-4, rtol=2e-3)
    finally:
        bind(lib, "dscuda_cublas_destroy", [], None)()
