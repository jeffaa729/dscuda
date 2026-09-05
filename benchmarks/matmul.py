"""Row-major NN GEMM with cuBLAS and H100 DeepGEMM references."""

import importlib

from common import I, P, Operation, bind, checked, library, pointers, stream, torch


def cases(args, family):
    is_sm90 = torch.cuda.get_device_capability() == (9, 0)
    if args.suite == "h100" and not is_sm90:
        raise ValueError("The H100 GEMM suite requires an SM90 GPU.")
    reference = args.reference or ("both" if args.suite == "h100" else "cublas")
    if reference not in ("cublas", "deepgemm", "both"):
        raise ValueError("GEMM references: cublas, deepgemm, or both")
    use_cublas = reference in ("cublas", "both")
    use_deepgemm = reference in ("deepgemm", "both")
    deepgemm = importlib.import_module("reference.python.deepgemm") if use_deepgemm else None

    lib = library("operator")
    if use_cublas:
        checked(lib, "operator", bind(lib, "dscuda_cublas_init", [])())
    gemm = bind(lib, "dscuda_gemm", [P] * 3 + [I] * 5 + [P])
    dtypes = (torch.bfloat16,) if is_sm90 or reference == "deepgemm" else (
        torch.float32, torch.bfloat16)

    try:
        for dtype in dtypes:
            shapes = (
                (32, 48, 64), (128, 256, 64),
                (640, 128, 96), (1152, 128, 32)
            ) if args.test else tuple(
                (n, n, n) for n in (
                    (2048,) if args.suite == "quick" else (2048, 4096, 8192)))
            if args.test and dtype == torch.float32:
                shapes += ((17, 33, 65),)

            for m, n, k in shapes:
                left = torch.randn((m, k), device="cuda", dtype=dtype) * .1
                right = torch.randn((k, n), device="cuda", dtype=dtype) * .1
                expected = (left.float() @ right.float()).to(dtype)
                # An empty/incomplete SM90 kernel must fail correctness before timing.
                custom_output = torch.full((m, n), float("nan"), device="cuda", dtype=dtype)

                def native(output, use_reference):
                    checked(lib, "operator", gemm(
                        *pointers((output, left, right)),
                        m, n, k, int(dtype == torch.bfloat16),
                        use_reference, stream()))
                    return output

                functions = {
                    "custom": lambda output=custom_output: native(output, 0)
                }
                if use_cublas:
                    cublas_output = torch.empty_like(custom_output)
                    functions["cuBLAS"] = (
                        lambda output=cublas_output: native(output, 1))
                if use_deepgemm and dtype == torch.bfloat16:
                    deepgemm_output = torch.empty_like(custom_output)
                    functions["DeepGEMM"] = (
                        lambda output=deepgemm_output:
                            deepgemm.gemm_nn(left, right, output))

                tolerance = 1e-2 if dtype == torch.bfloat16 else 2e-4
                yield Operation(
                    f"M={m},N={n},K={k}",
                    "bf16" if dtype == torch.bfloat16 else "fp32",
                    "NN", functions, (expected,), tolerance, tolerance)
    finally:
        if use_cublas:
            bind(lib, "dscuda_cublas_destroy", [], None)()
