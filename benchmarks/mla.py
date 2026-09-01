"""Dense C512/R64 MLA. PyTorch uses FP32 math; official decode uses BF16 output.

Official cache packing stays outside timing; the native FP32-to-BF16 output cast
is timed. This compares prepared operations with different cache layouts.
The h100 benchmark suite defaults to PyTorch plus FlashMLA dense decode.
"""
import ctypes

from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.mla import mla_forward, mla_backward


def cases(args, family):
    reference = args.reference or ("both" if args.suite == "h100" and not args.test else "pytorch")
    if reference == "all":
        reference = "both"
    if reference not in ("pytorch", "flashmla", "both"):
        raise ValueError("MLA references: pytorch, flashmla, or both (alias: all)")
    use_flashmla = reference != "pytorch"
    if use_flashmla:
        if args.operation not in (None, "decode"):
            raise ValueError("FlashMLA matches C512/R64 dense decode only; use --reference pytorch for forward/backward.")
        from reference.python.flashmla import FlashMLADecode, load_decode
        load_decode()
    lib = library("mla")
    forward = bind(lib, "dscuda_mla_forward", [P] * 6 + [I] * 5 + [F, P])
    backward = bind(lib, "dscuda_mla_backward", [P] * 11 + [I] * 5 + [F, I, P])
    decode = bind(lib, "dscuda_mla_decode", [P] * 8 + [I] * 6 + [F, P])
    workspace_size = bind(lib, "dscuda_mla_workspace_elements", [I] * 4, ctypes.c_size_t)
    shapes = [("prefill", 1, 128, 8), ("decode", 2, 1024, 16)]
    if args.test:
        shapes = [("prefill", 1, 1, 1), ("prefill", 2, 17, 3), ("prefill", 1, 65, 4),
                  ("decode", 2, 23, 3), ("decode", 3, 64, 16),
                  ("decode", 2, 65, 64), ("decode", 3, 129, 128)]
    elif args.suite == "h100":
        shapes = [("decode", 1, 1024, 64), ("decode", 4, 4096, 64),
                  ("decode", 4, 4096, 128), ("decode", 8, 8192, 128)]
    elif args.suite != "quick":
        shapes += [("prefill", 2, 257, 4), ("prefill", 1, 512, 8),
                   ("decode", 1, 256, 16), ("decode", 4, 4096, 32)]
    for mode, b, t, h in shapes:
        if use_flashmla and mode != "decode":
            continue
        q, c, r, splits = (t if mode == "prefill" else 1), 512, 64, 8
        inputs = tuple((torch.randn(shape, device="cuda") * .25).bfloat16()
                       for shape in ((b, q, h, c), (b, q, h, r), (b, t, c), (b, t, r)))
        oracle = tuple(x.float().detach().requires_grad_() for x in inputs)
        scale = ctypes.c_float((c + r)**-.5).value
        output, lse = torch.empty(b, q, h, c, device="cuda"), torch.empty(b, h, q, device="cuda")
        lengths = tuple(max(1, t - i * t // (2 * b)) for i in range(b))
        if args.test and mode == "decode":
            lengths = (1,) + lengths[1:-1] + (t,)
        length_tensor = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        mask = (torch.ones(t, t, device="cuda", dtype=torch.bool).tril() if mode == "prefill" else
                (torch.arange(t, device="cuda")[None] < length_tensor[:, None])[:, None, None])
        saved_oracle = mla_forward(*oracle, scale, mask)
        saved = tuple(x.detach() for x in saved_oracle)
        workspace = torch.empty(workspace_size(b, h, splits, c), device="cuda")

        def pytorch_forward(bf16_output=use_flashmla):
            out, logsum = mla_forward(*inputs, scale, mask)
            return (out.bfloat16() if bf16_output else out), logsum

        def custom_forward(bf16_output=use_flashmla):
            params = pointers((output, lse, *inputs))
            if mode == "prefill":
                checked(lib, "mla", forward(*params, b, t, h, c, r, scale, stream()))
            else:
                checked(lib, "mla", decode(*params, length_tensor.data_ptr(), workspace.data_ptr(),
                                            b, t, h, c, r, splits, scale, stream()))
            return (output.bfloat16() if bf16_output else output), lse

        functions = {"custom": custom_forward, "PyTorch": pytorch_forward}
        expected = saved
        if use_flashmla:
            official = FlashMLADecode(torch.cat(inputs[:2], -1), torch.cat(inputs[2:], -1), length_tensor, scale)
            if reference == "flashmla":
                functions.pop("PyTorch")
            functions["FlashMLA"] = official
            expected = (saved[0].bfloat16(), saved[1])
        size = f"B={b},Q={q},KV={t},H={h},C={c},RoPE={r}"
        if mode == "decode":
            size += f",lengths={lengths},custom_splits={splits}"
        if use_flashmla:
            size += ",page=64"
        dtype = "bf16->bf16" if use_flashmla else "bf16"
        yield Operation(size, dtype, "forward" if mode == "prefill" else "decode",
                        functions, expected, (5e-4 if use_flashmla else 2e-4, 1e-5), (8e-3, 1e-4))
        if args.test and mode == "decode" and not use_flashmla:
            # Test the library comparison's output cast locally, using real
            # custom/PyTorch operations; no substitute FlashMLA implementation.
            yield Operation(size, "bf16->bf16", "decode_bf16",
                            {"custom": lambda: custom_forward(True), "PyTorch": lambda: pytorch_forward(True)},
                            (saved[0].bfloat16(), saved[1]), (5e-4, 1e-5), (8e-3, 1e-4))
        if mode == "prefill":
            dout = torch.randn_like(output) * .2
            expected_gradients = torch.autograd.grad(saved_oracle[0], oracle, dout)
            gradients = tuple(torch.empty_like(x, dtype=torch.float32) for x in inputs)
            def custom_backward(accumulate=False):
                checked(lib, "mla", backward(*pointers((*gradients, dout, *saved, *inputs)),
                                              b, t, h, c, r, scale, int(accumulate), stream()))
                return gradients
            def pytorch_backward():
                return mla_backward(dout, *saved, *inputs, scale, mask)
            yield Operation(size, "bf16", "backward",
                            {"custom": custom_backward, "PyTorch": pytorch_backward}, expected_gradients)
            if args.test:
                for x in gradients:
                    x.fill_(1)
                for actual, expected in zip(custom_backward(True), expected_gradients):
                    torch.testing.assert_close(actual, expected + 1, atol=2e-4, rtol=2e-3)
