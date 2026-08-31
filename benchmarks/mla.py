"""Dense C512/R64 MLA. PyTorch uses FP32 math; official decode uses BF16 output.

Official cache packing stays outside timing; the native FP32-to-BF16 output cast
is timed. This compares prepared operations with different cache layouts.
"""
import ctypes

from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.mla import mla_forward, mla_backward


def cases(args, family):
    lib = library("mla")
    forward = bind(lib, "dscuda_mla_forward", [P] * 6 + [I] * 5 + [F, P])
    backward = bind(lib, "dscuda_mla_backward", [P] * 11 + [I] * 5 + [F, I, P])
    decode = bind(lib, "dscuda_mla_decode", [P] * 8 + [I] * 6 + [F, P])
    workspace_size = bind(lib, "dscuda_mla_workspace_elements", [I] * 4, ctypes.c_size_t)
    reference = args.reference or "pytorch"
    if reference not in ("pytorch", "flashmla", "both"):
        raise ValueError("MLA references: pytorch, flashmla, or both")
    shapes = [("prefill", 1, 128, 8), ("decode", 2, 1024, 16)]
    if args.test:
        shapes = [("prefill", 1, 1, 1), ("prefill", 2, 17, 3), ("prefill", 1, 65, 4),
                  ("decode", 2, 23, 3), ("decode", 3, 129, 5)]
    elif args.suite != "quick":
        shapes += [("prefill", 2, 257, 4), ("prefill", 1, 512, 8),
                   ("decode", 1, 256, 16), ("decode", 4, 4096, 32)]
    for mode, b, t, h in shapes:
        if reference != "pytorch" and mode != "decode":
            continue
        q, c, r, splits = (t if mode == "prefill" else 1), 512, 64, 8
        inputs = tuple((torch.randn(shape, device="cuda") * .25).bfloat16()
                       for shape in ((b, q, h, c), (b, q, h, r), (b, t, c), (b, t, r)))
        oracle = tuple(x.float().detach().requires_grad_() for x in inputs)
        scale = ctypes.c_float((c + r)**-.5).value
        output, lse = torch.empty(b, q, h, c, device="cuda"), torch.empty(b, h, q, device="cuda")
        lengths = tuple(max(1, t - i * t // (2 * b)) for i in range(b))
        if args.test and mode == "decode":
            lengths = (1,) + lengths[1:]
        length_tensor = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        mask = (torch.ones(t, t, device="cuda", dtype=torch.bool).tril() if mode == "prefill" else
                (torch.arange(t, device="cuda")[None] < length_tensor[:, None])[:, None, None])
        saved_oracle = mla_forward(*oracle, scale, mask)
        saved = tuple(x.detach() for x in saved_oracle)
        workspace = torch.empty(workspace_size(b, h, splits, c), device="cuda")

        def pytorch_forward():
            out, logsum = mla_forward(*inputs, scale, mask)
            return (out.bfloat16() if reference != "pytorch" else out), logsum

        def custom_forward():
            params = pointers((output, lse, *inputs))
            if mode == "prefill":
                checked(lib, "mla", forward(*params, b, t, h, c, r, scale, stream()))
            else:
                checked(lib, "mla", decode(*params, length_tensor.data_ptr(), workspace.data_ptr(),
                                            b, t, h, c, r, splits, scale, stream()))
            return (output.bfloat16() if reference != "pytorch" else output), lse

        functions = {"custom": custom_forward, "PyTorch": pytorch_forward}
        expected = saved
        if reference != "pytorch":
            from reference.python.flashmla import FlashMLADecode
            official = FlashMLADecode(torch.cat(inputs[:2], -1), torch.cat(inputs[2:], -1), length_tensor, scale)
            if reference == "flashmla":
                functions.pop("PyTorch")
            functions["FlashMLA"] = official
            expected = (saved[0].bfloat16(), saved[1])
        size = f"B={b},Q={q},KV={t},H={h},C={c},RoPE={r}"
        if mode == "decode":
            size += f",lengths={lengths},splits={splits}"
        if reference != "pytorch":
            size += ",out=bf16"
        yield Operation(size, "bf16", "forward" if mode == "prefill" else "decode",
                        functions, expected, 5e-4 if reference != "pytorch" else 2e-4, 8e-3)
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
