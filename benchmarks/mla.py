"""Dense C512/R64 BF16 MLA with PyTorch and official FlashMLA references."""

import ctypes
import importlib

from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch
from reference.python.mla import mla_forward, mla_backward, pack_cache


def cases(args, family):
    reference = args.reference or (
        "both" if args.suite == "h100" and not args.test else "pytorch")
    if reference == "all":
        reference = "both"
    if reference not in ("pytorch", "flashmla", "both"):
        raise ValueError("MLA references: pytorch, flashmla, or both (alias: all)")
    use_flashmla = reference != "pytorch"
    flashmla = None
    if use_flashmla:
        if args.operation not in (None, "decode"):
            raise ValueError(
                "FlashMLA matches C512/R64 dense decode only; "
                "use --reference pytorch for forward/backward.")
        flashmla = importlib.import_module("reference.python.flashmla")
        flashmla.load_decode()

    lib = library("mla")
    forward = bind(lib, "dscuda_mla_forward", [P] * 6 + [I] * 5 + [F, P])
    backward = bind(lib, "dscuda_mla_backward", [P] * 11 + [I] * 5 + [F, P])
    decode = bind(lib, "dscuda_mla_decode", [P] * 7 + [I] * 7 + [F, P])
    workspace_size = bind(
        lib, "dscuda_mla_workspace_elements", [I] * 4, ctypes.c_size_t)

    shapes = [("prefill", 1, 128, 8), ("decode", 2, 1024, 16)]
    if args.test:
        shapes = [("prefill", 1, 1, 1), ("prefill", 2, 17, 3),
                  ("prefill", 1, 65, 4), ("decode", 2, 23, 3),
                  ("decode", 3, 64, 16), ("decode", 2, 65, 64),
                  ("decode", 3, 129, 128)]
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
                       for shape in ((b, q, h, c), (b, q, h, r),
                                     (b, t, c), (b, t, r)))
        oracle = tuple(x.float().detach().requires_grad_() for x in inputs)
        scale = ctypes.c_float((c + r)**-.5).value
        output = torch.empty(b, q, h, c, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(b, h, q, device="cuda")
        lengths = tuple(max(1, t - i * t // (2 * b)) for i in range(b))
        if args.test and mode == "decode":
            lengths = (1,) + lengths[1:-1] + (t,)
        length_tensor = torch.tensor(lengths, device="cuda", dtype=torch.int32)
        mask = (
            torch.ones(t, t, device="cuda", dtype=torch.bool).tril()
            if mode == "prefill" else
            (torch.arange(t, device="cuda")[None] < length_tensor[:, None])[:, None, None]
        )
        saved_oracle = mla_forward(*oracle, scale, mask)
        saved = tuple(x.detach() for x in saved_oracle)
        workspace = torch.empty(workspace_size(b, h, splits, c), device="cuda")
        page_size = 64
        packed_query = torch.cat(inputs[:2], -1).contiguous()
        paged_cache, block_table = pack_cache(
            torch.cat(inputs[2:], -1), page_size)
        pages_per_sequence = block_table.shape[1]

        def pytorch_forward():
            return mla_forward(*inputs, scale, mask)

        def custom_forward():
            if mode == "prefill":
                checked(lib, "mla", forward(
                    *pointers((output, lse, *inputs)),
                    b, t, h, c, r, scale, stream()))
            else:
                checked(lib, "mla", decode(
                    *pointers((output, lse, packed_query, paged_cache,
                               block_table, length_tensor, workspace)),
                    b, h, c, r, page_size, pages_per_sequence, splits,
                    scale, stream()))
            return output, lse

        functions = {"custom": custom_forward, "PyTorch": pytorch_forward}
        if use_flashmla:
            official = flashmla.FlashMLADecode(
                packed_query, paged_cache, block_table, length_tensor, scale)
            if reference == "flashmla":
                functions.pop("PyTorch")
            functions["FlashMLA"] = official

        size = f"B={b},Q={q},KV={t},H={h},C={c},RoPE={r}"
        if mode == "decode":
            size += f",lengths={lengths},page={page_size},custom_splits={splits}"
        yield Operation(
            size, "bf16", "forward" if mode == "prefill" else "decode",
            functions, saved, (8e-3, 1e-4), (8e-3, 1e-4))

        if mode == "prefill":
            dout = (torch.randn_like(output) * .2).bfloat16()
            expected_gradients = tuple(x.bfloat16() for x in torch.autograd.grad(
                saved_oracle[0], oracle, dout))
            gradients = tuple(torch.empty_like(x) for x in inputs)

            def custom_backward():
                checked(lib, "mla", backward(
                    *pointers((*gradients, dout, *saved, *inputs)),
                    b, t, h, c, r, scale, stream()))
                return gradients

            def pytorch_backward():
                return mla_backward(dout, *saved, *inputs, scale, mask)

            yield Operation(
                size, "bf16", "backward",
                {"custom": custom_backward, "PyTorch": pytorch_backward},
                expected_gradients, 1e-2, 1e-2)
