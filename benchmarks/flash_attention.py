"""Causal D128 BF16 MHA/GQA/MQA: PyTorch correctness, FlashAttention runtime."""

from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch
from flash_attn import flash_attn_func


def cases(args, family):
    lib = library("flash_attention")
    forward = bind(lib, "dscuda_flash_forward", [P] * 5 + [I] * 5 + [F, P])
    backward = bind(lib, "dscuda_flash_backward", [P] * 9 + [I] * 4 + [F, P])
    reference = args.reference or ("pytorch" if args.test else "flash_attention")
    if reference not in ("pytorch", "flash_attention"):
        raise ValueError("FlashAttention references: pytorch or flash_attention")

    if args.test:
        shapes = (
            (1, 64, 2, 2, 128),
            (2, 64, 4, 2, 128),
            (1, 128, 4, 1, 128),
            (1, 256, 8, 8, 128),
        )
    elif args.suite == "quick":
        shapes = tuple((1, 512, 8, kv_heads, 128) for kv_heads in (8, 2, 1))
    else:
        shapes = tuple(
            (batch, sequence, 8, kv_heads, 128)
            for batch in (1, 4)
            for sequence in (128, 256, 512, 1024, 2048)
            for kv_heads in (8, 2, 1)
        )

    for b, t, query_heads, key_value_heads, d in shapes:
        query_shape = (b, t, query_heads, d)
        key_value_shape = (b, t, key_value_heads, d)
        query = (torch.randn(query_shape, device="cuda") * .5).bfloat16()
        key = (torch.randn(key_value_shape, device="cuda") * .5).bfloat16()
        value = (torch.randn(key_value_shape, device="cuda") * .5).bfloat16()
        inputs = (query, key, value)
        oracle = tuple(x.float().detach().requires_grad_() for x in inputs)
        group_size = query_heads // key_value_heads
        scale = d**-.5
        mask = torch.ones(t, t, device="cuda", dtype=torch.bool).tril()

        def pytorch_forward(tensors=inputs):
            q, k, v = tensors
            if group_size > 1:
                k = k.repeat_interleave(group_size, dim=2)
                v = v.repeat_interleave(group_size, dim=2)
            q, k, v = (x.float().transpose(1, 2) for x in (q, k, v))
            scores = (q @ k.transpose(-1, -2) * scale).masked_fill(~mask, -torch.inf)
            return (scores.softmax(-1) @ v).transpose(1, 2).bfloat16(), scores.logsumexp(-1)

        saved_oracle = pytorch_forward(oracle)
        expected = tuple(x.detach() for x in saved_oracle)
        output = torch.empty(query_shape, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(b, query_heads, t, device="cuda")

        def custom_forward():
            checked(lib, "flash", forward(
                *pointers((output, lse, *inputs)),
                b, t, query_heads, key_value_heads, d, scale, stream()))
            return output, lse

        functions = {"custom": custom_forward, "PyTorch": pytorch_forward}
        label = "PyTorch"
        official_inputs = None
        official_saved = None
        if reference == "flash_attention":
            official_inputs = tuple(x.detach().requires_grad_() for x in inputs)

            def official_forward():
                result = flash_attn_func(
                    *official_inputs, dropout_p=0., softmax_scale=scale,
                    causal=True, return_attn_probs=True)
                return result[:2]

            official_saved = official_forward()
            functions = {"custom": custom_forward, "FlashAttention": official_forward}
            label = "FlashAttention"

        size = f"B={b},T={t},Hq={query_heads},Hkv={key_value_heads},D={d}"
        yield Operation(size, "bf16", "forward", functions, expected,
                        (1e-2, 1e-4), (1e-2, 1e-5))

        # Backward remains MHA-only until shared dK/dV reduction is implemented.
        if query_heads != key_value_heads:
            continue

        dout = (torch.randn(query_shape, device="cuda") * .5).bfloat16()

        def pytorch_backward():
            return tuple(x.bfloat16() for x in torch.autograd.grad(
                saved_oracle[0], oracle, dout, retain_graph=True))

        expected_gradients = pytorch_backward()
        gradients = tuple(torch.empty_like(x) for x in inputs)

        def custom_backward():
            checked(lib, "flash", backward(
                *pointers((*gradients, dout, output, lse, *inputs)),
                b, t, query_heads, d, scale, stream()))
            return gradients

        custom_forward()
        backward_functions = {"custom": custom_backward, "PyTorch": pytorch_backward}
        if reference == "flash_attention":
            def official_backward():
                return torch.autograd.grad(
                    official_saved[0], official_inputs, dout, retain_graph=True)

            backward_functions = {"custom": custom_backward, label: official_backward}

        yield Operation(size, "bf16", "backward", backward_functions,
                        expected_gradients, 1e-2, 1e-2)
