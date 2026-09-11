"""Causal D128 BF16 MHA/GQA/MQA: PyTorch correctness and official FA runtime."""

import importlib

from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch


def load_reference_apis(reference):
    aliases = {
        "flash_attention": "flash_attention_2",
        "fa2": "flash_attention_2",
        "fa3": "flash_attention_3",
        "fa4": "flash_attention_4",
    }
    reference = aliases.get(reference, reference)
    valid = ("pytorch", "flash_attention_2", "flash_attention_3", "flash_attention_4", "all")
    if reference not in valid:
        raise ValueError("FlashAttention references: pytorch, fa2, fa3, fa4, or all")

    selected = (
        ("flash_attention_2", "flash_attention_3", "flash_attention_4")
        if reference == "all" else (reference,)
    )
    apis = {}
    if "flash_attention_2" in selected:
        apis["FlashAttention-2"] = importlib.import_module("flash_attn").flash_attn_func
    if "flash_attention_3" in selected:
        module = importlib.import_module("flash_attn_3.flash_attn_interface")
        apis["FlashAttention-3"] = module.flash_attn_func
    if "flash_attention_4" in selected:
        apis["FlashAttention-4"] = importlib.import_module("flash_attn.cute").flash_attn_func
    return reference, apis


def cases(args, family):
    lib = library("flash_attention")
    forward = bind(lib, "dscuda_flash_forward", [P] * 5 + [I] * 5 + [F, P])
    backward = bind(lib, "dscuda_flash_backward", [P] * 9 + [I] * 4 + [F, P])
    default_reference = "pytorch" if args.test else ("all" if args.suite == "h100" else "flash_attention_2")
    reference, reference_apis = load_reference_apis(args.reference or default_reference)

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

        functions = {"custom": custom_forward}
        reference_inputs = {}
        if reference == "pytorch":
            functions["PyTorch"] = pytorch_forward

        for label, api in reference_apis.items():
            official_inputs = tuple(x.detach().requires_grad_() for x in inputs)
            reference_inputs[label] = official_inputs
            if label == "FlashAttention-2":
                def official_forward(tensors=official_inputs, function=api):
                    return function(
                        *tensors, dropout_p=0., softmax_scale=scale,
                        causal=True, return_attn_probs=True)[:2]
            elif label == "FlashAttention-3":
                def official_forward(tensors=official_inputs, function=api):
                    return function(
                        *tensors, softmax_scale=scale, causal=True,
                        return_attn_probs=True)[:2]
            else:
                def official_forward(tensors=official_inputs, function=api):
                    return function(
                        *tensors, softmax_scale=scale, causal=True,
                        return_lse=True)[:2]
            functions[label] = official_forward

        size = f"B={b},T={t},Hq={query_heads},Hkv={key_value_heads},D={d}"
        yield Operation(size, "bf16", "forward", functions, expected,
                        (1e-2, 1e-4), (1e-2, 1e-5))

        # Backward remains MHA-only and keeps its existing PyTorch/FA2 path.
        if query_heads != key_value_heads or reference not in ("pytorch", "flash_attention_2"):
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
        backward_functions = {"custom": custom_backward}
        if reference == "pytorch":
            backward_functions["PyTorch"] = pytorch_backward
        else:
            official_inputs = reference_inputs["FlashAttention-2"]
            official_saved = functions["FlashAttention-2"]()

            def official_backward():
                return torch.autograd.grad(
                    official_saved[0], official_inputs, dout, retain_graph=True)

            backward_functions["FlashAttention-2"] = official_backward

        yield Operation(size, "bf16", "backward", backward_functions,
                        expected_gradients, 1e-2, 1e-2)
