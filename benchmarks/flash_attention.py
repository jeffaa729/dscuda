"""Causal D128 attention: PyTorch correctness, official FlashAttention runtime."""
from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch


def cases(args, family):
    lib = library("flash_attention")
    forward = bind(lib, "dscuda_flash_forward", [P] * 5 + [I] * 4 + [F, P])
    backward = bind(lib, "dscuda_flash_backward", [P] * 9 + [I] * 4 + [F, P])
    float_forward = bind(lib, "dscuda_flash_forward_fp32_output", [P] * 5 + [I] * 5 + [F, P])
    float_backward = bind(lib, "dscuda_flash_backward_fp32_output", [P] * 9 + [I] * 5 + [F, P])
    reference = args.reference or ("pytorch" if args.test else "flash_attention")
    if reference not in ("pytorch", "flash_attention"):
        raise ValueError("FlashAttention references: pytorch or flash_attention")
    if reference == "flash_attention":
        from flash_attn import flash_attn_func
    shapes = ((1, 17, 2, 128), (2, 64, 3, 128), (1, 128, 4, 128), (1, 256, 2, 128)) if args.test else (
        ((1, 512, 8, 128),) if args.suite == "quick" else
        tuple((b, t, h, 128) for b in (1, 4) for t in (128, 256, 512, 1024, 2048) for h in (4, 8)))
    for shape in shapes:
        b, t, h, d = shape
        variants = ("fp32", "bf16-fp32", "bf16") if args.test and t % 64 == 0 else (
            ("fp32", "bf16-fp32") if args.test else ("bf16",))
        for variant in variants:
            dtype = torch.float32 if variant == "fp32" else torch.bfloat16
            out_dtype = torch.bfloat16 if variant == "bf16" else torch.float32
            inputs = tuple((torch.randn(shape, device="cuda") * .5).to(dtype) for _ in range(3))
            oracle = tuple(x.float().detach().requires_grad_() for x in inputs)
            scale = d**-.5
            mask = torch.ones(t, t, device="cuda", dtype=torch.bool).tril()

            def pytorch_forward(tensors=inputs):
                q, k, v = (x.float().transpose(1, 2) for x in tensors)
                scores = (q @ k.transpose(-1, -2) * scale).masked_fill(~mask, -torch.inf)
                return (scores.softmax(-1) @ v).transpose(1, 2).to(out_dtype), scores.logsumexp(-1)

            saved = pytorch_forward(oracle)
            dout = (torch.randn(shape, device="cuda") * .5).to(dtype).to(out_dtype)
            def pytorch_backward():
                return tuple(x.to(out_dtype) for x in torch.autograd.grad(
                    saved[0], oracle, dout, retain_graph=True))
            expected_gradients = pytorch_backward()
            output = torch.empty(shape, device="cuda", dtype=out_dtype)
            lse = torch.empty(b, h, t, device="cuda")
            gradients = tuple(torch.empty_like(output) for _ in inputs)

            def custom_forward():
                params = (*pointers((output, lse, *inputs)), *shape)
                function = forward if variant == "bf16" else float_forward
                if variant != "bf16":
                    params += (int(dtype == torch.bfloat16),)
                checked(lib, "flash", function(*params, scale, stream()))
                return output, lse

            def custom_backward():
                params = (*pointers((*gradients, dout, output, lse, *inputs)), *shape)
                function = backward if variant == "bf16" else float_backward
                if variant != "bf16":
                    for x in gradients:
                        x.zero_()  # The legacy FP32-output API accumulates gradients.
                    params += (int(dtype == torch.bfloat16),)
                checked(lib, "flash", function(*params, scale, stream()))
                return gradients

            custom_forward()
            functions = {"forward": pytorch_forward, "backward": pytorch_backward}
            label = "PyTorch"
            if reference == "flash_attention":
                if variant != "bf16":
                    raise ValueError("The official comparison uses native BF16 IO.")
                official_inputs = tuple(x.detach().requires_grad_() for x in inputs)
                def official_forward():
                    result = flash_attn_func(*official_inputs, dropout_p=0., softmax_scale=scale,
                                             causal=True, return_attn_probs=True)
                    return result[:2]
                official_saved = official_forward()
                def official_backward():
                    return torch.autograd.grad(official_saved[0], official_inputs, dout, retain_graph=True)
                functions = {"forward": official_forward, "backward": official_backward}
                label = "FlashAttention"
            size = f"B={b},T={t},H={h},D={d}"
            if variant == "bf16-fp32":
                size += ",out=fp32"
            # Keep the existing BF16 benchmark bound: Tensor Core P/dS/dO round
            # internally even with FP32 output. LSE remains an FP32 calculation.
            tolerance = 1e-2 if dtype == torch.bfloat16 and t % 64 == 0 else 2e-4
            yield Operation(size, "fp32" if dtype == torch.float32 else "bf16", "forward",
                            {"custom": custom_forward, label: functions["forward"]},
                            tuple(x.detach() for x in saved), (tolerance, 1e-4), (1e-2, 1e-5))
            yield Operation(size, "fp32" if dtype == torch.float32 else "bf16", "backward",
                            {"custom": custom_backward, label: functions["backward"]},
                            expected_gradients, tolerance, 1e-2)
