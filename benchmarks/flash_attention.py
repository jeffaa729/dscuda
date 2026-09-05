"""Causal D128 BF16 attention: PyTorch correctness, FlashAttention runtime."""

from common import F, I, P, Operation, bind, checked, library, pointers, stream, torch
from flash_attn import flash_attn_func


def cases(args, family):
    lib = library("flash_attention")
    forward = bind(lib, "dscuda_flash_forward", [P] * 5 + [I] * 4 + [F, P])
    backward = bind(lib, "dscuda_flash_backward", [P] * 9 + [I] * 4 + [F, P])
    reference = args.reference or ("pytorch" if args.test else "flash_attention")
    if reference not in ("pytorch", "flash_attention"):
        raise ValueError("FlashAttention references: pytorch or flash_attention")
    shapes = (
        ((1, 64, 2, 128), (2, 64, 3, 128),
         (1, 128, 4, 128), (1, 256, 2, 128))
        if args.test else
        (((1, 512, 8, 128),) if args.suite == "quick" else
         tuple((b, t, h, 128) for b in (1, 4)
               for t in (128, 256, 512, 1024, 2048) for h in (4, 8)))
    )

    for shape in shapes:
        b, t, h, d = shape
        inputs = tuple((torch.randn(shape, device="cuda") * .5).bfloat16()
                       for _ in range(3))
        oracle = tuple(x.float().detach().requires_grad_() for x in inputs)
        scale = d**-.5
        mask = torch.ones(t, t, device="cuda", dtype=torch.bool).tril()

        def pytorch_forward(tensors=inputs):
            q, k, v = (x.float().transpose(1, 2) for x in tensors)
            scores = (q @ k.transpose(-1, -2) * scale).masked_fill(~mask, -torch.inf)
            return (scores.softmax(-1) @ v).transpose(1, 2).bfloat16(), scores.logsumexp(-1)

        saved_oracle = pytorch_forward(oracle)
        expected = tuple(x.detach() for x in saved_oracle)
        dout = (torch.randn(shape, device="cuda") * .5).bfloat16()

        def pytorch_backward():
            return tuple(x.bfloat16() for x in torch.autograd.grad(
                saved_oracle[0], oracle, dout, retain_graph=True))

        expected_gradients = pytorch_backward()
        output = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        lse = torch.empty(b, h, t, device="cuda")
        gradients = tuple(torch.empty_like(output) for _ in inputs)

        def custom_forward():
            checked(lib, "flash", forward(
                *pointers((output, lse, *inputs)), *shape, scale, stream()))
            return output, lse

        def custom_backward():
            checked(lib, "flash", backward(
                *pointers((*gradients, dout, output, lse, *inputs)),
                *shape, scale, stream()))
            return gradients

        custom_forward()
        functions = {"forward": pytorch_forward, "backward": pytorch_backward}
        label = "PyTorch"
        if reference == "flash_attention":
            official_inputs = tuple(x.detach().requires_grad_() for x in inputs)

            def official_forward():
                result = flash_attn_func(
                    *official_inputs, dropout_p=0., softmax_scale=scale,
                    causal=True, return_attn_probs=True)
                return result[:2]

            official_saved = official_forward()

            def official_backward():
                return torch.autograd.grad(
                    official_saved[0], official_inputs, dout, retain_graph=True)

            functions = {"forward": official_forward, "backward": official_backward}
            label = "FlashAttention"

        size = f"B={b},T={t},H={h},D={d}"
        yield Operation(size, "bf16", "forward",
                        {"custom": custom_forward, label: functions["forward"]},
                        expected, (1e-2, 1e-4), (1e-2, 1e-5))
        yield Operation(size, "bf16", "backward",
                        {"custom": custom_backward, label: functions["backward"]},
                        expected_gradients, 1e-2, 1e-2)
