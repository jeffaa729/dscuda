"""KDA forward/backward with FLA timing and a recurrent PyTorch oracle."""

from common import Operation, torch
from reference.python.kda import fla_forward, kda_forward


def cases(args, family):
    reference = args.reference or "fla"
    if reference not in ("fla", "pytorch"):
        raise ValueError("KDA references: fla or pytorch")
    function = fla_forward if reference == "fla" else kda_forward
    label = "FLA" if reference == "fla" else "PyTorch"
    tokens = (65,) if args.test else (
        (128,) if args.suite == "quick" else (256, 512, 1024))

    for token_count in tokens:
        # The correctness shape crosses FLA's chunk boundary. Runtime uses the
        # fixed matrix geometry recorded in BENCHMARK_MATRIX.md.
        batch, heads, value_heads, key_width, value_width = (
            (1, 2, 4, 32, 32) if args.test else (1, 4, 4, 128, 128))

        def normalized():
            values = torch.randn(
                batch, token_count, heads, key_width, device="cuda")
            return torch.nn.functional.normalize(values, dim=-1).bfloat16()

        inputs = tuple(tensor.requires_grad_() for tensor in (
            normalized(),
            normalized(),
            torch.randn(
                batch, token_count, value_heads, value_width,
                device="cuda").bfloat16() * .1,
            -torch.rand(
                batch, token_count, value_heads, key_width,
                device="cuda") * .1,
            torch.rand(batch, token_count, value_heads, device="cuda"),
            torch.randn(
                batch, value_heads, key_width, value_width,
                device="cuda") * .05,
        ))
        saved = function(*inputs)
        expected = kda_forward(*inputs)
        output_gradients = (
            torch.randn_like(expected[0]),
            torch.randn_like(expected[1]) * .1,
        )
        expected_gradients = torch.autograd.grad(
            expected, inputs, output_gradients, retain_graph=True)
        size = (
            f"B={batch},T={token_count},H={heads},HV={value_heads},"
            f"K={key_width},V={value_width}")

        yield Operation(
            size, "bf16", "forward", {label: lambda: function(*inputs)},
            expected, 1e-3, 2e-2)
        yield Operation(
            size, "bf16", "backward",
            {label: lambda: torch.autograd.grad(
                saved, inputs, output_gradients, retain_graph=True)},
            expected_gradients, 3e-3, 4e-2)
