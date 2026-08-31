"""Reference-only HCA/DSA/CSA workloads and FLA KDA checked against PyTorch.

There are no custom CUDA kernels for these families yet; their custom times stay blank.
"""
from common import Operation, torch
from reference.python.attention import sparse_attention
from reference.python.compression import compress, local_indices
from reference.python.csa import csa_forward
from reference.python.dsa import dsa_forward, index_scores, select_indices
from reference.python.hca import hca_forward
from reference.python.kda import fla_forward, kda_forward


def attention_stages(family, t):
    b, h, d, hi, di = 1, 4, 512, 4, 128
    def rand(*shape):
        return (torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * .1).requires_grad_()
    qi, weights = rand(b, t, hi, di), rand(b, t, hi)
    q = rand(b, t, h, 576 if family == "dsa" else d)
    local = rand(b, t, q.shape[-1])
    positions = torch.arange(t, device="cuda")
    if family == "dsa":
        ki = rand(b, t, di)
        visible = positions[None] <= positions[:, None]
        indexer = lambda: (select_indices(index_scores(qi, ki, weights, visible), 64),)
        ids = indexer()[0]
        attention = lambda: sparse_attention(q, local, local[..., :512], ids)
        stages = dict(indexer=indexer, attention=attention,
                      pipeline=lambda: dsa_forward(q, local, qi, ki, weights, 64)[:2])
        differentiable = (q, local)
        size = f"B={b},Q={t},KV={t},H={h},C=512,R=64,Hi={hi},Di={di},topk=64"
    else:
        ratio, overlap = (128, False) if family == "hca" else (4, True)
        channels = d * (2 if overlap else 1)
        values, gates = rand(b, t, channels).float(), rand(b, t, channels).float()
        bias = torch.zeros(ratio, channels, device="cuda")
        norm = torch.ones(d, device="cuda")
        compress_main = lambda: compress(values, gates, bias, ratio, norm,
                                          overlap=overlap, storage_dtype=torch.bfloat16)
        compressed = compress_main().detach().requires_grad_()
        sink = torch.zeros(h, device="cuda", requires_grad=True)
        if family == "hca":
            attention = lambda: hca_forward(q, local, compressed, sink=sink)
            stages = dict(compression=lambda: (compress_main(),), attention=attention,
                          pipeline=lambda: hca_forward(q, local, compress_main(), sink=sink))
        else:
            iv, ig = rand(b, t, 2 * di).float(), rand(b, t, 2 * di).float()
            ib, iw = torch.zeros(ratio, 2 * di, device="cuda"), torch.ones(di, device="cuda")
            compress_index = lambda: compress(iv, ig, ib, ratio, iw, overlap=True,
                                              storage_dtype=torch.bfloat16)
            ik = compress_index().detach()
            visible = (torch.arange(t // ratio, device="cuda")[None] + 1) * ratio <= positions[:, None] + 1
            indexer = lambda: (select_indices(index_scores(qi, ik, weights, visible), 32),)
            selected = indexer()[0]
            ids = torch.cat((local_indices(positions, 128)[None],
                             torch.where(selected >= 0, selected + t, -1)), -1)
            def attention():
                cache = torch.cat((local, compressed), 1)
                return sparse_attention(q, cache, cache, ids, sink=sink)
            stages = dict(compression=lambda: (compress_main(), compress_index()),
                          indexer=indexer, attention=attention,
                          pipeline=lambda: csa_forward(q, local, compress_main(), qi,
                                                      compress_index(), weights, 32, sink=sink)[:2])
        differentiable = (q, local, compressed, sink)
        size = f"B={b},Q={t},KV={t},H={h},D={d},ratio={ratio},window=128"
        if family == "csa":
            size += f",Hi={hi},Di={di},topk=32"
    saved = attention()[0]
    dout = torch.randn_like(saved)
    stages["backward"] = lambda: torch.autograd.grad(saved, differentiable, dout, retain_graph=True)
    return size, stages


def kda_cases(args, t):
    # The correctness shape crosses a chunk boundary and includes grouped value heads.
    b, h, hv, k, v = (1, 2, 4, 32, 32) if args.test else (1, 4, 4, 128, 128)
    def normalized():
        return torch.nn.functional.normalize(torch.randn(b, t, h, k, device="cuda"), dim=-1).bfloat16()
    inputs = tuple(x.requires_grad_() for x in (
        normalized(), normalized(), torch.randn(b, t, hv, v, device="cuda").bfloat16() * .1,
        -torch.rand(b, t, hv, k, device="cuda") * .1, torch.rand(b, t, hv, device="cuda"),
        torch.randn(b, hv, k, v, device="cuda") * .05))
    reference = args.reference or "fla"
    if reference not in ("fla", "pytorch"):
        raise ValueError("KDA references: fla or pytorch")
    function = fla_forward if reference == "fla" else kda_forward
    label = "FLA" if reference == "fla" else "PyTorch"
    saved, expected = function(*inputs), kda_forward(*inputs)
    dout = (torch.randn_like(expected[0]), torch.randn_like(expected[1]) * .1)
    expected_gradients = torch.autograd.grad(expected, inputs, dout, retain_graph=True)
    size = f"B={b},T={t},H={h},HV={hv},K={k},V={v}"
    yield Operation(size, "bf16", "forward", {label: lambda: function(*inputs)}, expected, 1e-3, 2e-2)
    yield Operation(size, "bf16", "backward",
                    {label: lambda: torch.autograd.grad(saved, inputs, dout, retain_graph=True)},
                    expected_gradients, 3e-3, 4e-2)


def cases(args, family):
    tokens = (65,) if args.test else ((128,) if args.suite == "quick" else (256, 512, 1024))
    for t in tokens:
        if family == "kda":
            yield from kda_cases(args, t)
        else:
            if args.reference not in (None, "pytorch"):
                raise ValueError("HCA/DSA/CSA currently have PyTorch references only")
            size, stages = attention_stages(family, t)
            for name, function in stages.items():
                yield Operation(size, "bf16", name, {"PyTorch": function}, None)
