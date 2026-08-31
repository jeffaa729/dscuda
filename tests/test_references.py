"""Independent FP64 oracles/gradchecks for the new kernel reference boundaries."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from reference.python.attention import dense_attention, rotary, sparse_attention
from reference.python.compression import compress, compressed_indices
from reference.python.csa import csa_forward
from reference.python.dsa import dsa_forward, index_scores, select_indices
from reference.python.hca import hca_forward
from reference.python.kda import activate_gates, kda_forward
from reference.python.moe import combine, dispatch, grouped_gemm, moe_forward, route


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(71)

    def rand(self, *shape, grad=False):
        return (torch.randn(*shape, dtype=torch.float64) * 0.2).requires_grad_(grad)

    def test_grouped_empty_experts_and_gradients(self):
        x, w = self.rand(5, 3, grad=True), self.rand(4, 3, 2, grad=True)
        offsets = (0, 0, 2, 2, 5)
        expected = torch.stack([x[i] @ w[1 if i < 2 else 3] for i in range(5)])
        actual = grouped_gemm(x, w, offsets)
        torch.testing.assert_close(actual, expected)
        torch.autograd.gradcheck(lambda a, b: grouped_gemm(a, b, offsets), (x, w))

    def test_benchmark_distributions_and_reference_only_table(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from benchmark_flash_attention_runtime import comparison_table
        from benchmark_grouped_runtime import counts_for

        for rows, experts in ((512, 8), (4096, 8), (8192, 16)):
            for distribution in ("uniform", "hot", "empty"):
                counts = counts_for(rows, experts, distribution)
                self.assertEqual(sum(counts), rows)
                self.assertEqual(len(counts), experts)
                self.assertTrue(all(c >= 0 for c in counts))
        report = comparison_table(
            [
                dict(
                    size="B=1,T=128,H=4,D=512",
                    dtype="bf16",
                    operation="attention",
                    backend="reference",
                    median_ms=0.125,
                )
            ],
            ("size", "dtype", "operation"),
            lambda r: r["size"],
            lambda r: r["dtype"],
            "PyTorch",
        )
        cells = [x.strip() for x in report.splitlines()[-1].split("|")[1:-1]]
        self.assertEqual(cells[-3:], ["-", "125.00", "-"])
        self.assertEqual(len(set(map(len, report.splitlines()))), 1)

    def test_route_dispatch_combine_and_full_ffn(self):
        x = self.rand(4, 3)
        logits, bias = self.rand(4, 4), self.rand(4)
        ids, weights = route(logits, bias, 2, 1.5)
        torch.testing.assert_close(
            weights.sum(-1), torch.full((4,), 1.5, dtype=x.dtype)
        )
        packed, offsets, slots = dispatch(x, ids, 4)
        torch.testing.assert_close(packed[slots], x[:, None].expand(-1, 2, -1))
        self.assertEqual(offsets[-1], 8)
        up, gate, down = self.rand(4, 3, 5), self.rand(4, 3, 5), self.rand(4, 5, 3)
        expected = torch.zeros_like(x)
        for t in range(4):
            for j in range(2):
                e = ids[t, j]
                hidden = torch.nn.functional.silu(x[t] @ gate[e]) * (x[t] @ up[e])
                expected[t] += (hidden @ down[e]) * weights[t, j]
        torch.testing.assert_close(
            moe_forward(x, logits, bias, up, gate, down, 2, 1.5), expected
        )
        output = self.rand(8, 3, grad=True)
        weights = weights.requires_grad_()
        torch.autograd.gradcheck(lambda y, r: combine(y, slots, r), (output, weights))

    def test_routing_bias_and_groups(self):
        logits = torch.zeros(1, 8, dtype=torch.float64)
        bias = torch.tensor(
            [0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64
        )
        ids, weights = route(logits, bias, 2, groups=4, selected_groups=1)
        self.assertEqual(ids.tolist(), [[2, 3]])
        torch.testing.assert_close(weights, torch.full_like(weights, 0.5))
        self.assertEqual(route(logits, bias * 0, 2)[0].tolist(), [[0, 1]])

    def test_sparse_vs_dense_forward_backward_and_masked_rows(self):
        q, k, v = (
            self.rand(1, 3, 2, 3, grad=True),
            self.rand(1, 4, 3, grad=True),
            self.rand(1, 4, 2, grad=True),
        )
        ids = torch.tensor([[[-1, -1], [2, 0], [3, 1]]])
        mask = torch.tensor(
            [[[False] * 4, [True, False, True, False], [False, True, False, True]]]
        )
        for sink in (None, self.rand(2, grad=True)):
            a, lse = sparse_attention(q, k, v, ids, sink=sink)
            b, ref_lse = dense_attention(q, k, v, mask, sink=sink)
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(lse, ref_lse)
            variables = (q, k, v) if sink is None else (q, k, v, sink)
            ga = torch.autograd.grad(a.square().sum(), variables, retain_graph=True)
            gb = torch.autograd.grad(b.square().sum(), variables, retain_graph=True)
            for x, y in zip(ga, gb):
                self.assertTrue(torch.isfinite(x).all())
                torch.testing.assert_close(x, y)
        torch.autograd.gradcheck(
            lambda a, b, c: sparse_attention(a, b, c, ids)[0], (q, k, v)
        )

    def test_sink_zero_value_not_separate_softmax(self):
        q = torch.zeros(1, 1, 2, 2, dtype=torch.float64)
        kv = torch.ones(1, 1, 2, dtype=torch.float64)
        out, lse = sparse_attention(
            q, kv, kv, torch.zeros(1, 1, 1, dtype=torch.long), sink=torch.zeros(2)
        )
        torch.testing.assert_close(out, torch.full_like(out, 0.5))
        torch.testing.assert_close(lse, torch.full_like(lse, math.log(2)))
        out, _ = sparse_attention(
            q,
            kv,
            kv,
            torch.zeros(1, 1, 1, dtype=torch.long),
            sink=torch.full((2,), torch.inf),
        )
        self.assertTrue((out == 0).all())

    def test_compression_oracle_overlap_norm_and_rope(self):
        for overlap in (False, True):
            d, r, t = 3, 2, 7
            c = d * (2 if overlap else 1)
            x, g, bias, norm = (
                self.rand(1, t, c),
                self.rand(1, t, c),
                self.rand(r, c),
                self.rand(d),
            )
            expected = []
            for block in range(t // r):
                values, gates = [], []
                for offset in range(r):
                    values.append(x[0, block * r + offset, -d:])
                    gates.append(g[0, block * r + offset, -d:] + bias[offset, -d:])
                if overlap and block > 0:
                    for offset in range(r):
                        values.append(x[0, (block - 1) * r + offset, :d])
                        gates.append(
                            g[0, (block - 1) * r + offset, :d] + bias[offset, :d]
                        )
                weighted = (torch.stack(values) * torch.stack(gates).softmax(0)).sum(0)
                expected.append(
                    weighted / torch.sqrt(weighted.square().mean() + 1e-6) * norm
                )
            expected = torch.stack(expected)[None]
            torch.testing.assert_close(
                compress(x, g, bias, r, norm, overlap=overlap), expected
            )
            angles = self.rand(t, 1)
            actual = compress(x, g, bias, r, norm, overlap=overlap, angles=angles)
            torch.testing.assert_close(actual, rotary(expected, angles[:6:r]))
            torch.testing.assert_close(
                rotary(actual, angles[:6:r], inverse=True), expected
            )
            self.assertEqual(
                compress(x[:, :1], g[:, :1], bias, r, norm, overlap=overlap).shape,
                (1, 0, d),
            )

    def test_compressor_gradcheck(self):
        args = (
            self.rand(1, 4, 4, grad=True),
            self.rand(1, 4, 4, grad=True),
            self.rand(2, 4, grad=True),
            self.rand(2, grad=True),
        )
        torch.autograd.gradcheck(
            lambda x, g, p, w: compress(x, g, p, 2, w, overlap=True), args
        )

    def test_hca_visibility_joint_softmax_and_decode(self):
        q, local, compressed, sink = (
            self.rand(1, 7, 2, 3),
            self.rand(1, 7, 3),
            self.rand(1, 3, 3),
            self.rand(2),
        )
        out, lse = hca_forward(q, local, compressed, ratio=2, window=3, sink=sink)
        expected = []
        for t in range(7):
            cache = torch.cat(
                (local[:, max(0, t - 2) : t + 1], compressed[:, : (t + 1) // 2]), 1
            )
            expected.append(
                dense_attention(q[:, t : t + 1], cache, cache, sink=sink)[0]
            )
        torch.testing.assert_close(out, torch.cat(expected, 1))
        decode = hca_forward(
            q[:, -1:], local, compressed, ratio=2, window=3, start_pos=6, sink=sink
        )
        torch.testing.assert_close(decode[0], out[:, -1:])
        torch.testing.assert_close(decode[1], lse[:, :, -1:])
        changed = compressed.clone()
        changed[:, 1:] = 100
        torch.testing.assert_close(
            hca_forward(q, local, changed, ratio=2, window=3, sink=sink)[0][:, :3],
            out[:, :3],
        )
        self.assertEqual(
            compressed_indices(torch.arange(4), 2, 2).tolist(),
            [[-1, -1], [0, -1], [0, -1], [0, 1]],
        )

    def test_dsa_indexer_loop_and_dense_limit(self):
        q, kv, qi, ki, w = (
            self.rand(1, 5, 2, 6),
            self.rand(1, 5, 6),
            self.rand(1, 5, 3, 2),
            self.rand(1, 5, 2),
            self.rand(1, 5, 3),
        )
        visible = torch.ones(5, 5, dtype=torch.bool).tril()
        score = index_scores(qi, ki, w, visible)
        for t in range(5):
            for n in range(t + 1):
                expected = sum(
                    max(0, torch.dot(qi[0, t, h], ki[0, n])) * w[0, t, h]
                    for h in range(3)
                )
                torch.testing.assert_close(score[0, t, n], expected)
        output, lse, ids = dsa_forward(q, kv, qi, ki, w, 5, value_dim=4)
        expected, expected_lse = dense_attention(q, kv, kv[..., :4], visible)
        torch.testing.assert_close(output, expected)
        torch.testing.assert_close(lse, expected_lse)
        self.assertTrue(((ids < 0) | (ids <= torch.arange(5)[None, :, None])).all())
        selected = select_indices(score, 2)
        self.assertTrue(
            (
                score.gather(-1, selected.clamp_min(0))[:, 2:].sort(-1).values
                == score[:, 2:].topk(2, -1).values.sort(-1).values
            ).all()
        )

    def test_csa_dense_compressed_limit_and_empty_prefix(self):
        q, local, compressed = (
            self.rand(1, 7, 2, 3),
            self.rand(1, 7, 3),
            self.rand(1, 3, 3),
        )
        qi, ki, w, sink = (
            self.rand(1, 7, 2, 4),
            self.rand(1, 3, 4),
            self.rand(1, 7, 2),
            self.rand(2),
        )
        out, lse, _ = csa_forward(
            q, local, compressed, qi, ki, w, 3, ratio=2, window=3, sink=sink
        )
        expected, expected_lse = hca_forward(
            q, local, compressed, ratio=2, window=3, sink=sink
        )
        torch.testing.assert_close(out, expected)
        torch.testing.assert_close(lse, expected_lse)
        partial = csa_forward(
            q[:, :1],
            local[:, :1],
            compressed[:, :0],
            qi[:, :1],
            ki[:, :0],
            w[:, :1],
            3,
            ratio=2,
            window=3,
            sink=sink,
        )
        torch.testing.assert_close(partial[0], out[:, :1])
        decode = csa_forward(
            q[:, -1:],
            local,
            compressed,
            qi[:, -1:],
            ki,
            w[:, -1:],
            3,
            ratio=2,
            window=3,
            start_pos=6,
            sink=sink,
        )
        torch.testing.assert_close(decode[0], out[:, -1:])
        torch.testing.assert_close(decode[1], lse[:, :, -1:])
        # Attention loss does NOT differentiate through discrete index selection.
        qi.requires_grad_()
        local.requires_grad_()
        result = csa_forward(q, local, compressed, qi, ki, w, 1, ratio=2, window=3)[0]
        self.assertIsNone(torch.autograd.grad(result.sum(), qi, allow_unused=True)[0])

    def test_indexer_gradients_without_discrete_topk(self):
        q, k, w = (
            self.rand(1, 2, 2, 2, grad=True),
            self.rand(1, 3, 2, grad=True),
            self.rand(1, 2, 2, grad=True),
        )
        visible = torch.ones(2, 3, dtype=torch.bool)
        torch.autograd.gradcheck(
            lambda q, k, w: index_scores(q, k, w, visible), (q, k, w)
        )

    def test_kda_scalar_equation_state_carry_and_gradients(self):
        q, k, v = self.rand(1, 3, 1, 1), self.rand(1, 3, 1, 1), self.rand(1, 3, 1, 1)
        g, beta, state = (
            -self.rand(1, 3, 1, 1).abs(),
            torch.full((1, 3, 1), 0.7, dtype=q.dtype),
            self.rand(1, 1, 1, 1),
        )
        initial = state.clone()
        actual, final = kda_forward(q, k, v, g, beta, initial)
        expected = []
        for t in range(3):
            state = state * g[0, t, 0, 0].exp()
            state = state + beta[0, t, 0] * k[0, t, 0, 0] * (
                v[0, t, 0, 0] - k[0, t, 0, 0] * state
            )
            expected.append((q[0, t, 0, 0] * state).item())
        torch.testing.assert_close(
            actual.flatten(), torch.tensor(expected, dtype=q.dtype)
        )
        torch.testing.assert_close(final, state)
        saved = initial.clone()
        _, prefix = kda_forward(
            q[:, :1], k[:, :1], v[:, :1], g[:, :1], beta[:, :1], initial
        )
        suffix, suffix_state = kda_forward(
            q[:, 1:], k[:, 1:], v[:, 1:], g[:, 1:], beta[:, 1:], prefix
        )
        torch.testing.assert_close(suffix, actual[:, 1:])
        torch.testing.assert_close(suffix_state, final)
        torch.testing.assert_close(initial, saved)
        args = tuple(x.requires_grad_() for x in (q, k, v, g, beta, initial))
        torch.autograd.gradcheck(kda_forward, args)

    def test_kda_value_heads_and_gate_activation(self):
        q, k, v = self.rand(1, 3, 1, 2), self.rand(1, 3, 1, 2), self.rand(1, 3, 2, 3)
        rawg, rawb = self.rand(1, 3, 2, 2), self.rand(1, 3, 2)
        g, beta = activate_gates(rawg, rawb, self.rand(2), self.rand(4))
        out, state = kda_forward(q, k, v, g, beta)
        self.assertEqual(state.shape, (1, 2, 2, 3))
        for h in range(2):
            expected = kda_forward(
                q, k, v[:, :, h : h + 1], g[:, :, h : h + 1], beta[:, :, h : h + 1]
            )[0]
            torch.testing.assert_close(out[:, :, h : h + 1], expected)
        safe, _ = activate_gates(rawg, rawb, self.rand(2), self.rand(4), -5.0)
        self.assertTrue(((safe > -5) & (safe < 0)).all())


if __name__ == "__main__":
    unittest.main()
