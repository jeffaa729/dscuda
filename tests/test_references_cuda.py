"""CUDA parity and official FLA forward/backward checks; H100 adapters are opt-in."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from reference.python.attention import sparse_attention
from reference.python.compression import compress
from reference.python.flashmla import flatten_sparse_indices, pack_cache
from reference.python.kda import fla_forward, kda_forward


class LayoutTests(unittest.TestCase):
    def test_external_adapters_never_silently_fallback(self):
        from reference.python.deepgemm import DeepGEMMGrouped
        from reference.python.flashmla import FlashMLADecode, FlashMLASparse

        q = torch.zeros(1, 1, 64, 576, dtype=torch.bfloat16)
        kv = torch.zeros(1, 128, 576, dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "SM90"):
            FlashMLADecode(q, kv, torch.tensor([128]))
        with self.assertRaisesRegex(RuntimeError, "SM90"):
            FlashMLASparse(q, kv, torch.zeros(1, 1, 128, dtype=torch.int32))
        with self.assertRaisesRegex(RuntimeError, "SM90"):
            DeepGEMMGrouped(kv, kv, torch.tensor([128]), 128)

    def test_paged_cache_and_batch_indices(self):
        kv = torch.arange(2 * 65 * 3).reshape(2, 65, 3)
        cache, table = pack_cache(kv)
        self.assertEqual(cache.shape, (4, 64, 1, 3))
        self.assertEqual(table.tolist(), [[0, 1], [2, 3]])
        torch.testing.assert_close(cache.reshape(2, 128, 3)[:, :65], kv)
        self.assertTrue((cache.reshape(2, 128, 3)[:, 65:] == 0).all())
        ids = torch.tensor([[[0, 64, -1]], [[0, 64, 65]]])
        self.assertEqual(
            flatten_sparse_indices(ids, 65).tolist(), [[[0, 64, -1]], [[65, 129, -1]]]
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def test_hca_dsa_csa_pipeline_cpu_cuda(self):
        from reference.python.csa import csa_forward
        from reference.python.dsa import dsa_forward
        from reference.python.hca import hca_forward

        q = torch.randn(2, 9, 2, 8, dtype=torch.bfloat16) * 0.1
        local = torch.randn(2, 9, 8, dtype=torch.bfloat16) * 0.1
        compressed = torch.randn(2, 4, 8, dtype=torch.bfloat16) * 0.1
        qi, ki, w = torch.rand(2, 9, 2, 4), torch.rand(2, 9, 4), torch.rand(2, 9, 2)
        sink = torch.rand(2)

        def call(family, args):
            q, local, compressed, qi, ki, w, sink = args
            if family == "hca":
                return hca_forward(q, local, compressed, ratio=2, window=3, sink=sink)[
                    :2
                ]
            if family == "dsa":
                return dsa_forward(q, local, qi, ki, w, 3, value_dim=6)[:2]
            return csa_forward(
                q, local, compressed, qi, ki[:, :4], w, 2, ratio=2, window=3, sink=sink
            )[:2]

        for family in ("hca", "dsa", "csa"):
            cpu = tuple(
                x.detach().requires_grad_()
                for x in (q, local, compressed, qi, ki, w, sink)
            )
            gpu = tuple(x.detach().cuda().requires_grad_() for x in cpu)
            expected, actual = call(family, cpu), call(family, gpu)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a.cpu(), e, atol=1e-5, rtol=2e-4)
            a_grad = torch.autograd.grad(
                actual[0].square().sum(), gpu[:3], allow_unused=True
            )
            e_grad = torch.autograd.grad(
                expected[0].square().sum(), cpu[:3], allow_unused=True
            )
            for a, e in zip(a_grad, e_grad):
                if e is None:
                    self.assertIsNone(a)
                else:
                    torch.testing.assert_close(
                        a.cpu().float(), e.float(), atol=2e-3, rtol=2e-2
                    )

    def test_compression_and_sparse_cpu_cuda_gradients(self):
        x = torch.randn(1, 9, 16, dtype=torch.bfloat16).float().requires_grad_()
        g = torch.randn_like(x).requires_grad_()
        bias, norm = torch.randn(4, 16), torch.randn(8)
        q = torch.randn(1, 9, 2, 8).requires_grad_()
        ids = torch.tensor([[[-1, -1]] * 3 + [[0, -1]] * 4 + [[0, 1]] * 2])

        def run(x, g, q, bias, norm, ids):
            kv = compress(x, g, bias, 4, norm, overlap=True)
            return sparse_attention(q, kv, kv, ids)[0]

        inputs = (x, g, q)
        expected = run(*inputs, bias, norm, ids)
        reference_grads = torch.autograd.grad(expected.square().sum(), inputs)
        gpu = tuple(t.detach().cuda().requires_grad_() for t in inputs)
        output = run(*gpu, bias.cuda(), norm.cuda(), ids.cuda())
        grads = torch.autograd.grad(output.square().sum(), gpu)
        for a, b in zip((output, *grads), (expected, *reference_grads)):
            torch.testing.assert_close(a.cpu(), b, atol=2e-5, rtol=2e-4)

    def test_fla_kda_forward_backward_and_carried_state(self):
        # T crosses a chunk boundary; GVA and nonzero initial state are exercised.
        b, t, h, hv, k, v = 1, 65, 2, 4, 32, 32
        q = torch.nn.functional.normalize(
            torch.randn(b, t, h, k, device="cuda"), dim=-1
        ).bfloat16()
        key = torch.nn.functional.normalize(
            torch.randn_like(q.float()), dim=-1
        ).bfloat16()
        value = (torch.randn(b, t, hv, v, device="cuda") * 0.2).bfloat16()
        g = -torch.rand(b, t, hv, k, device="cuda") * 0.1
        beta = torch.rand(b, t, hv, device="cuda") * 0.8
        initial = torch.randn(b, hv, k, v, device="cuda") * 0.05
        inputs = tuple(x.requires_grad_() for x in (q, key, value, g, beta, initial))
        expected = kda_forward(*inputs)
        actual = fla_forward(*inputs)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a.float(), e.float(), atol=1e-3, rtol=2e-2)
        do, ds = torch.randn_like(actual[0]), torch.randn_like(actual[1]) * 0.1
        expected_grads = torch.autograd.grad(expected, inputs, (do, ds))
        actual_grads = torch.autograd.grad(actual, inputs, (do, ds))
        for a, e in zip(actual_grads, expected_grads):
            torch.testing.assert_close(a.float(), e.float(), atol=3e-3, rtol=4e-2)
        _, middle = fla_forward(
            *(x[:, :32].detach() for x in inputs[:5]), initial.detach()
        )
        suffix, state = fla_forward(*(x[:, 32:].detach() for x in inputs[:5]), middle)
        torch.testing.assert_close(
            suffix.float(), actual[0][:, 32:].float(), atol=1e-3, rtol=2e-2
        )
        torch.testing.assert_close(state, actual[1], atol=1e-3, rtol=2e-2)


@unittest.skipUnless(
    os.environ.get("DSCUDA_TEST_H100_REFERENCES") == "1",
    "set DSCUDA_TEST_H100_REFERENCES=1 on configured H100",
)
class H100ReferenceTests(unittest.TestCase):
    def test_flashmla_decode(self):
        from reference.python.attention import dense_attention
        from reference.python.flashmla import FlashMLADecode

        q = torch.randn(2, 1, 16, 576, device="cuda", dtype=torch.bfloat16) * 0.1
        kv = torch.randn(2, 129, 576, device="cuda", dtype=torch.bfloat16) * 0.1
        lengths = torch.tensor([129, 71], device="cuda", dtype=torch.int32)
        mask = (torch.arange(129, device="cuda")[None] < lengths[:, None])[:, None]
        expected = dense_attention(q, kv, kv[..., :512], mask)
        call = FlashMLADecode(q, kv, lengths)
        for _ in range(2):
            actual = call()
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a.float(), e.float(), atol=2e-3, rtol=2e-2)

    def test_flashmla_sparse(self):
        from reference.python.flashmla import FlashMLASparse

        for d in (512, 576):
            q = torch.randn(2, 16, 64, d, device="cuda", dtype=torch.bfloat16) * 0.1
            kv = torch.randn(2, 128, d, device="cuda", dtype=torch.bfloat16) * 0.1
            ids = torch.arange(128, device="cuda")[None, None].expand(2, 16, -1).clone()
            ids[0, 0, 1:] = -1
            sink = torch.randn(64, device="cuda")
            expected = sparse_attention(q, kv, kv[..., :512], ids, sink=sink)
            actual = FlashMLASparse(q, kv, ids, sink=sink)()
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a.float(), e.float(), atol=2e-3, rtol=2e-2)

    def test_deepgemm_grouped(self):
        from reference.python.deepgemm import DeepGEMMGrouped, masked_grouped_reference

        x = torch.randn(4, 128, 256, device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.randn(4, 256, 256, device="cuda", dtype=torch.bfloat16) * 0.1
        counts = torch.tensor([128, 71, 3, 0], device="cuda", dtype=torch.int32)
        expected = masked_grouped_reference(x, w, counts)
        call = DeepGEMMGrouped(x, w, counts, 64)
        valid = torch.arange(128, device="cuda")[None] < counts[:, None]
        for _ in range(2):
            torch.testing.assert_close(
                call()[valid].float(), expected[valid].float(), atol=2e-3, rtol=2e-2
            )


if __name__ == "__main__":
    unittest.main()
