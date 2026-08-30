"""Validate the PyTorch MLA equations independently with packed Q/K and autograd."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference/python"))
from mla import mla_forward, mla_backward


class MlaReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def inputs(self, batch=2, query=5, sequence=5, heads=3, rank=7, rope=4):
        torch = self.torch
        torch.manual_seed(41)
        shapes = ((batch, query, heads, rank), (batch, query, heads, rope),
                  (batch, sequence, rank), (batch, sequence, rope))
        return tuple((torch.randn(s, dtype=torch.float64) * .3).requires_grad_() for s in shapes)

    def packed_forward(self, inputs, scale, mask):
        torch = self.torch
        qc, qr, latent, kr = inputs
        q = torch.cat((qc, qr), dim=-1).transpose(1, 2)
        k = torch.cat((latent, kr), dim=-1)[:, None]
        score = (q @ k.transpose(-1, -2) * scale).masked_fill(~mask, float("-inf"))
        output = (score.softmax(dim=-1) @ latent[:, None]).transpose(1, 2).contiguous()
        return output, score.logsumexp(dim=-1)

    def test_forward_and_shared_kv_gradients(self):
        torch = self.torch
        inputs = self.inputs()
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        expected, expected_lse = self.packed_forward(inputs, .31, mask)
        actual, lse = mla_forward(*inputs, .31, mask)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(lse, expected_lse, atol=1e-12, rtol=1e-12)
        dout = torch.randn_like(expected)
        gradients = torch.autograd.grad(expected, inputs, dout)
        analytical = mla_backward(dout, actual, lse, *inputs, .31, mask)
        for value, reference in zip(analytical, gradients):
            torch.testing.assert_close(value, reference, atol=1e-11, rtol=1e-11)

    def test_future_tokens_cannot_change_prefix(self):
        torch = self.torch
        inputs = self.inputs()
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        original = mla_forward(*inputs, .31, mask)[0]
        changed = tuple(t.detach().clone() for t in inputs)
        changed[2][:, 3:].fill_(100)
        changed[3][:, 3:].fill_(-100)
        actual = mla_forward(*changed, .31, mask)[0]
        torch.testing.assert_close(actual[:, :3], original[:, :3], atol=0, rtol=0)

    def test_decode_unequal_lengths(self):
        torch = self.torch
        inputs = self.inputs(query=1, sequence=9)
        lengths = (1, 9)
        mask = (torch.arange(9)[None, :] < torch.tensor(lengths)[:, None])[:, None, None, :]
        output, lse = mla_forward(*inputs, .31, mask)
        for batch, length in enumerate(lengths):
            local = tuple(t[batch:batch+1] if index < 2 else t[batch:batch+1, :length]
                          for index, t in enumerate(inputs))
            expected, expected_lse = self.packed_forward(
                local, .31, torch.ones(1, length, dtype=torch.bool))
            torch.testing.assert_close(output[batch:batch+1], expected, atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(lse[batch:batch+1], expected_lse, atol=1e-12, rtol=1e-12)

    def test_bf16_storage_fp32_results(self):
        torch = self.torch
        inputs = tuple(t.detach().to(torch.bfloat16) for t in self.inputs())
        mask = torch.ones(5, 5, dtype=torch.bool).tril()
        output, lse = mla_forward(*inputs, .31, mask)
        gradients = mla_backward(torch.ones_like(output), output, lse, *inputs, .31, mask)
        self.assertTrue(all(t.dtype == torch.float32 for t in (output, lse, *gradients)))
        self.assertTrue(all(t.is_contiguous() for t in (output, lse, *gradients)))


if __name__ == "__main__":
    unittest.main()
