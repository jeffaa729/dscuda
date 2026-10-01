"""Small CPU checks for the forward-only BF16 MoE contract."""

import unittest

import torch
import torch.nn.functional as F

from reference.python.moe import moe_forward


class MoEReferenceTests(unittest.TestCase):
    def test_uniform_and_skewed_routing(self):
        torch.manual_seed(7)
        x = torch.randn(5, 8, dtype=torch.bfloat16)
        w1 = torch.randn(4, 8, 8, dtype=torch.bfloat16) / 8 ** .5
        w2 = torch.randn(4, 8, 4, dtype=torch.bfloat16) / 4 ** .5
        weights = torch.rand(5, 2)
        ids_cases = (torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]]),
                     torch.tensor([[0, 1], [0, 2], [0, 3], [0, 1], [0, 2]]))
        for ids in ids_cases:
            expected = torch.zeros_like(x, dtype=torch.float32)
            for token in range(x.shape[0]):
                for slot in range(ids.shape[1]):
                    expert = ids[token, slot]
                    gate_up = (x[token].float() @ w1[expert].float().T).bfloat16().float()
                    hidden = w2.shape[-1]
                    activation = (F.silu(gate_up[:hidden].clamp(max=10)) *
                                  gate_up[hidden:].clamp(-10, 10) * weights[token, slot]).bfloat16()
                    expected[token] += (activation.float() @ w2[expert].float().T).bfloat16().float()
            torch.testing.assert_close(moe_forward(x, ids, weights, w1, w2), expected.bfloat16())


if __name__ == "__main__":
    unittest.main()
