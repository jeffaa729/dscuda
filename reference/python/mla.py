"""Unfused absorbed-query MLA with BF16 activations and FP32 math/LSE."""


def _compute(tensor):
    # Keep double precision available for independent CPU/autograd tests.
    return tensor if tensor.element_size() >= 4 else tensor.float()


def _scores(query, query_rope, latent, key_rope, scale, mask):
    import torch

    score = (torch.einsum("bqhc,bkc->bhqk", query, latent)
             + torch.einsum("bqhr,bkr->bhqk", query_rope, key_rope)) * scale
    return score.masked_fill(~mask, float("-inf"))


def mla_forward(query, query_rope, latent, key_rope, scale, mask):
    import torch

    query, query_rope, latent, key_rope = map(
        _compute, (query, query_rope, latent, key_rope))
    score = _scores(query, query_rope, latent, key_rope, scale, mask)
    lse = score.logsumexp(dim=-1)
    probability = (score - lse.unsqueeze(-1)).exp()
    output = torch.einsum("bhqk,bkc->bqhc", probability, latent)
    return output.bfloat16().contiguous(), lse.contiguous()


def mla_backward(dout, output, lse, query, query_rope, latent, key_rope, scale, mask):
    import torch

    dout, output, query, query_rope, latent, key_rope = map(
        _compute, (dout, output, query, query_rope, latent, key_rope))
    score = _scores(query, query_rope, latent, key_rope, scale, mask)
    probability = (score - lse.unsqueeze(-1)).exp()
    dprobability = torch.einsum("bqhc,bkc->bhqk", dout, latent)
    delta = (dout * output).sum(dim=-1).transpose(1, 2).unsqueeze(-1)
    dscore = probability * (dprobability - delta) * scale
    dquery = torch.einsum("bhqk,bkc->bqhc", dscore, latent)
    dquery_rope = torch.einsum("bhqk,bkr->bqhr", dscore, key_rope)
    # Shared latent gradients include both the score/key path and the value path.
    dlatent = (torch.einsum("bhqk,bqhc->bkc", dscore, query)
               + torch.einsum("bhqk,bqhc->bkc", probability, dout))
    dkey_rope = torch.einsum("bhqk,bqhr->bkr", dscore, query_rope)
    return tuple(t.bfloat16().contiguous()
                 for t in (dquery, dquery_rope, dlatent, dkey_rope))
