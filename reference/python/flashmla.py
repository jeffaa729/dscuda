"""Optional official FlashMLA adapters; no silent PyTorch fallback.

Dense decode uses paged BF16 C512/R64 storage and SM90. Sparse prefill uses
the upstream BF16 MQA interface on SM90/SM100. Both return BF16 output.
"""

import torch


def load_decode(device=None):
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError("FlashMLA dense BF16 decode requires SM90 (H100/H800); use --reference pytorch here.")
    try:
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    except ImportError as error:
        raise RuntimeError(
            "Install FlashMLA in this uv environment: uv pip install --no-build-isolation "
            "git+https://github.com/deepseek-ai/FlashMLA.git@15f13e5030374295491c5ce31b02d7e63a7772c6"
        ) from error
    return flash_mla_with_kvcache, get_mla_metadata


def require_hopper(x):
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] not in (9, 10):
        raise RuntimeError(
            "FlashMLA requires supported SM90/SM100 hardware and an upstream installation"
        )
    if x.dtype != torch.bfloat16:
        raise ValueError("this adapter requires BF16 inputs")


def pack_cache(kv, page_size=64):
    """[B,N,576] -> [B*ceil(N/P),P,1,576] plus a disjoint int32 page table.

    Run outside timing; original sequence lengths exclude the zero padding.
    No cache is shared between batches.
    """
    b, n, d = kv.shape
    pages = (n + page_size - 1) // page_size
    padded = torch.nn.functional.pad(kv, (0, 0, 0, pages * page_size - n))
    table = torch.arange(b * pages, device=kv.device, dtype=torch.int32).reshape(
        b, pages
    )
    return padded.reshape(b * pages, page_size, 1, d).contiguous(), table


def flatten_sparse_indices(indices, kv_length):
    valid = (indices >= 0) & (indices < kv_length)
    offsets = (
        torch.arange(indices.shape[0], device=indices.device)[:, None, None] * kv_length
    )
    return (
        torch.where(valid, indices + offsets, -1)
        .reshape(-1, 1, indices.shape[-1])
        .int()
        .contiguous()
    )


class FlashMLADecode:
    """Fixed-shape decode; the runner's untimed check initializes scheduler metadata."""

    def __init__(self, q, cache, block_table, lengths, scale=None):
        require_hopper(q)
        self.call, get_mla_metadata = load_decode(q.device)
        if q.shape[-1] != 576 or cache.shape[-1] != 576 or q.shape[1] != 1:
            raise ValueError("dense decode requires Q=1 and packed C512/R64 Q/KV")
        if cache.dtype != q.dtype or cache.device != q.device:
            raise ValueError("Q/KV must share device and BF16 dtype")
        if block_table.shape[0] != q.shape[0]:
            raise ValueError("block table and Q must share batch size")
        capacity = block_table.shape[1] * cache.shape[1]
        if lengths.shape != (q.shape[0],) or not bool(
            ((lengths > 0) & (lengths <= capacity)).all()
        ):
            raise ValueError(
                "cache lengths must be positive and within each packed sequence"
            )
        self.q = q.contiguous()
        self.cache = (
            cache.contiguous(),
            block_table.to(device=q.device, dtype=torch.int32).contiguous(),
        )
        self.lengths = lengths.to(device=q.device, dtype=torch.int32).contiguous()
        self.metadata, self.splits = get_mla_metadata()
        self.scale = scale

    def __call__(self):
        cache, table = self.cache
        return self.call(
            self.q,
            cache,
            table,
            self.lengths,
            512,
            self.metadata,
            self.splits,
            softmax_scale=self.scale,
            causal=False,
            is_fp8_kvcache=False,
        )


class FlashMLASparse:
    """Prepared sparse prefill with flattened batches; selection is outside timing.

    Q/KV width is 576 (DSA) or 512 (V4); values are the first 512 channels.
    Converts the upstream sink-excluding LSE to our sink-inclusive convention.
    """

    def __init__(self, q, kv, indices, scale=None, sink=None):
        require_hopper(q)
        if q.shape[-1] not in (512, 576) or kv.shape[-1] != q.shape[-1]:
            raise ValueError("FlashMLA sparse requires Dqk=512/576 and Dv=512")
        if kv.dtype != q.dtype or kv.device != q.device or kv.shape[0] != q.shape[0]:
            raise ValueError("Q/KV must share batch size, device and BF16 dtype")
        if (
            q.shape[2] not in (64, 128)
            or indices.shape[-1] == 0
            or indices.shape[-1] % 128
        ):
            raise ValueError(
                "this adapter targets upstream H=64/128 and topk multiples of 128"
            )
        from flash_mla import flash_mla_sparse_fwd

        self.call = flash_mla_sparse_fwd
        self.shape = q.shape[:3]
        self.q = q.flatten(0, 1).contiguous()
        self.kv = kv.flatten(0, 1).unsqueeze(1).contiguous()
        self.indices = flatten_sparse_indices(indices, kv.shape[1])
        self.scale = q.shape[-1] ** -0.5 if scale is None else scale
        self.sink = (
            None
            if sink is None
            else sink.to(device=q.device, dtype=torch.float32).contiguous()
        )

    def __call__(self):
        out, _, lse = self.call(
            self.q, self.kv, self.indices, self.scale, 512, self.sink
        )
        if self.sink is not None:
            lse = torch.logaddexp(lse, self.sink[None])
        return out.reshape(*self.shape, 512), lse.reshape(*self.shape).transpose(1, 2)
