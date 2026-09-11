"""Official FlashMLA dense decode adapter."""

import torch
from flash_mla import flash_mla_with_kvcache, get_mla_metadata


def load_decode(device=None):
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise RuntimeError("FlashMLA dense BF16 decode requires SM90 (H100/H800); use --reference pytorch here.")
    return flash_mla_with_kvcache, get_mla_metadata


def require_hopper(x):
    if not x.is_cuda or torch.cuda.get_device_capability(x.device)[0] not in (9, 10):
        raise RuntimeError(
            "FlashMLA requires supported SM90/SM100 hardware and an upstream installation"
        )
    if x.dtype != torch.bfloat16:
        raise ValueError("this adapter requires BF16 inputs")


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
