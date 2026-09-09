"""Shared precision helper for Python mathematical references."""

import torch


def compute(x):
    return x if x.dtype == torch.float64 else x.float()
