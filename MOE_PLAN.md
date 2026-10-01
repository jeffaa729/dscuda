# Multi-GPU Hopper MegaMoE Plan

## Goal and scope

Build a forward-only BF16 MoE pipeline for **2–4 H100 SXM GPUs**. The final kernel should overlap expert-parallel dispatch/combine over NVLink with two expert GEMMs and SwiGLU. Use [DeepGEMM's Blackwell BF16 MegaMoE](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh), its [scheduler](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh), and [buffer layout](https://github.com/deepseek-ai/DeepGEMM/blob/main/deep_gemm/include/deep_gemm/layout/mega_moe.cuh) as the architectural sources. [Hopper PR #323](https://github.com/deepseek-ai/DeepGEMM/pull/323) is secondary guidance, not the implementation or performance target.

Inputs are local BF16 tokens `[T,H]`, precomputed global expert IDs `[T,topk]`, and FP32 route weights `[T,topk]`. Each rank owns `E/ranks` experts with BF16 weights `W1[E/ranks,2I,H]` and `W2[E/ranks,H,I]`. Compute `gate,up = x @ W1.T`, then `BF16(silu(clamp(gate,max=10)) * clamp(up,-10,10) * route_weight) @ W2.T`; combine top-k expert outputs on the source rank into BF16 `[T,H]`. Router selection, shared experts, backward, FP8, and FP4 are out of scope. Start with `H=1024,I=512,E=16,topk=4`; sweep 16–4096 tokens per rank and add a larger shape after correctness.

## Milestones

1. **Contract and oracle:** one compact PyTorch reference using the contract above; check one ordinary and one skewed/uneven routing case. No large numerical-test matrix.
2. **Unfused multi-GPU baseline:** one process per GPU; DeepEP `ElasticBuffer.dispatch` and `.combine`, two DeepGEMM SM90 grouped BF16 NT GEMMs, and a simple SwiGLU operation. Run on 2 H100s; repeat on 4. This is the primary latency reference.
3. **Hopper translation:** implement symmetric peer buffers, token/source metadata, bounded expert pools, persistent one-CTA-per-SM scheduling, and explicit arrival/readiness counters. Replace Blackwell 2-CTA `tcgen05`/TMEM with Hopper TMA, WGMMA, and register accumulators. Validate each phase before overlap.
4. **Fused multi-GPU path:** overlap remote token pulls with expert compute; fuse L1 GEMM epilogue with SwiGLU; run L2; push output to source-rank combine slots. Prove repeated runs do not hang.
5. **Evaluation:** compare custom fused versus the milestone-2 baseline on the same 2- and 4-GPU nodes, using decode-like and larger token loads. Include a no-overlap ablation if practical; report regressions honestly.

## Measurement and handoff

Begin timing from identical local input tensors. Include all per-iteration preparation, dispatch, expert work, and combine; exclude allocation and one-time weight transformation. Report measured microseconds and baseline/custom speedup. Use Nsight Systems to support overlap claims; do not present runtime-derived bandwidth as a measured counter. For multi-rank timing, synchronize before each trial and report the slowest rank. Require same-node NVLink peer access (`nvidia-smi topo -m`) before renting time for overlap experiments.

Local editing may use the RTX 4060, but H100 compilation, correctness, and benchmarking are a **manual remote gate**: the user connects to the rented machine, runs the provided commands, and returns build errors/results. Do not claim a milestone is GPU-verified until those results are seen.

## Manual H100 gate

Use 2 or 4 H100 SXM GPUs on one NVLink-connected node. The current Docker image includes DeepGEMM but **not DeepEP**. Its locked PyTorch 2.11 CUDA 12.8 environment ships NCCL 2.28.9, while official [DeepEP V2](https://github.com/deepseek-ai/DeepEP#quick-start) requires NCCL 2.30.4 or newer. On the remote node, install `nvidia-nccl-cu12==2.30.4` into `DSCUDA_PYTHON` with `--no-deps`, then build/install DeepEP V2 against that environment. This overrides PyTorch's exact NCCL dependency; verify distributed operation before trusting results, and do not run `uv sync --locked` afterward because it restores 2.28.9. A missing DeepEP module or NCCL mismatch is an environment failure, not a kernel failure.

```bash
nvidia-smi topo -m
uv pip install --python "$DSCUDA_PYTHON" --no-deps --index-url https://pypi.org/simple nvidia-nccl-cu12==2.30.4
git clone --recursive https://github.com/deepseek-ai/DeepEP.git /tmp/DeepEP
TORCH_CUDA_ARCH_LIST=9.0 uv pip install --python "$DSCUDA_PYTHON" --no-build-isolation --no-deps /tmp/DeepEP
$DSCUDA_PYTHON -c 'import torch; print("NCCL", torch.cuda.nccl.version())'
$DSCUDA_PYTHON -c 'import torch, deep_ep, deep_gemm; print(torch.cuda.device_count())'
bash scripts/benchmark_moe.sh 2 --tokens 16 --check-only
bash scripts/benchmark_moe.sh 2 --check
# On a separate 4-GPU NVLink node:
bash scripts/benchmark_moe.sh 4 --check
```

Run the correctness gate first, then retain the benchmark table and any error trace for review. This initial baseline uses CUDA events around the whole Python-launched sequence; later fused comparisons must use the same launch/timing mode, or rerun both under CUDA Graph capture. The script does not claim graph-replay or isolated kernel latency.
