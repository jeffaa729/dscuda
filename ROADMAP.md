# dscuda Roadmap

## Project goal

Build a CUDA-first LLM training and kernel-engineering project. The main result
is a set of correct forward/backward kernels whose performance is measured
against production libraries; small dense GPT, DeepSeek-V3-style, and
V4-style models demonstrate that the kernels compose into a real training
system.

Serving-system work such as paged allocation, continuous batching, prefix
sharing, request scheduling, and production KV-cache management belongs in a
separate future inference-engine project. Generation in this repository is a
functional integration test, not a serving benchmark.

## Benchmark contract

Every optimized-kernel comparison must:

- use the same GPU, tensor shapes, layout, datatype, causal mask, scale, and
  input data for the custom and reference implementations;
- validate outputs before profiling and validate gradients for training
  kernels;
- exclude allocation, host-device copies, and datatype/layout conversion from
  the measured operation unless both implementations intrinsically need them;
- include every kernel required to finish the operation;
- profile both implementations with the same Nsight Compute settings and
  overwrite the previous `.ncu-rep` report on every run;
- extract latency, TFLOP/s or effective bandwidth, registers per thread,
  shared memory, occupancy, cache behavior, and spills into a concise table;
- record the GPU, CUDA version, reference commit, compiler flags, and shape.

Relative performance is:

```text
reference latency / custom latency * 100%
```

The optimization target is at least 80% over a representative shape set, not
one selected shape. An 80% result means the custom kernel takes no more than
1.25x the reference latency.

## Phase 1: Completed training foundation

- [x] Implement shared tensor, allocator, checkpoint, tokenizer, dataset, and
  CUDA utility code.
- [x] Implement and test embedding, RMSNorm, RoPE, residual, SwiGLU,
  cross-entropy, gradient clipping, AdamW, and matrix multiplication.
- [x] Assemble, train, checkpoint, resume, and sample from a dense GPT model.
- [x] Add FP32 and BF16 paths, Tensor Core GEMM, FLOPs/token, TFLOP/s,
  tokens/s, and MFU reporting.
- [x] Implement composed causal attention and fused FlashAttention-style
  forward/backward paths with online softmax and backward recomputation.
- [x] Implement MLA forward/backward, split-KV compressed decode, DeepSeekMoE,
  routing-bias balancing, shared experts, and sequential MTP.
- [x] Train and generate text with parameter-matched dense GPT and
  DeepSeek-V3-style TinyStories models.

## Phase 2: Unified comparison harness

- [x] Define deterministic benchmark cases shared by custom and external
  implementations.
- [x] Add small CPU or PyTorch correctness cases and larger profiling cases.
- [x] Separate training/prefill workloads from decode workloads.
- [x] Produce one CSV and one concise Markdown table per kernel family.
- [x] Add an aggregate report containing correctness, relative performance,
  and the primary Nsight Compute bottleneck for every kernel.

Initial attention workloads:

```text
training/prefill:
  B = 1, 4
  T = 128, 256, 512, 1024, 2048
  H = 4, 8
  D = 64, 128

decode:
  B = 1, 8, 32
  Q = 1
  KV = 128, 512, 2048, 8192, 32768
```

Primary attention comparisons use BF16 inputs with FP32 softmax or MMA
accumulation unless the reference contract requires otherwise.

## Phase 3: GEMM versus cuBLAS

- [x] Implement FP32 tiled GEMM and BF16 Tensor Core GEMM.
- [x] Add CPU references, forward/backward tests, and a cuBLAS benchmark.
- [x] Test square matrices of size 2048, 4096, and 8192.
- [x] Move results into the unified report format.
- [x] Record final latency, TFLOP/s, relative cuBLAS performance, tile
  configuration, registers, shared memory, occupancy, and spills.
- [x] Freeze this family after reaching the 80% target or clearly documenting
  the remaining hardware bottleneck.

The full SM89 sweep covers forward and both backward GEMMs at 2048, 4096, and
8192. Median relative performance is 100.4% for FP32 and 91.0% for BF16, while
the weakest transposed/backward cases fall to 38.0% and 61.7% respectively.
Nsight Compute reports register pressure across the family; the large
transpose cases also expose the current tile/layout sensitivity. GEMM is
therefore frozen with that limitation documented rather than adding more
specialized kernels now.

## Phase 4: CUDA FlashAttention versus official FlashAttention

- [x] Implement causal BF16 FlashAttention-style forward and backward.
- [x] Integrate fused attention into dense GPT training.
- [x] Validate output, log-sum-exp, and Q/K/V gradients.
- [x] Add an adapter for the official FlashAttention-2 Python package.
- [x] Add identical causal BF16 forward and backward benchmark cases for head
  dimensions 64 and 128 over the shared prefill shape matrix.
- [x] Capture and report complete forward and backward operations separately.
- [ ] Run the official comparison matrix on the same local or rented GPU and
  record the exact FlashAttention commit and environment.
- [ ] Optimize register pressure, K/V load pipelining, Tensor Core shape
  coverage, and avoidable conversions until reaching 80% or documenting the
  limiting resource.

The local SM89 RTX 4060 is the development GPU. A local quick correctness and
kernel-time snapshot against flash-attn 2.8.3.post1 passes for D64 and D128. At
`B=1,T=512,H=8`, the custom D64 forward/backward take 31.8/162.5 us versus
417.3/713.8 us for the official kernels; the generic custom D128 path takes
1447.7/3464.5 us versus 508.9/1127.2 us. These Nsight Systems timings identify
D128 Tensor Core coverage as the immediate gap; the full Nsight Compute matrix
and exact upstream commit remain open.

The final official comparison may run locally or on a rented H100 using the
`h100` suite, but custom and official timings must come from that same machine.
Record the exact reference commit.

## Phase 5: Frozen CUDA MLA implementation

- [x] Implement MLA CPU references, training forward/backward, and compressed
  cached decode.
- [x] Implement an SM89 BF16 Tensor Core path for the small model shape.
- [x] Validate sequential compressed-cache decode against full causal MLA.
- [x] Profile training forward/backward and compressed cached decode with the
  unified Nsight Compute table.
- [x] Document compressed-cache bytes per token and validate token-by-token
  decode against full causal MLA.

MLA is frozen at this point. Generalizing to FlashMLA production layouts and a
same-H100 comparison is a deferred extension, not a gate for phases 6 or 7.
If resumed, both implementations must run on the same SM90 machine; local RTX
4060 timings must never be compared with published H800 results.

## Phase 6: Secondary MoE kernel demonstration

- [x] Implement and test sigmoid top-k routing, no-drop dispatch, grouped
  routed experts, shared experts, combine, backward, and bias updates.
- [x] Replace serial dispatch-map construction with parallel histograms,
  prefix offsets, and token permutation.
- [x] Implement and CPU-check BF16 Tensor Core grouped GEMM for variable expert
  loads, including an empty expert.
- [x] Add uniform, skewed, and hot-expert grouped-GEMM benchmark cases.
- [x] Compare the same grouped rows and BF16 operands against per-expert
  cuBLAS Tensor Core GEMMs.
- [x] Retain complete FP32 MoE forward/backward CPU-CUDA integration tests.

This secondary demonstration is deliberately scoped to parallel dispatch and
grouped GEMM. Fusing gate/up projection with the SwiGLU epilogue and comparing
a complete production MoE contract with CUTLASS, FlashInfer, or DeepGEMM are
optional H100 follow-ups and must not delay the attention work.

## Phase 7: HCA, DSA, and CSA CUDA kernels

### Shared token compression

- [ ] Implement token-wise KV compression forward/backward with a CPU or
  PyTorch reference.
- [ ] Benchmark compression independently before attention fusion.

### Heavily Compressed Attention

- [ ] Implement HCA over non-overlapping compressed KV blocks.
- [ ] Add the uncompressed sliding-window branch for local dependencies and
  strict causality inside a compressed block.
- [ ] Add the selected V4-style query/KV normalization, partial RoPE,
  inverse-position output RoPE, grouped output projection, and attention sinks.
- [ ] Implement and validate HCA backward.
- [ ] Compare with the same PyTorch equations locally and a matching FlashInfer
  HCA implementation on rented supported hardware.

### DeepSeek Sparse Attention

- [ ] Implement lightning-indexer logits.
- [ ] Implement GPU top-k selection and packed sparse indices.
- [ ] Implement indexed sparse MLA forward with online softmax.
- [ ] Implement sparse backward with indexed gradient scatter/reduction.
- [ ] Benchmark indexer and sparse attention separately and together.
- [ ] Compare matching kernels with DeepGEMM, FlashMLA, or FlashInfer on rented
  supported hardware.

### Compressed Sparse Attention

- [ ] Implement overlapping compressed KV blocks.
- [ ] Index compressed blocks and select top-k blocks.
- [ ] Implement sparse shared-KV attention over selected compressed blocks plus
  the local sliding-window branch.
- [ ] Implement and validate CSA backward.
- [ ] Compare the exact CSA contract against a readable reference and a
  compatible production implementation on the same rented GPU.

Implementation order:

```text
shared token compression -> HCA -> DSA indexer -> DSA sparse attention -> CSA
```

Architecture-changing kernels are checked against references implementing the
same equations. Dense attention, MLA, HCA, DSA, and CSA are not treated as
numerically interchangeable models.

## Phase 8: Integration demonstrations

- [x] Dense GPT: train, checkpoint, resume, report utilization, and generate
  TinyStories text.
- [x] DeepSeek-V3-style: integrate MLA, MoE, balancing, shared experts, MTP,
  checkpointing, and compressed-cache generation.
- [ ] V4-style hybrid attention: integrate alternating HCA/CSA layers and train
  a small correctness-scale model on the same dataset.
- [ ] Compare parameter count, loss, tokens/s, peak training memory, and
  generated samples using documented small configurations.
- [ ] Use synthetic long-context kernel benchmarks, not short TinyStories
  training, to demonstrate compression and sparse-attention scaling.

Unless mHC, Muon, and the remaining V4 changes are implemented, label the last
model a `V4-style hybrid-compressed-attention model`, not a complete
DeepSeek-V4 reproduction.

## Final portfolio demonstration

1. GEMM versus cuBLAS on FP32 and BF16 workloads.
2. CUDA FlashAttention forward/backward versus official FlashAttention-2.
3. Frozen CUDA MLA training/decode correctness and compressed-cache study.
4. HCA, DSA, and CSA correctness and performance studies.
5. A secondary MoE routing/dispatch/grouped-GEMM study.
6. Dense GPT, DeepSeek-V3-style, and V4-style training demonstrations.
7. One consolidated table of numerical error, latency, relative library
   performance, TFLOP/s or bandwidth, registers, shared memory, occupancy, and
   spills.

## Primary references

- FlashAttention: https://github.com/Dao-AILab/flash-attention
- FlashMLA: https://github.com/deepseek-ai/FlashMLA
- CUTLASS: https://github.com/NVIDIA/cutlass
- FlashInfer: https://github.com/flashinfer-ai/flashinfer
- DeepGEMM: https://github.com/deepseek-ai/DeepGEMM
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
- DeepSeek-V3.2 / DSA: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp
- DeepSeek-V4: https://arxiv.org/abs/2606.19348
