# dscuda Roadmap

The project will first build a complete, trainable DeepSeek-V3-style model and then add selected DeepSeek-V4 mechanisms. Each operation should have a CPU reference, CUDA forward and backward implementations when training requires them, correctness tests, and an Nsight Compute benchmark where the operation is performance-critical.

## Current checkpoint: dense transformer block

- Complete: FP32 matmul, RMSNorm, RoPE, causal softmax, composed dense causal attention, SwiGLU, and residual forward/backward operators.
- Complete: vectorized token-embedding lookup and atomic repeated-token gradient accumulation.
- Complete: vectorized FP32 AdamW with bias correction, decoupled weight decay, and persistent first/second moments.
- Complete: fused stable vocabulary cross-entropy with log-sum-exp recomputation, plus global gradient norm and clipping.
- Complete: a pre-norm dense block with `RMSNorm -> QKV -> RoPE -> attention -> output projection -> residual -> RMSNorm -> gate/up -> SwiGLU -> down -> residual`.
- Complete: scalar CPU recomputation, accumulated input and parameter gradients, finite-difference checks, and Nsight Compute workloads at sequence lengths 64, 128, and 256.
- Next boundary: language-model head and repeated-block model assembly for a trainable dense GPT baseline.

## Phase 1: Core training runtime

- Complete the shared tensor, allocator, parameter, checkpoint, and CUDA utility code.
- Integrate token embedding with repeated transformer blocks and a tied language-model head.
- Integrate cross-entropy, global gradient clipping, and AdamW into the training loop.
- Assemble a small dense GPT-2-style model as the first end-to-end training check.

## Phase 2: DeepSeek-V3 path

- Implement MLA projections and a readable CPU reference.
- Adapt the FlashAttention-2 tiling strategy into a BF16 causal MLA training kernel with asymmetric QK and V dimensions, FP32 online-softmax accumulation, saved log-sum-exp, and backward recomputation.
- Implement compressed-KV MLA decoding separately from the training/prefill kernel.
- Implement DeepSeekMoE routing, token dispatch, grouped expert GEMM, token combine, shared experts, and their backward passes.
- Add auxiliary-loss-free load balancing and Multi-Token Prediction.
- Train and compare a small dense GPT model and a parameter-matched DeepSeek-V3-style model.

## Phase 3: DeepSeek-V4 architecture extensions

### Hybrid compressed attention

- Implement the shared token-wise KV compression primitive and its backward pass.
- Implement Heavily Compressed Attention first: non-overlapping heavy KV compression followed by dense shared-KV MQA.
- Implement Compressed Sparse Attention next: overlapping KV compression, a lightning indexer, top-k compressed-block selection, and sparse shared-KV MQA.
- Add grouped attention-output projection to avoid one very large output projection.
- Add per-head query and compressed-KV RMSNorm immediately before attention.
- Add partial RoPE on the final 64 dimensions of queries and KV entries, plus inverse-position RoPE on attention outputs.
- Add the uncompressed sliding-window attention branch needed for local dependencies and strict causality inside a compressed block.
- Add learnable attention-sink logits to the online-softmax denominator.

### Manifold-Constrained Hyper-Connections

- Expand the residual stream into a small number of parallel streams.
- Implement dynamic input, residual, and output mappings.
- Constrain the residual mapping to a doubly stochastic matrix and constrain input/output mappings with sigmoid.
- Fuse the small mappings, residual mixing, and layer input/output operations after the CPU reference and backward tests pass.

### Muon optimizer

- Keep AdamW for embeddings, the prediction head, RMSNorm weights, and the specified mHC biases and gates.
- Implement Muon momentum, Nesterov update, update rescaling, weight decay, and ten hybrid Newton-Schulz iterations for the remaining matrix parameters.
- Use BF16 matrix multiplication with FP32 norms and reductions, and verify the optimizer against a CPU reference on small matrices.

### V4 MoE changes

- Change routed-expert affinity scoring from sigmoid to `sqrt(softplus(x))`.
- Add the small sequence-wise expert-balance loss.
- Add token-ID hash routing for the initial MoE blocks as a separate routing mode.

## Phase 4: Systems extensions

- Add deterministic sparse-attention backward using separate partial-gradient buffers followed by a fixed-order reduction.
- Add multi-GPU expert parallelism, then overlap dispatch, expert GEMM, and combine in waves.
- Add context parallelism only after single-GPU compressed attention is correct and benchmarked.
- Treat activation checkpointing, heterogeneous KV-cache management, and on-disk KV cache as optional stretch work.

## Explicitly out of scope

- FP4 kernels and FP4 quantization-aware training.
- Reproducing trillion-parameter training scale.
- Implementing every production serving optimization in FlashMLA or DeepSeek's distributed infrastructure.

## Final demonstration

Train a small GPT baseline, a DeepSeek-V3-style MLA plus MoE model, and a V4-style hybrid-compressed-attention variant on the same dataset. Report correctness, validation loss, tokens per second, peak training memory, KV-cache bytes per token, and attention FLOPs as context length grows; the V4 demonstration should emphasize exact compression and scaling properties rather than claiming frontier-model quality at small scale.

## Primary references

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeek-V4 Technical Report: https://arxiv.org/abs/2606.19348
- FlashMLA: https://github.com/deepseek-ai/FlashMLA

## Attention implementation references

- Use LeetCUDA's split-Q FlashAttention-2 kernel as the primary SM89 implementation reference for `cp.async`, `ldmatrix`, MMA tiling, and online softmax.
- Also study `C:\Users\Jeff\Downloads\flash_attn.cu` for its BF16 causal mask, grouped-query head mapping, `[B, S, H, D]` layout, lazy online-softmax rescaling, and predicated output writeback.
- Do not directly port that second kernel to the current RTX 4060 target: its TMA bulk copies and `mbarrier` synchronization require Hopper-class SM90 hardware. It is forward-only, fixes the head dimension to 128, assumes equal QK/V dimensions, and does not save log-sum-exp for backward.
