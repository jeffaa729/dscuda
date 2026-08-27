# dscuda Roadmap
## Phase 1: Core training runtime

- [x] Complete the shared tensor, allocator, parameter, checkpoint, and CUDA utility code.
- [x] Integrate token embedding with repeated transformer blocks and a tied language-model head.
- [x] Integrate cross-entropy, global gradient clipping, and AdamW into the training loop.
- [x] Assemble, train, checkpoint, resume, and sample from a small dense GPT-2-style model.
- [x] Add selectable composed and fused causal attention training paths with online softmax, saved log-sum-exp, backward recomputation, and FP32/BF16 inputs.
- [x] Move the D=64 fused attention forward and backward tile products to BF16 Tensor Core MMA and benchmark against the composed baseline.
- [ ] Reduce fused-attention register pressure, pipeline K/V loads, and add Tensor Core shapes beyond D=64.

## Phase 2: DeepSeek-V3 path

- Implement MLA projections and a readable CPU reference.
- Adapt the dense fused-attention interface and Tensor Core tiling strategy into a BF16 causal MLA training kernel with asymmetric QK and V dimensions, FP32 online-softmax accumulation, saved log-sum-exp, and backward recomputation.
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

## Final demonstration

Train a small GPT baseline, a DeepSeek-V3-style MLA plus MoE model, and a V4-style hybrid-compressed-attention variant on the same dataset. Report correctness, validation loss, tokens per second, peak training memory, KV-cache bytes per token, and attention FLOPs as context length grows; the V4 demonstration should emphasize exact compression and scaling properties rather than claiming frontier-model quality at small scale.

## Primary references

- DeepSeek-V3 Technical Report: https://arxiv.org/abs/2412.19437
- DeepSeek-V4 Technical Report: https://arxiv.org/abs/2606.19348
- FlashMLA: https://github.com/deepseek-ai/FlashMLA
