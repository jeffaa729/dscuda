# DeepSeek-V3 path

This repository now contains a complete, small single-GPU
DeepSeek-V3-style training and generation path. It is an educational CUDA
implementation rather than a reproduction of the 671B model: the goal is to
make the architecture, backward graph, state layout, and kernel behavior
inspectable.

## Implemented graph

The default configuration in `configs/tinystories_v3.conf` is:

```text
token embedding
  -> 3 x [RMSNorm -> MLA -> residual -> RMSNorm -> dense SwiGLU -> residual]
  -> 1 x [RMSNorm -> MLA -> residual -> RMSNorm -> DeepSeekMoE -> residual]
  -> RMSNorm -> tied vocabulary head -> cross-entropy
  + 1 sequential MTP module during training
```

The implementation includes:

- low-rank query compression and decoupled-RoPE MLA projections;
- BF16 compressed causal attention with FP32 online-softmax state;
- complete MLA backward recomputation without a quadratic probability buffer;
- no-drop expert routing, token dispatch/combine, routed and shared SwiGLU
  experts, and backward passes;
- auxiliary-loss-free routing-bias updates plus the complementary
  sequence-wise balance loss;
- dense initial FFN layers and one sequential Multi-Token Prediction module;
- tied embedding/head gradients, clipping, AdamW, checkpoint/resume, and text
  generation;
- a compressed BF16 MLA cache containing only `[kv_rank] + [rope_size]` per
  layer and token.

MTP is used only to form its weighted training objective. Generation executes
the main model and does not run the MTP module.

## FlashMLA-inspired decode

The decode path follows the central systems idea demonstrated by
[FlashMLA](https://github.com/deepseek-ai/FlashMLA): keep the compressed latent
KV representation in cache, partition a long cache among independent CTAs,
compute an online-softmax state in every split, and merge those states in a
small combine kernel. It is a project-specific educational kernel, not copied
FlashMLA code and not a comparison among multiple MLA versions.

`DeepSeekV3Model::forward_last_logits()` incrementally processes only new
tokens. Every layer projects the new normalized hidden state, appends one
compressed KV/RoPE entry, decodes against the prefix, runs its dense or MoE
FFN, and passes one hidden state to the next layer. A changed prompt or changed
weights invalidate the prefix state.

For the final V3 configuration, the cache is:

```text
4 layers * (64 latent + 32 RoPE) * 2 BF16 bytes = 768 bytes/token
```

The 7-layer parameter-matched dense MHA baseline would store BF16 K and V for
every layer:

```text
2 * 7 layers * 256 hidden * 2 BF16 bytes = 7168 bytes/token
```

That is a 9.33x cache-size reduction. The comparison script also reports the
attention-core decode FLOPs as context grows:

| context | dense attention FLOPs | MLA attention FLOPs | reduction |
|---:|---:|---:|---:|
| 128 | 917,504 | 655,360 | 1.40x |
| 512 | 3,670,016 | 2,621,440 | 1.40x |
| 2,048 | 14,680,064 | 10,485,760 | 1.40x |

The projection work, vocabulary head, and FFN work are independent of cache
length and are intentionally excluded from this context-scaling table.

## Correctness gates

`test_mla` compares compressed attention forward and all four backward outputs
with scalar CPU equations. Its selected SM89 Tensor Core shape measured a
maximum output error of `1.379e-4`; all gradient errors were below `4.3e-6`.

`test_mla_layer_decode` checks a whole MLA layer token by token against its
full causal forward result. `test_deepseek_v3_model` extends that check through
embedding, dense and MoE blocks, final normalization, and the tied head; the
cached model logits differed by at most `2.980e-8` in the tested graph.

Run every test with:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## Nsight Compute results

For `B=2`, four heads, `C=64`, and `R=32`, the selected MLA training forward
uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` for both score and
probability-value tiles.

| sequence | Tensor Core forward | registers/thread | static shared/block | spills |
|---:|---:|---:|---:|---:|
| 128 | 16.32 us | 210 | 32.77 KiB | 0 |
| 256 | 34.34 us | 210 | 32.77 KiB | 0 |
| 512 | 71.62 us | 210 | 32.77 KiB | 0 |

The profile also makes the current limitation clear: scalar query and shared-KV
backward kernels dominate the MLA training time. At sequence 512, forward was
`0.072 ms`, query backward `10.50 ms`, and shared-KV backward `17.53 ms`.

The split-KV decode profile was:

| cached context | split + combine time | split registers | combine registers | spills |
|---:|---:|---:|---:|---:|
| 128 | 29.70 us | 56 | 40 | 0 |
| 512 | 105.32 us | 56 | 40 | 0 |
| 2,048 | 407.84 us | 56 | 40 | 0 |

Reproduce and overwrite the reports with:

```bash
bash scripts/profile.sh mla
bash scripts/profile.sh mla_decode
```

Both commands first run the matching correctness test, save `.ncu-rep` files
under `profiles/reports`, and print extracted timing, bandwidth, occupancy,
cache, register, shared-memory, and spill tables.

## Parameter-matched TinyStories demonstration

Both final models used the same tokenizer, data windows, seed, batch `2`,
context `128`, AdamW settings, and 5,000 updates.

| model | parameters | final train loss | final validation loss | final-window tokens/s | allocated memory |
|---|---:|---:|---:|---:|---:|
| dense GPT, 7 layers, FFN 800 | 7.188M | 1.744 | 2.137 | 20,021 | 154.0 MiB |
| V3 style, 3 dense + 1 MoE + MTP | 7.193M | 1.977 | 2.448 | 10,970 | 177.6 MiB |

The dense model is faster and reaches a lower loss at this scale. The V3 demo
therefore demonstrates the actual CUDA architecture, sparse routing, MTP
training graph, and compressed inference state; it does not claim that a tiny
MoE should outperform a dense model or that the current scalar backward is
production optimized.

Run a short fresh comparison with:

```bash
bash scripts/compare_architectures.sh 200
```

Train the complete runs with:

```bash
./build/train_dscuda --config configs/tinystories_dense_matched.conf
./build/train_dscuda --config configs/tinystories_v3.conf
```

The final checkpoints generated coherent TinyStories-style continuations:

```text
V3:   Once upon a time, there was a little girl named Lily. She was very sad ...
Dense: Once upon a time, there was a little girl named Lily. She had a big tree ...
```

Generate from the V3 checkpoint with:

```bash
./build/generate_dscuda \
  --checkpoint checkpoints/tinystories_v3_complete/step_00005000 \
  --tokenizer data/tinystories/tokenizer.bin \
  --prompt "Once upon a time" \
  --tokens 60 --temperature 0.75 --top-k 30 --seed 1337
```

The architecture follows the
[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) and uses the
[official repository's weight-layout documentation](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/README_WEIGHTS.md)
as a naming and shape reference.
