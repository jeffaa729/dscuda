# Dense GPT training

The first trainable model is a bias-free, pre-norm dense causal language model:

```text
tokens -> tied embedding -> transformer block x L -> RMSNorm
       -> tied embedding transpose -> vocabulary cross-entropy
```

Every parameter and gradient occupies one aligned flat FP32 buffer. Backward
walks transformer blocks in reverse, accumulates both tied-embedding gradient
paths, computes the global gradient norm, clips it, and applies AdamW.

## Correctness gate

Build and run the complete test suite:

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

`test_model` compares the complete one-layer CUDA graph against the scalar CPU
reference, including loss, logits, and every parameter gradient.

## Fixed-batch overfit

Always run this before longer training after changing model wiring:

```bash
./build/train_dscuda --mode overfit --steps 200 --log-every 20
```

The test uses a small fixed token batch and exits with failure unless the final
loss is below `0.10` and less than five percent of its initial value.

## TinyStories smoke run

```bash
./build/train_dscuda \
    --mode tinystories \
    --data-dir data/tinystories \
    --steps 20 \
    --batch 1 --seq 32 \
    --layers 1 --hidden 64 --heads 4 --ffn 192 --rotary 16 \
    --log-every 5
```

## Train the default model and save checkpoints

```bash
./build/train_dscuda --config configs/tinystories.conf
```

The default TinyStories configuration is `L=4`, `H=256`, four heads,
`FFN=768`, `B=4`, and `T=256`, which is approximately 4.46 million parameters.
Use the command-line shape options to scale only after the fixed-batch test
continues to pass.

For the larger 8-layer, 512-hidden model:

```bash
./build/train_dscuda --config configs/dense_gpt_8l_512h.conf
```

Configuration files use one `name = value` setting per line. Command-line
values override the file, so a short check does not require editing it:

```bash
./build/train_dscuda \
    --config configs/dense_gpt_8l_512h.conf \
    --steps 10 --checkpoint-every 0
```

Each log line reports training-only tokens/s, achieved TFLOP/s, and estimated
model FLOPs utilization (MFU). The default `--peak-tflops 14.56` is the nominal
FP32 peak used for the RTX 4060 Laptop GPU; pass the appropriate FP32 or BF16
Tensor Core peak when running a different GPU or precision. FLOPs/token uses
the dense Transformer estimate `6N + 12LHQT` and intentionally excludes
elementwise and optimizer operations.

## FP32 versus BF16 mixed precision

Select the matrix backend with `precision = fp32` or `precision = bf16` in a
config file. The BF16 path keeps FP32 master parameters, AdamW moments,
gradients, residual activations, normalization, causal attention softmax, and
loss. It uses BF16 Tensor Core operands with FP32 accumulation for every block
linear projection and the tied vocabulary head.

Train the BF16 model with:

```bash
./build/train_dscuda --config configs/tinystories_bf16.conf
```

The larger BF16 model also has a complete config:

```bash
./build/train_dscuda --config configs/dense_gpt_8l_512h_bf16.conf
```

Compare both backends on identical initialization, data order, shape, and
training steps:

```bash
bash scripts/compare_precision.sh 200
```

Pass a second argument to compare another model configuration:

```bash
bash scripts/compare_precision.sh 200 configs/dense_gpt_8l_512h.conf
```

The script discards the first timed step as warm-up and prints one table with
loss, tokens/s, achieved TFLOP/s, MFU, allocated memory, and relative
throughput. FP32 uses a 14.56 TFLOP/s peak and BF16 uses a 58.25 TFLOP/s dense
Tensor Core peak for the RTX 4060 Laptop GPU. MFU is an architectural estimate:
the BF16 numerator includes the complete Transformer FLOPs estimate even though
non-matmul kernels still execute in FP32.

This mixed-precision stage retains the complete FP32 training state and adds
BF16 parameter shadows plus conversion workspace. Flash2 additionally retains
the BF16 Q/K/V tensors consumed by forward so backward can reuse them instead
of converting the saved FP32 tensors again.

## Fused causal attention

Select the fused attention path independently from matrix precision:

```bash
./build/train_dscuda \
    --config configs/tinystories_flash2.conf
```

The equivalent configuration setting is `attention = flash2`; use
`attention = composed` to retain the QK matmul, materialized probability
matrix, softmax, and PV matmul baseline. The fused path tiles K/V through
shared memory, performs FP32 online softmax, saves one log-sum-exp value per
`[B,H,T]` row, and reconstructs probabilities during backward. Its saved
attention state is therefore `O(BHT)` rather than `O(BHT^2)`.

For BF16 inputs with `D=64` and a sequence length divisible by 64, the SM89
kernel uses native `mma.sync` instructions for `QK^T`, `PV`, `dO V^T`, `dS K`,
`dS^T Q`, and `P^T dO`. Softmax maxima, denominators, log-sum-exp, and output
accumulation remain FP32. Other supported shapes use the readable warp-level
fallback. On the RTX 4060 Laptop test system, Nsight Compute measured:

| sequence | composed time | fused time | composed DRAM reads | fused DRAM reads |
|---:|---:|---:|---:|---:|
| 128 | 0.122 ms | 0.055 ms | 18.11 MB | 6.16 MB |
| 256 | 0.323 ms | 0.136 ms | 48.97 MB | 12.20 MB |
| 512 | 1.293 ms | 0.312 ms | 151.44 MB | 24.30 MB |

For the default four-layer TinyStories configuration at `B=4`, `T=256`, and
`H=256`, a short identical-data comparison measured 100,908 tokens/s and
274.7 MiB with composed attention versus 123,623 tokens/s and 221.7 MiB with
the fused Tensor Core path. Tokens/s varies with laptop power and clock state,
so use the full-step Nsight report for kernel-level attribution.

Reproduce the fused-kernel profile and overwrite its previous reports with:

```bash
bash scripts/profile.sh flash_attention
```

Profile one complete BF16 training step in both attention modes with:

```bash
bash scripts/profile.sh training_step
```

The benchmark warms the model once, captures one complete forward, backward,
gradient-clipping, AdamW, and BF16-weight-refresh step, overwrites the previous
reports, and prints totals grouped by GEMM, conversion, attention, loss,
optimizer, normalization, and elementwise stages. After retaining BF16 Q/K/V
for backward, the RTX 4060 Laptop profile measured:

| attention | launches | GPU kernel time | BF16 conversion time | GEMM time | attention time |
|---|---:|---:|---:|---:|---:|
| composed | 285 | 6.287 ms | 0.905 ms | 3.113 ms | 0.615 ms |
| Flash2 | 261 | 5.545 ms | 1.025 ms | 2.342 ms | 0.511 ms |

Persisting Q/K/V removes three backward conversions per layer. For this
four-layer model, Flash2 conversion launches fell from 112 to 100 and total
profiled kernel time fell from 5.599 ms to 5.545 ms. The remaining twelve
Flash2-only conversions are the forward Q/K/V casts because the projection
GEMMs still write FP32 activations.

The next dataflow optimization is making Q/K/V projection GEMMs write BF16,
then applying RoPE in BF16 or while Flash2 loads Q/K. After that, reduce the
current 197-220 attention registers per thread, double-buffer K/V tile loads,
and add Tensor Core dispatches for more head dimensions.

Each `step_XXXXXXXX` checkpoint contains a versioned architecture header,
parameters, both AdamW moment buffers, the completed step, and a `DONE` marker.
The marker is written last, so an interrupted `.part` file is never treated as
a resumable checkpoint.

## Resume training

`--steps` is the final global step, not the number of additional steps:

```bash
./build/train_dscuda \
    --mode tinystories \
    --data-dir data/tinystories \
    --resume checkpoints/tinystories/step_00003000 \
    --steps 6000 \
    --log-every 100 \
    --checkpoint-every 1000 \
    --output-dir checkpoints/tinystories
```

Training windows are derived from the seed and global step. Resuming with the
same seed therefore continues the data order without serializing a C++ random
engine implementation.

## Generate text

```bash
./build/generate_dscuda \
    --checkpoint checkpoints/tinystories/step_00003000 \
    --tokenizer data/tinystories/tokenizer.bin \
    --prompt "Once upon a time" \
    --tokens 100 \
    --temperature 0.75 \
    --top-k 30 \
    --seed 1337
```

The sampler supports greedy decoding with `--temperature 0`, or stable
temperature plus top-k multinomial sampling. This first correctness-oriented
generator recomputes the prefix for every token and stops at the model context
length (`T=256` by default); a KV cache is a later performance optimization.
