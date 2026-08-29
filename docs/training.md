# Dense GPT training

The dense model is a bias-free, pre-norm causal Transformer:

```text
tokens -> tied embedding -> transformer block x L -> RMSNorm
       -> tied embedding transpose -> vocabulary cross-entropy
```

Backward walks the blocks in reverse, accumulates both tied-embedding gradient
paths, clips by the global gradient norm, and applies AdamW. FP32 master
parameters, gradients, and optimizer moments are retained in both precision
modes; the BF16 path uses Tensor Core operands with FP32 accumulation for
linear layers and BF16 inputs with FP32 softmax state in fused attention.

## Build and correctness

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

`test_model` compares a complete one-layer CUDA graph with the scalar CPU
reference. `test_model_bf16` checks the mixed-precision graph, and
`test_flash_attention` checks fused output, log-sum-exp state, and Q/K/V
gradients.

Always run the fixed-batch overfit gate after changing model wiring:

```bash
./build/train_dscuda --mode overfit --steps 200 --log-every 20
```

## TinyStories configuration

The repository keeps one dense training configuration and one V3-style
configuration:

```text
configs/tinystories_dense_matched.conf
configs/tinystories_v3.conf
```

Train the dense model and write resumable checkpoints with:

```bash
./build/train_dscuda --config configs/tinystories_dense_matched.conf
```

Configuration files use one `name = value` setting per line. Command-line
values override the file, so precision, attention, or a short smoke run does
not need another nearly identical config:

```bash
./build/train_dscuda \
  --config configs/tinystories_dense_matched.conf \
  --precision bf16 \
  --steps 20 --log-every 5 --checkpoint-every 0

./build/train_dscuda \
  --config configs/tinystories_dense_matched.conf \
  --attention composed \
  --steps 20 --log-every 5 --checkpoint-every 0
```

The default dense config uses fused `flash2` attention. `attention = composed`
retains the QK matmul, materialized causal probability matrix, softmax, and PV
matmul baseline. The fused path uses online softmax and stores one
log-sum-exp value per `[B,H,T]` row instead of an `O(BHT^2)` probability
matrix.

Compare FP32 and BF16 on the same model, initialization, and data order with:

```bash
bash scripts/compare_precision.sh 200
```

Training logs report loss, tokens/s, achieved TFLOP/s, and estimated model
FLOPs utilization. Set `peak-tflops` to the appropriate peak for the selected
GPU and datatype; it is only the denominator used for MFU.

## Kernel profiling

The comparison profiler builds the matching tests, runs them, captures timing
and hardware metrics in separate Nsight Compute reports, and overwrites the
previous reports. It writes concise CSV and Markdown tables under
`profiles/results`.

Python benchmark dependencies are locked with uv. Create or synchronize the
repository-local environment before profiling:

```bash
uv sync
```

The profiling script uses `.venv/bin/python` directly.

```bash
# Fast local shape set
bash scripts/profile.sh matmul quick
bash scripts/profile.sh flash_attention quick
bash scripts/profile.sh moe quick

# Full 2048/4096/8192 GEMM set and the complete attention shape matrix
bash scripts/profile.sh matmul full
bash scripts/profile.sh flash_attention full

# Frozen MLA training/decode snapshot
bash scripts/profile.sh mla full
```

The FlashAttention command automatically adds the official `flash-attn`
backend from the locked uv environment. Set `DSCUDA_REQUIRE_EXTERNAL=1` when
a missing reference should
fail rather than run custom-only. On a rented H100, use `h100` and optionally
set `DSCUDA_CUDA_ARCH=90`; all relative timings must come from the same GPU.
Record the checked-out official commit in the generated environment report:

```bash
DSCUDA_REQUIRE_EXTERNAL=1 \
DSCUDA_CUDA_ARCH=90 \
DSCUDA_FLASH_ATTN_COMMIT=<official-flash-attention-commit> \
bash scripts/profile.sh flash_attention h100
```

## Checkpoint and resume

Each `step_XXXXXXXX` checkpoint contains a versioned architecture header,
parameters, both AdamW moment buffers, the completed step, and a `DONE` marker.
The marker is written last, so an interrupted `.part` directory is not treated
as resumable.

`--steps` is the final global step, not the number of additional steps:

```bash
./build/train_dscuda \
  --config configs/tinystories_dense_matched.conf \
  --resume checkpoints/tinystories_dense_matched/step_00003000 \
  --steps 6000
```

Training windows are derived from the seed and global step, so resuming with
the same seed continues the data order.

## Generate text

```bash
./build/generate_dscuda \
  --checkpoint checkpoints/tinystories_dense_matched/step_00005000 \
  --tokenizer data/tinystories/tokenizer.bin \
  --prompt "Once upon a time" \
  --tokens 100 --temperature 0.75 --top-k 30 --seed 1337
```

Dense GPT recomputes the prefix for every generated token. The V3-style model
instead maintains its compressed MLA cache; production serving and paged
KV-cache management remain outside this training-kernel project.

The V3 graph, cache layout, matched TinyStories comparison, and generation
command are documented in [`deepseek_v3.md`](deepseek_v3.md).
