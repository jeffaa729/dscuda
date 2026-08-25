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
./build/train_dscuda \
    --mode tinystories \
    --data-dir data/tinystories \
    --steps 3000 \
    --log-every 100 \
    --checkpoint-every 1000 \
    --output-dir checkpoints/tinystories
```

The default TinyStories configuration is `L=4`, `H=256`, four heads,
`FFN=768`, `B=4`, and `T=256`, which is approximately 4.46 million parameters.
Use the command-line shape options to scale only after the fixed-batch test
continues to pass.

Each log line reports training-only tokens/s, achieved TFLOP/s, and estimated
model FLOPs utilization (MFU). The default `--peak-tflops 14.56` is the nominal
FP32 peak used for the RTX 4060 Laptop GPU; pass the appropriate FP32 or BF16
Tensor Core peak when running a different GPU or precision. FLOPs/token uses
the dense Transformer estimate `6N + 12LHQT` and intentionally excludes
elementwise and optimizer operations.

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
