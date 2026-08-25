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

## Default initial model

```bash
./build/train_dscuda \
    --mode tinystories \
    --data-dir data/tinystories \
    --steps 1000 \
    --log-every 10
```

The default TinyStories configuration is `L=4`, `H=256`, four heads,
`FFN=768`, `B=4`, and `T=256`, which is approximately 4.46 million parameters.
Use the command-line shape options to scale only after the fixed-batch test
continues to pass.
