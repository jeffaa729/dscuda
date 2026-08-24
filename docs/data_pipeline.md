# Tokenizer and dataset pipeline

The tokenizer is a from-scratch byte-level BPE implementation. It begins with
all 256 byte values, reserves token 256 for BOS and token 257 for EOS, and then
learns ranked byte-pair merges. Because the base vocabulary contains every
byte, arbitrary UTF-8 input is representable and there is no unknown token.

## Prepare TinyStories

From the repository root:

```bash
python3 tools/prepare_tinystories.py \
    --output-dir data/tinystories \
    --vocab-size 4096 \
    --train-bytes 32m \
    --validation-bytes 4m \
    --tokenizer-bytes 16m
```

The bounded download makes an initial end-to-end training dataset. A larger
run should use a new output directory so the cached raw subset and tokenizer
remain reproducible.

```bash
python3 tools/prepare_tinystories.py \
    --output-dir data/tinystories-256m \
    --vocab-size 4096 \
    --train-bytes 256m \
    --validation-bytes 16m \
    --tokenizer-bytes 64m
```

## Generated files

- `tokenizer.bin`: versioned tokenizer header followed by ordered BPE merge
  records. Python and C++ read the same file.
- `train.bin`: packed little-endian `uint16` training token IDs.
- `val.bin`: packed little-endian `uint16` validation token IDs.
- `metadata.json`: vocabulary, special-token, document, byte, and token counts.

The raw corpus and generated artifacts are ignored by Git. They are rebuilt by
the preparation command rather than committed to the repository.

## Training-loop interface

```cpp
dscuda::Tokenizer tokenizer("data/tinystories/tokenizer.bin");
dscuda::TokenDataset train("data/tinystories/train.bin");

std::vector<int> inputs;
std::vector<int> targets;
train.get_batch(start_token, batch_size, sequence_length, inputs, targets);
```

For each flattened position, `targets[i]` is the token immediately following
`inputs[i]`. EOS separates stories, so fixed-length training sequences may be
packed continuously without padding.
