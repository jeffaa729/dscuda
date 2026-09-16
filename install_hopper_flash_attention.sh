#!/usr/bin/env bash
set -euo pipefail

python_bin="${DSCUDA_PYTHON:-/opt/dscuda-env/.venv/bin/python}"
repository="${FLASH_ATTENTION_REPOSITORY:-https://github.com/Dao-AILab/flash-attention.git}"
revision="${FLASH_ATTENTION_REF:-main}"
jobs="${CUDA_BUILD_JOBS:-4}"

if [[ ! -x "$python_bin" ]]; then
    echo "Python environment not found: $python_bin" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi --query-gpu=compute_cap --format=csv,noheader | grep -q '^9\.0'; then
    echo "FlashAttention-3 installation requires an H100-class SM90 GPU." >&2
    exit 1
fi

source_dir="$(mktemp -d)"
trap 'rm -rf "$source_dir"' EXIT

git clone --filter=blob:none --no-checkout "$repository" "$source_dir/FlashAttention"
git -C "$source_dir/FlashAttention" fetch --depth 1 origin "$revision"
git -C "$source_dir/FlashAttention" checkout --detach FETCH_HEAD
git -C "$source_dir/FlashAttention" submodule update --init --depth 1 csrc/cutlass

uv pip install --python "$python_bin" setuptools wheel ninja packaging
BUILD_TARGET=cuda \
MAX_JOBS="$jobs" \
FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
FLASH_ATTENTION_DISABLE_FP16=TRUE \
FLASH_ATTENTION_DISABLE_FP8=TRUE \
FLASH_ATTENTION_DISABLE_VARLEN=TRUE \
FLASH_ATTENTION_DISABLE_HDIM64=TRUE \
FLASH_ATTENTION_DISABLE_HDIM96=TRUE \
FLASH_ATTENTION_DISABLE_HDIM192=TRUE \
FLASH_ATTENTION_DISABLE_HDIM256=TRUE \
uv pip install --python "$python_bin" --no-build-isolation "$source_dir/FlashAttention/hopper"

"$python_bin" -c 'import flash_attn_interface; assert hasattr(flash_attn_interface, "flash_attn_func")'
uv pip install --python "$python_bin" flash-attn-4
"$python_bin" -c 'import flash_attn.cute; assert hasattr(flash_attn.cute, "flash_attn_func")'

echo "FlashAttention-3 and FlashAttention-4 are ready."
