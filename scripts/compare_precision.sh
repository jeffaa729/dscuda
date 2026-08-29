#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
steps="${1:-200}"
base_config="${2:-configs/tinystories_dense_matched.conf}"
cd "$repo_root"

build_log="$(mktemp)"
trap 'rm -f "$build_log"' EXIT

if ! cmake \
        -S "$repo_root" \
        -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=89 > "$build_log" 2>&1; then
    cat "$build_log"
    exit 1
fi
if ! cmake --build "$build_dir" --target train_dscuda -j \
        > "$build_log" 2>&1; then
    cat "$build_log"
    exit 1
fi

measure() {
    local precision="$1"
    local peak_tflops="$2"
    local output
    output="$(
        "$build_dir/train_dscuda" \
            --config "$repo_root/$base_config" \
            --precision "$precision" \
            --peak-tflops "$peak_tflops" \
            --steps "$steps" \
            --log-every "$steps" \
            --checkpoint-every 0
    )"
    local result
    local memory
    result="$(printf '%s\n' "$output" | awk '/^step / { line=$0 } END { print line }')"
    memory="$(printf '%s\n' "$output" | awk '/^Parameters:/ { print $7 }')"
    printf '%s\n' "$result" | awk -v memory="$memory" '
        {
            mfu=$13
            sub(/%/, "", mfu)
            printf "%s %s %s %s %s %s\n", $4, $6, $9, $11, mfu, memory
        }
    '
}

read -r fp32_train fp32_val fp32_tokens fp32_tflops fp32_mfu fp32_memory \
    <<< "$(measure fp32 14.56)"
read -r bf16_train bf16_val bf16_tokens bf16_tflops bf16_mfu bf16_memory \
    <<< "$(measure bf16 58.25)"

fp32_relative="1.00x"
bf16_relative="$(awk -v bf16="$bf16_tokens" -v fp32="$fp32_tokens" \
    'BEGIN { printf "%.2fx", bf16 / fp32 }')"

printf 'Dense GPT precision comparison: %s (%s timed steps after warm-up)\n\n' \
    "$base_config" "$((steps - 1))"
printf '%-10s %12s %12s %12s %12s %10s %12s %11s\n' \
    backend train_loss val_loss tokens/s TFLOP/s MFU memory_MiB relative
printf '%-10s %12s %12s %12s %12s %9s%% %12s %11s\n' \
    FP32 "$fp32_train" "$fp32_val" "$fp32_tokens" "$fp32_tflops" \
    "$fp32_mfu" "$fp32_memory" "$fp32_relative"
printf '%-10s %12s %12s %12s %12s %9s%% %12s %11s\n' \
    BF16 "$bf16_train" "$bf16_val" "$bf16_tokens" "$bf16_tflops" \
    "$bf16_mfu" "$bf16_memory" "$bf16_relative"
