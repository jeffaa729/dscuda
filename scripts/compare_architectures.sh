#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
steps="${1:-200}"
dense_config="${2:-configs/tinystories_dense_matched.conf}"
v3_config="${3:-configs/tinystories_v3.conf}"
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
    local config="$1"
    local output
    output="$(
        "$build_dir/train_dscuda" \
            --config "$repo_root/$config" \
            --steps "$steps" \
            --log-every "$steps" \
            --checkpoint-every 0
    )"
    printf '%s\n' "$output" | awk '
        /^Parameters:/ {
            parameters=$2
            memory=$7
        }
        /^step / {
            train=$4
            validation=$6
            tokens=$9
            tflops=$11
            mfu=$13
        }
        END {
            sub(/%/, "", mfu)
            printf "%s %s %s %s %s %s %s\n", parameters, train,
                validation, tokens, tflops, mfu, memory
        }
    '
}

read -r dense_params dense_train dense_val dense_tokens dense_tflops \
    dense_mfu dense_memory <<< "$(measure "$dense_config")"
read -r v3_params v3_train v3_val v3_tokens v3_tflops v3_mfu v3_memory \
    <<< "$(measure "$v3_config")"

dense_relative="1.00x"
v3_relative="$(awk -v v3="$v3_tokens" -v dense="$dense_tokens" \
    'BEGIN { printf "%.2fx", v3 / dense }')"

printf 'Parameter-matched TinyStories comparison (%s timed steps after warm-up)\n\n' \
    "$((steps - 1))"
printf '%-18s %10s %11s %11s %11s %10s %9s %11s %10s\n' \
    architecture params_M train_loss val_loss tokens/s TFLOP/s MFU memory_MiB relative
printf '%-18s %10s %11s %11s %11s %10s %8s%% %11s %10s\n' \
    dense_GPT "$dense_params" "$dense_train" "$dense_val" "$dense_tokens" \
    "$dense_tflops" "$dense_mfu" "$dense_memory" "$dense_relative"
printf '%-18s %10s %11s %11s %11s %10s %8s%% %11s %10s\n' \
    DeepSeek_V3 "$v3_params" "$v3_train" "$v3_val" "$v3_tokens" \
    "$v3_tflops" "$v3_mfu" "$v3_memory" "$v3_relative"

printf '\nInference scaling (BF16 cache, attention core only)\n'
printf '%-10s %18s %18s %12s\n' context dense_attention_FLOPs MLA_attention_FLOPs reduction
for context in 128 512 2048; do
    dense_flops=$((4 * 7 * 256 * context))
    mla_flops=$((2 * 4 * 4 * (2 * 64 + 32) * context))
    reduction="$(awk -v dense="$dense_flops" -v mla="$mla_flops" \
        'BEGIN { printf "%.2fx", dense / mla }')"
    printf '%-10d %18d %18d %12s\n' \
        "$context" "$dense_flops" "$mla_flops" "$reduction"
done
printf '\n%-18s %18s %14s\n' architecture KV_bytes_per_token cache_reduction
printf '%-18s %18d %14s\n' dense_GPT 7168 1.00x
printf '%-18s %18d %14s\n' DeepSeek_V3 768 9.33x
