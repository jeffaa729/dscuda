#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
python_bin="$repo_root/.venv/bin/python"
family="${1:-}"
suite="${2:-quick}"

if [[ "$family" != "flash_attention"
    || "$suite" != "quick" && "$suite" != "full" && "$suite" != "h100" ]]; then
    echo "usage: bash scripts/benchmark.sh flash_attention [quick|full|h100]" >&2
    exit 1
fi

if [[ ! -x "$python_bin" ]]; then
    echo "uv environment not found; run: uv sync" >&2
    exit 1
fi

nvcc_bin="${CUDACXX:-}"
if [[ -z "$nvcc_bin" && -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin=/usr/local/cuda/bin/nvcc
fi
if [[ -z "$nvcc_bin" ]]; then
    nvcc_bin="$(command -v nvcc || true)"
fi
if [[ -z "$nvcc_bin" ]]; then
    echo "CUDA compiler not found" >&2
    exit 1
fi

cuda_arch="${DSCUDA_CUDA_ARCH:-}"
if [[ -z "$cuda_arch" ]] && command -v nvidia-smi >/dev/null; then
    cuda_arch="$(
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | head -n 1 | tr -d '. '
    )"
fi
cuda_arch="${cuda_arch:-89}"

cmake \
    -S "$repo_root" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
    -DCMAKE_CUDA_COMPILER="$nvcc_bin"
cmake --build "$build_dir" \
    --target test_flash_attention dscuda_flash_attention_bench -j
ctest --test-dir "$build_dir" --output-on-failure -R '^(flash_attention|attention_benchmark_python)$'

warmup_ms="${DSCUDA_BENCHMARK_WARMUP_MS:-1000}"
graph_operations="${DSCUDA_GRAPH_OPERATIONS:-100}"
graph_replays="${DSCUDA_GRAPH_REPLAYS:-10}"
iterations="${DSCUDA_BENCHMARK_ITERATIONS:-50}"
trials="${DSCUDA_BENCHMARK_TRIALS:-9}"
output_dir="$repo_root/profiles/runtime"

"$python_bin" "$repo_root/scripts/benchmark_flash_attention_runtime.py" \
    --library "$build_dir/libdscuda_flash_attention_bench.so" \
    --suite "$suite" \
    --output-dir "$output_dir" \
    --warmup-ms "$warmup_ms" \
    --graph-operations "$graph_operations" \
    --graph-replays "$graph_replays" \
    --iterations "$iterations" \
    --trials "$trials"

printf '\nMatched BF16 graph/API results: %s/flash_attention.{csv,md}\n' \
    "$output_dir"
