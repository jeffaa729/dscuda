#!/usr/bin/env bash
# Reliable build on WSL-native filesystem (avoids drvfs clock skew).
set -uo pipefail
repo_root=/mnt/c/Users/Jeff/Documents/GitHub/dscuda
build_dir="$HOME/dscuda-build"
cd "$repo_root"
if [[ ! -f "$build_dir/CMakeCache.txt" ]]; then
    cmake -S "$repo_root" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
        -DDSCUDA_PYTHON="$repo_root/.venv/bin/python" -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
fi
cmake --build "$build_dir" -j6 2>&1 | grep -E "error|Error" | head -5
/usr/local/cuda/bin/cuobjdump --list-elf "$build_dir/libdscuda_flash_attention_bench.so" | head -6
echo "NATIVE BUILD DONE"
