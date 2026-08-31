#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${DSCUDA_PYTHON:-$repo_root/.venv/bin/python}"
if [[ $# == 0 || "$1" == "--help" || "$1" == "-h" ]]; then
    echo "usage: bash scripts/benchmark.sh FAMILY [quick|full|h100] [options]"
    echo "families: matmul, grouped_gemm, flash_attention, mla, expert_dispatch, hca, dsa, csa, kda, all"
    echo "options: --reference pytorch|cublas|flash_attention|flashmla|both|fla, --operation NAME"
    exit 0
fi
family="$1"
shift
suite=quick
if (( $# )) && [[ "$1" != --* ]]; then suite="$1"; shift; fi
if [[ ! -x "$python_bin" ]]; then
    echo "Run uv sync --locked first." >&2
    exit 1
fi
mkdir -p "$repo_root/build"
if [[ ! "$family" =~ ^(hca|dsa|csa|kda)$ ]]; then
    if ! bash "$repo_root/scripts/build.sh" >"$repo_root/build/benchmark_build.log" 2>&1; then
        cat "$repo_root/build/benchmark_build.log" >&2
        exit 1
    fi
fi
options=()
reference="${DSCUDA_REFERENCE_BACKEND:-}"
[[ "$family" != mla ]] || reference="${DSCUDA_MLA_REFERENCE:-$reference}"
[[ -z "$reference" ]] || options+=(--reference "$reference")
if ! "$python_bin" "$repo_root/benchmarks/run.py" "$family" --suite "$suite" \
        --warmup-ms "${DSCUDA_BENCHMARK_WARMUP_MS:-1000}" \
        --graph-operations "${DSCUDA_GRAPH_OPERATIONS:-10}" \
        --trials "${DSCUDA_BENCHMARK_TRIALS:-9}" \
        "${options[@]}" "$@" >"$repo_root/build/${family}_runtime.log" 2>&1; then
    cat "$repo_root/build/${family}_runtime.log" >&2
    exit 1
fi
# Keep successful output table-only; retain import warnings and errors in the log.
sed -n '/^|/p' "$repo_root/build/${family}_runtime.log"
