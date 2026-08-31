#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
python_bin="${DSCUDA_PYTHON:-$repo_root/.venv/bin/python}"
family="${1:-}"
suite="${2:-quick}"
[[ "$family" != "gemm" ]] || family=matmul
[[ "$family" != "moe" ]] || family=grouped_gemm
extra_targets=()

usage() {
    echo "usage: bash scripts/benchmark.sh {flash_attention|mla|matmul|grouped_gemm|hca|dsa|csa|kda} [quick|full|h100]"
    echo "Runs correctness checks, then same-process CUDA Graph comparisons."
    echo "For Nsight metrics and other operators, use scripts/profile.sh."
}
if [[ "$family" == "--help" || "$family" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "$suite" != "quick" && "$suite" != "full" && "$suite" != "h100" ]]; then
    usage >&2
    exit 1
fi
if [[ "$family" =~ ^(grouped_gemm|hca|dsa|csa|kda)$ ]]; then
    if [[ ! -x "$python_bin" ]]; then
        echo "uv environment not found; run: uv sync --locked" >&2
        exit 1
    fi
    mkdir -p "$build_dir"
    if [[ "$family" == "grouped_gemm" ]]; then
        if ! bash "$repo_root/scripts/build.sh" test_grouped_gemm test_expert_dispatch dscuda_operator_bench \
                >"$build_dir/${family}_build.log" 2>&1; then
            cat "$build_dir/${family}_build.log" >&2
            exit 1
        fi
        if ! ctest --test-dir "$build_dir" --output-on-failure -R '^(grouped_gemm|expert_dispatch)$' \
                >"$build_dir/${family}_native_tests.log" 2>&1; then
            cat "$build_dir/${family}_native_tests.log" >&2
            exit 1
        fi
    fi
    if ! "$python_bin" "$repo_root/tests/test_references.py" >"$build_dir/${family}_tests.log" 2>&1; then
        cat "$build_dir/${family}_tests.log" >&2
        exit 1
    fi
    backend_args=()
    [[ -z "${DSCUDA_REFERENCE_BACKEND:-}" ]] || backend_args=(--backend "$DSCUDA_REFERENCE_BACKEND")
    if ! "$python_bin" "$repo_root/scripts/benchmark_references.py" "$family" --suite "$suite" \
            "${backend_args[@]}" --output-dir "$repo_root/profiles/runtime" \
            --warmup-ms "${DSCUDA_BENCHMARK_WARMUP_MS:-1000}" \
            --graph-operations "${DSCUDA_GRAPH_OPERATIONS:-1}" \
            --graph-replays "${DSCUDA_GRAPH_REPLAYS:-3}" \
            --trials "${DSCUDA_BENCHMARK_TRIALS:-9}" >"$build_dir/${family}_runtime.log" 2>&1; then
        cat "$build_dir/${family}_runtime.log" >&2
        exit 1
    fi
    # Upstream JIT/import warnings stay in the log; successful stdout is table-only.
    sed -n '/^|/p' "$build_dir/${family}_runtime.log"
    exit 0
fi
case "$family" in
    flash_attention)
        bridge=dscuda_flash_attention_bench
        runner=benchmark_flash_attention_runtime.py
        test_regex='^(flash_attention|attention_benchmark_python)$'
        default_operations=100
        default_replays=10
        extra_args=(--iterations "${DSCUDA_BENCHMARK_ITERATIONS:-50}")
        ;;
    mla)
        bridge=dscuda_mla_bench
        runner=benchmark_mla_runtime.py
        test_regex='^(mla|mla_decode|mla_benchmark_python|mla_reference_python|mla_pytorch_cuda)$'
        default_operations=5
        default_replays=3
        extra_args=()
        extra_targets=(test_mla_decode)
        ;;
    matmul)
        bridge=dscuda_operator_bench
        runner=benchmark_operators_runtime.py
        test_regex="^($family|operator_benchmark_python)$"
        default_operations=10
        default_replays=3
        extra_args=("$family")
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
if [[ ! -x "$python_bin" ]]; then
    echo "uv environment not found; run: uv sync --locked" >&2
    exit 1
fi
mkdir -p "$build_dir"
if ! bash "$repo_root/scripts/build.sh" "test_$family" "$bridge" "${extra_targets[@]}" \
        >"$build_dir/${family}_build.log" 2>&1; then
    cat "$build_dir/${family}_build.log" >&2
    exit 1
fi
if ! ctest --test-dir "$build_dir" --output-on-failure -R "$test_regex" \
        >"$build_dir/${family}_tests.log" 2>&1; then
    cat "$build_dir/${family}_tests.log" >&2
    exit 1
fi
"$python_bin" "$repo_root/scripts/$runner" "${extra_args[@]}" \
    --library "$build_dir/lib$bridge.so" \
    --suite "$suite" \
    --output-dir "$repo_root/profiles/runtime" \
    --warmup-ms "${DSCUDA_BENCHMARK_WARMUP_MS:-1000}" \
    --graph-operations "${DSCUDA_GRAPH_OPERATIONS:-$default_operations}" \
    --graph-replays "${DSCUDA_GRAPH_REPLAYS:-$default_replays}" \
    --trials "${DSCUDA_BENCHMARK_TRIALS:-9}"
