#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
report_dir="$repo_root/profiles/reports"
result_dir="$repo_root/profiles/results"
python_bin="${DSCUDA_PYTHON:-$repo_root/.venv/bin/python}"
family="${1:-}"
[[ "$family" != "moe" ]] || family=grouped_gemm
[[ "$family" != "gemm" ]] || family=matmul
suite="${2:-quick}"
run_mode="${3:-profile}"

usage() {
    echo "usage: bash scripts/profile.sh {matmul|grouped_gemm|flash_attention|mla} [quick|full|h100] [profile|extract]" >&2
}

if [[ "$family" == "--help" || "$family" == "-h" ]]; then
    usage
    exit 0
fi

if [[ -z "$family"
    || "$suite" != "quick" && "$suite" != "full" && "$suite" != "h100"
    || "$run_mode" != "profile" && "$run_mode" != "extract" ]]; then
    usage
    exit 1
fi

if [[ ! -x "$python_bin" ]]; then
    echo "uv environment not found; run: uv sync" >&2
    exit 1
fi

nvcc_bin="${CUDACXX:-}"
if [[ -z "$nvcc_bin" && -f "$build_dir/CMakeCache.txt" ]]; then
    nvcc_bin="$(
        sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
            "$build_dir/CMakeCache.txt" | head -n 1
    )"
fi
if [[ -z "$nvcc_bin" && -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin=/usr/local/cuda/bin/nvcc
fi
if [[ -z "$nvcc_bin" ]]; then
    nvcc_bin="$(
        for candidate in /usr/local/cuda-*/bin/nvcc; do
            [[ -x "$candidate" ]] && printf '%s\n' "$candidate"
        done | sort -V | tail -n 1
    )"
fi

ncu_bin=""
for candidate in /usr/local/cuda/bin/ncu /usr/local/cuda-*/bin/ncu; do
    [[ -x "$candidate" ]] && ncu_bin="$candidate"
done
if [[ -z "$nvcc_bin" || -z "$ncu_bin" ]]; then
    echo "CUDA compiler or Nsight Compute was not found under /usr/local/cuda*" >&2
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

case "$family" in
    matmul)
        test_regex='^matmul$'
        build_targets=(test_matmul benchmark_matmul)
        ;;
    flash_attention)
        test_regex='^flash_attention$'
        build_targets=(test_flash_attention benchmark_flash_attention)
        ;;
    mla)
        test_regex='^(mla|mla_decode)$'
        build_targets=(
            test_mla test_mla_decode
            benchmark_mla benchmark_mla_decode
        )
        ;;
    grouped_gemm)
        test_regex='^(expert_dispatch|grouped_gemm)$'
        build_targets=(
            test_expert_dispatch test_grouped_gemm
            benchmark_grouped_gemm
        )
        ;;
    *)
        usage
        exit 1
        ;;
esac

mkdir -p "$build_dir"
if ! CUDACXX="$nvcc_bin" DSCUDA_CUDA_ARCH="$cuda_arch" \
        bash "$repo_root/scripts/build.sh" "${build_targets[@]}" \
        >"$build_dir/${family}_build.log" 2>&1; then
    cat "$build_dir/${family}_build.log" >&2
    exit 1
fi
if ! ctest --test-dir "$build_dir" --output-on-failure -R "$test_regex" \
        >"$build_dir/${family}_tests.log" 2>&1; then
    cat "$build_dir/${family}_tests.log" >&2
    exit 1
fi

mkdir -p "$report_dir" "$result_dir"
report_arguments=()

gpu_info="$(
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader \
        2>/dev/null | head -n 1 || true
)"
cuda_version="$($nvcc_bin --version | sed -n 's/.*release \([^,]*\).*/\1/p')"
ncu_version="$($ncu_bin --version | tail -n 1)"
flash_version="$(
    "$python_bin" -c \
        'import importlib.metadata; print(importlib.metadata.version("flash-attn"))' \
        2>/dev/null || printf 'not installed'
)"
{
    printf '# %s benchmark environment\n\n' "$family"
    printf -- '- GPU: %s\n' "${gpu_info:-unavailable}"
    printf -- '- CUDA compiler: %s\n' "${cuda_version:-unknown}"
    printf -- '- Nsight Compute: %s\n' "$ncu_version"
    printf -- '- CUDA architecture: sm_%s\n' "$cuda_arch"
    printf -- '- Build: CMake Release\n'
    printf -- '- dscuda commit: %s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf -- '- Shape suite: %s\n' "$suite"
    printf -- '- flash-attn package: %s\n' "$flash_version"
    printf -- '- flash-attn commit: %s\n' \
        "${DSCUDA_FLASH_ATTN_COMMIT:-not recorded}"
} >"$result_dir/${family}_environment.md"

profile_case() {
    local label="$1"
    local stem="$2"
    local kernel_pattern="$3"
    shift 3
    local speed_report="$report_dir/${family}_${stem}_speed.ncu-rep"
    local metrics_report="$report_dir/${family}_${stem}_metrics.ncu-rep"

    if [[ "$run_mode" == "profile" ]]; then
        if ! "$ncu_bin" \
            --profile-from-start off \
            --cache-control none \
            --section SpeedOfLight \
            --kernel-name "$kernel_pattern" \
            --export "$speed_report" \
            --force-overwrite \
            "$@" >"$speed_report.log" 2>&1; then
            cat "$speed_report.log" >&2
            echo "Nsight profiling failed; counter-free comparisons are available in scripts/benchmark.sh." >&2
            return 1
        fi
        if ! "$ncu_bin" \
            --profile-from-start off \
            --cache-control all \
            --section LaunchStats \
            --section Occupancy \
            --section MemoryWorkloadAnalysis \
            --metrics dram__bytes_read.sum,dram__bytes_write.sum \
            --kernel-name "$kernel_pattern" \
            --export "$metrics_report" \
            --force-overwrite \
            "$@" >"$metrics_report.log" 2>&1; then
            cat "$metrics_report.log" >&2
            return 1
        fi
    else
        [[ -f "$speed_report" && -f "$metrics_report" ]] || {
            echo "missing report pair for $label" >&2
            exit 1
        }
    fi
    report_arguments+=("$label" "$speed_report" "$metrics_report")
}

profile_matmul() {
    local sizes=(2048)
    [[ "$suite" != "quick" ]] && sizes=(2048 4096 8192)
    local backends=(fp32 bf16 cublas_fp32 cublas_bf16)
    local operations=(forward left_backward right_backward)
    local size backend operation pattern label stem
    for size in "${sizes[@]}"; do
        for backend in "${backends[@]}"; do
            for operation in "${operations[@]}"; do
                pattern='regex:.*matmul_kernel.*'
                [[ "$backend" == "bf16" ]] && \
                    pattern='regex:.*matmul_tensor_core_.*kernel.*'
                [[ "$backend" == cublas_* ]] && pattern='regex:.*'
                label="M=${size},N=${size},K=${size}/${backend}/${operation}"
                stem="${size}_${backend}_${operation}"
                profile_case \
                    "$label" "$stem" "$pattern" \
                    "$build_dir/benchmark_matmul" \
                    "$size" "$size" "$size" "$backend" "$operation"
            done
        done
    done
}

official_flash_available() {
    "$python_bin" -c 'import torch; from flash_attn import flash_attn_func' \
        >/dev/null 2>&1
}

profile_flash_attention() {
    local cases=("1 512 8 128")
    if [[ "$suite" != "quick" ]]; then
        cases=()
        local batch sequence heads
        for batch in 1 4; do
            for sequence in 128 256 512 1024 2048; do
                for heads in 4 8; do
                    cases+=("$batch $sequence $heads 128")
                done
            done
        done
    fi

    local have_official=0
    if official_flash_available; then
        have_official=1
    elif [[ "${DSCUDA_REQUIRE_EXTERNAL:-0}" == "1" ]]; then
        echo "PyTorch and the official flash-attn package are required" >&2
        exit 1
    else
        echo "Official flash-attn is not installed; profiling the custom kernel only." >&2
    fi

    local case_spec batch sequence heads dimension operation shape stem
    local custom_dump official_dump
    for case_spec in "${cases[@]}"; do
        read -r batch sequence heads dimension <<<"$case_spec"
        shape="B=${batch},T=${sequence},H=${heads},D=${dimension}"
        stem="B${batch}_T${sequence}_H${heads}_D${dimension}"

        if (( have_official )); then
            custom_dump="${TMPDIR:-/tmp}/dscuda_${stem}_custom.bin"
            official_dump="${TMPDIR:-/tmp}/dscuda_${stem}_official.bin"
            "$build_dir/benchmark_flash_attention" \
                "$batch" "$sequence" "$heads" "$dimension" all \
                "$custom_dump" >/dev/null
            "$python_bin" "$repo_root/reference/python/flash_attention_official.py" \
                "$batch" "$sequence" "$heads" "$dimension" all \
                "$official_dump" >/dev/null
            if ! "$python_bin" "$repo_root/scripts/compare_attention_dumps.py" \
                "$custom_dump" "$official_dump" \
                "$batch" "$sequence" "$heads" "$dimension" \
                >"$build_dir/flash_attention_comparison.log" 2>&1; then
                cat "$build_dir/flash_attention_comparison.log" >&2
                exit 1
            fi
            rm -f -- "$custom_dump" "$official_dump"
        fi

        for operation in forward backward; do
            profile_case \
                "$shape/custom/$operation" "${stem}_custom_${operation}" \
                'regex:flash_attention_.*_kernel' \
                "$build_dir/benchmark_flash_attention" \
                "$batch" "$sequence" "$heads" "$dimension" "$operation"
            if (( have_official )); then
                profile_case \
                    "$shape/official/$operation" \
                    "${stem}_official_${operation}" 'regex:.*flash.*' \
                    "$python_bin" \
                    "$repo_root/reference/python/flash_attention_official.py" \
                    "$batch" "$sequence" "$heads" "$dimension" "$operation"
            fi
        done
    done
}

profile_mla() {
    local sequences=(128)
    [[ "$suite" != "quick" ]] && sequences=(128 256 512)
    local sequence
    for sequence in "${sequences[@]}"; do
        profile_case \
            "B=1,T=${sequence},H=8,D=512,R=64/custom/forward_backward" \
            "T${sequence}_forward_backward" \
            'regex:mla_(forward|query_backward|kv_backward)_kernel' \
            "$build_dir/benchmark_mla" 1 "$sequence" 8 512 64
        profile_case \
            "B=2,KV=${sequence},H=16,D=512,R=64/custom/decode" \
            "KV${sequence}_decode" 'regex:mla_decode_(split|combine)_kernel' \
            "$build_dir/benchmark_mla_decode" 2 "$sequence" 16 512 64 8
    done
}

profile_grouped_gemm() {
    local rows=4096
    local experts=8
    local input_size=512
    local output_size=1536
    if [[ "$suite" == "h100" ]]; then
        rows=16384
        experts=64
        input_size=1024
        output_size=3072
    fi
    local distribution backend label stem pattern label_backend
    for distribution in uniform skewed hot; do
        for backend in custom cublas; do
            pattern='regex:grouped_linear_bf16_tensor_core_kernel'
            label_backend=custom_bf16
            if [[ "$backend" == "cublas" ]]; then
                pattern='regex:.*'
                label_backend=cublas_bf16
            fi
            label="M=${rows},E=${experts},K=${input_size},N=${output_size},load=${distribution}/${label_backend}/forward"
            stem="M${rows}_E${experts}_K${input_size}_N${output_size}_${distribution}_${backend}"
            profile_case \
                "$label" "$stem" "$pattern" \
                "$build_dir/benchmark_grouped_gemm" \
                "$rows" "$experts" "$input_size" "$output_size" \
                "$distribution" "$backend"
        done
    done
}

case "$family" in
    matmul) profile_matmul ;;
    flash_attention) profile_flash_attention ;;
    mla) profile_mla ;;
    grouped_gemm) profile_grouped_gemm ;;
esac

"$python_bin" "$repo_root/scripts/extract_ncu.py" \
    --csv-out "$result_dir/${family}.csv" \
    --markdown-out "$result_dir/${family}.md" \
    "$ncu_bin" "${report_arguments[@]}"
"$python_bin" "$repo_root/scripts/update_benchmark_summary.py" "$result_dir"
