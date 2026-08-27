#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
report_dir="$repo_root/profiles/reports"
benchmark="${1:-swiglu}"
test_name="$benchmark"
nvcc_bin="${CUDACXX:-}"
if [[ -z "$nvcc_bin" && -f "$build_dir/CMakeCache.txt" ]]; then
    nvcc_bin="$(
        sed -n 's/^CMAKE_CUDA_COMPILER:[^=]*=//p' \
            "$build_dir/CMakeCache.txt" | head -n 1
    )"
fi
if [[ -z "$nvcc_bin" && -x /usr/local/cuda/bin/nvcc ]]; then
    nvcc_bin="/usr/local/cuda/bin/nvcc"
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
    if [[ -x "$candidate" ]]; then
        ncu_bin="$candidate"
    fi
done
if [[ -z "$nvcc_bin" || -z "$ncu_bin" ]]; then
    echo "CUDA compiler or Nsight Compute was not found under /usr/local/cuda*" >&2
    exit 1
fi

case "$benchmark" in
    rmsnorm)
        test_target="test_rmsnorm"
        benchmark_target="benchmark_rmsnorm"
        kernel_pattern="regex:rmsnorm_.*_kernel"
        workload_name="rows"
        workloads=(512 2048 8192)
        ;;
    swiglu)
        test_target="test_swiglu"
        benchmark_target="benchmark_swiglu"
        kernel_pattern="regex:swiglu_.*_kernel"
        workload_name="elements"
        workloads=(262144 1048576 4194304)
        ;;
    matmul)
        test_target="test_matmul"
        benchmark_target="benchmark_matmul"
        kernel_pattern="regex:.*matmul_kernel.*"
        workload_name="size"
        workloads=(2048 4096 8192)
        ;;
    rope)
        test_target="test_rope"
        benchmark_target="benchmark_rope"
        kernel_pattern="regex:rope_.*_kernel"
        workload_name="sequence"
        workloads=(512 2048 8192)
        ;;
    softmax)
        test_target="test_softmax"
        benchmark_target="benchmark_softmax"
        kernel_pattern="regex:causal_softmax_.*_kernel"
        workload_name="sequence"
        workloads=(128 512 1024)
        ;;
    attention)
        test_target="test_attention"
        benchmark_target="benchmark_attention"
        kernel_pattern="regex:(attention_.*_kernel|.*matmul_kernel.*|causal_softmax_.*_kernel)"
        workload_name="sequence"
        workloads=(128 256 512)
        ;;
    flash_attention)
        test_target="test_flash_attention"
        benchmark_target="benchmark_flash_attention"
        kernel_pattern="regex:flash_attention_.*_kernel"
        workload_name="sequence"
        workloads=(128 256 512 1024)
        ;;
    transformer_block)
        test_target="test_transformer_block"
        benchmark_target="benchmark_transformer_block"
        kernel_pattern="regex:(rmsnorm_.*_kernel|rope_.*_kernel|attention_.*_kernel|causal_softmax_.*_kernel|swiglu_.*_kernel|residual_.*_kernel|.*matmul_kernel.*)"
        workload_name="sequence"
        workloads=(64 128 256)
        ;;
    training_step)
        test_name="model_bf16"
        test_target="test_model_bf16"
        benchmark_target="benchmark_training_step"
        kernel_pattern="regex:.*kernel.*"
        workload_name="attention"
        workloads=(composed flash2)
        ;;
    embedding)
        test_target="test_embedding"
        benchmark_target="benchmark_embedding"
        kernel_pattern="regex:embedding_.*_kernel"
        workload_name="tokens"
        workloads=(512 2048 8192)
        ;;
    adamw)
        test_target="test_adamw"
        benchmark_target="benchmark_adamw"
        kernel_pattern="regex:adamw_.*_kernel"
        workload_name="elements"
        workloads=(1048576 4194304 16777216)
        ;;
    cross_entropy)
        test_target="test_cross_entropy"
        benchmark_target="benchmark_cross_entropy"
        kernel_pattern="regex:cross_entropy_.*_kernel"
        workload_name="vocabulary"
        workloads=(8192 16384 32768)
        ;;
    global_norm)
        test_target="test_global_norm"
        benchmark_target="benchmark_global_norm"
        kernel_pattern="regex:(global_norm_.*_kernel|clip_gradients_kernel)"
        workload_name="elements"
        workloads=(1048576 4194304 16777216)
        ;;
    *)
        echo "usage: bash scripts/profile.sh {rmsnorm|swiglu|matmul|rope|softmax|attention|flash_attention|transformer_block|training_step|embedding|adamw|cross_entropy|global_norm}" >&2
        exit 1
        ;;
esac

cmake \
    -S "$repo_root" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$nvcc_bin" \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$build_dir" --target "$test_target" "$benchmark_target" -j
ctest --test-dir "$build_dir" --output-on-failure -R "^${test_name}$"

mkdir -p "$report_dir"
report_arguments=()
for workload in "${workloads[@]}"; do
    backends=(custom)
    if [[ "$benchmark" == "matmul" ]]; then
        backends=(fp32 bf16 cublas_fp32 cublas_bf16)
    fi

    for backend in "${backends[@]}"; do
        operations=(all)
        if [[ "$benchmark" == "matmul" ]]; then
            operations=(forward left_backward right_backward)
        fi

        for operation in "${operations[@]}"; do
            report="$report_dir/${benchmark}_${workload_name}_${workload}.ncu-rep"
            profile_label="${workload_name}=${workload}"
            benchmark_arguments=("$workload")
            active_kernel_pattern="$kernel_pattern"
            if [[ "$benchmark" == "rope" ]]; then
                benchmark_arguments=(4 "$workload" 16 128 64)
            elif [[ "$benchmark" == "softmax" ]]; then
                benchmark_arguments=(1 8 "$workload" 0.125)
            elif [[ "$benchmark" == "attention" ]]; then
                benchmark_arguments=(2 "$workload" 8 64 0.125)
            elif [[ "$benchmark" == "flash_attention" ]]; then
                benchmark_arguments=(2 "$workload" 8 64 bf16)
            elif [[ "$benchmark" == "transformer_block" ]]; then
                benchmark_arguments=(2 "$workload" 512 8 1536)
            elif [[ "$benchmark" == "training_step" ]]; then
                benchmark_arguments=("$workload" 4 256 4 256 4 768 4096)
            elif [[ "$benchmark" == "embedding" ]]; then
                benchmark_arguments=("$workload" 1024 32768)
            elif [[ "$benchmark" == "cross_entropy" ]]; then
                benchmark_arguments=(2048 "$workload")
            fi
            if [[ "$benchmark" == "matmul" ]]; then
                benchmark_arguments=(
                    "$workload" "$workload" "$workload" "$backend" "$operation"
                )
                profile_label="${profile_label}/${backend}/${operation}"
                report="$report_dir/${benchmark}_${workload_name}_${workload}_${backend}_${operation}.ncu-rep"
                case "$backend" in
                    bf16)
                        active_kernel_pattern="regex:.*matmul_tensor_core_.*kernel.*"
                        ;;
                    cublas_fp32|cublas_bf16)
                        active_kernel_pattern="regex:.*"
                        ;;
                esac
            fi

            printf 'Profiling %s: %s backend=%s operation=%s\n' "$benchmark" \
                "${workload_name}=${workload}" "$backend" "$operation"
            report_base="${report%.ncu-rep}"
            speed_report="${report_base}_speed.ncu-rep"
            metrics_report="${report_base}_metrics.ncu-rep"

            # Timing is collected alone because adding memory sections forces
            # replay passes that can perturb the duration used for comparison.
            "$ncu_bin" \
                --profile-from-start off \
                --cache-control none \
                --section SpeedOfLight \
                --kernel-name "$active_kernel_pattern" \
                --export "$speed_report" \
                --force-overwrite \
                "$build_dir/$benchmark_target" "${benchmark_arguments[@]}" \
                > /dev/null
            "$ncu_bin" \
                --profile-from-start off \
                --cache-control all \
                --section LaunchStats \
                --section Occupancy \
                --section MemoryWorkloadAnalysis \
                --metrics dram__bytes_read.sum,dram__bytes_write.sum \
                --kernel-name "$active_kernel_pattern" \
                --export "$metrics_report" \
                --force-overwrite \
                "$build_dir/$benchmark_target" "${benchmark_arguments[@]}" \
                > /dev/null
            report_arguments+=("$profile_label" "$speed_report" "$metrics_report")
        done
    done
done

printf '\nNsight Compute summary: %s\n' "$benchmark"
python3 "$repo_root/scripts/extract_ncu.py" "$ncu_bin" "${report_arguments[@]}"
printf '\nReports: %s/%s_%s_*.ncu-rep\n' "$report_dir" "$benchmark" "$workload_name"
