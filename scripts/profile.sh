#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
report_dir="$repo_root/profiles/reports"
benchmark="${1:-swiglu}"
ncu_bin="/usr/local/cuda/bin/ncu"

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
    *)
        echo "usage: bash scripts/profile.sh {rmsnorm|swiglu|matmul}" >&2
        exit 1
        ;;
esac

cmake \
    -S "$repo_root" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$build_dir" --target "$test_target" "$benchmark_target" -j
ctest --test-dir "$build_dir" --output-on-failure -R "^${benchmark}$"

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
                --cache-control none \
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
