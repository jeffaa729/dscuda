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
    *)
        echo "usage: bash scripts/profile.sh {rmsnorm|swiglu}" >&2
        exit 1
        ;;
esac

cmake \
    -S "$repo_root" \
    -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build "$build_dir" --target "$test_target" "$benchmark_target" -j
ctest --test-dir "$build_dir" --output-on-failure -R "^${benchmark}$"

mkdir -p "$report_dir"
report_arguments=()
for workload in "${workloads[@]}"; do
    report="$report_dir/${benchmark}_${workload_name}_${workload}.ncu-rep"
    printf 'Profiling %s: %s=%s\n' "$benchmark" "$workload_name" "$workload"
    "$ncu_bin" \
        --section LaunchStats \
        --section SpeedOfLight \
        --section Occupancy \
        --section MemoryWorkloadAnalysis \
        --metrics dram__bytes_read.sum,dram__bytes_write.sum \
        --kernel-name "$kernel_pattern" \
        --export "$report" \
        --force-overwrite \
        "$build_dir/$benchmark_target" "$workload" \
        > /dev/null
    report_arguments+=("${workload_name}=${workload}" "$report")
done

printf '\nNsight Compute summary: %s\n' "$benchmark"
python3 "$repo_root/scripts/extract_ncu.py" "$ncu_bin" "${report_arguments[@]}"
printf '\nReports: %s/%s_%s_*.ncu-rep\n' "$report_dir" "$benchmark" "$workload_name"
