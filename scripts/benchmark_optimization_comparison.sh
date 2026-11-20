#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
baseline_commit="34a3ef6"
comparison_tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/gloomy-ablation.XXXXXX")"
trap 'rm -rf -- "$comparison_tmpdir"' EXIT
baseline_dir="$comparison_tmpdir/baseline"
mkdir -p "$baseline_dir"
git -C "$repo_root" archive "$baseline_commit" | tar -x -C "$baseline_dir"
cp "$repo_root/src/OptimizationComparison.cpp" "$baseline_dir/src/OptimizationComparison.cpp"

compiler="${CXX:-g++}"
read -r -a compile_flags <<< "${CXXFLAGS:--Wall -O2 -std=c++17}"
common_sources=(
    DenseLayer.cpp NeuralNetwork.cpp Optimizer.cpp LearningEngine.cpp LossFunction.cpp Metrics.cpp
    ImportanceScorer.cpp TrainingScheduler.cpp MomentumOptimizer.cpp AdamOptimizer.cpp
)

baseline_sources=()
current_sources=()
for source in "${common_sources[@]}"; do
    baseline_sources+=("$baseline_dir/src/$source")
    current_sources+=("$repo_root/src/$source")
done

"$compiler" "${compile_flags[@]}" -I"$baseline_dir/src/headers" \
    "$baseline_dir/src/OptimizationComparison.cpp" "${baseline_sources[@]}" \
    -o "$comparison_tmpdir/baseline-benchmark"
"$compiler" "${compile_flags[@]}" -I"$repo_root/src/headers" \
    "$repo_root/src/OptimizationComparison.cpp" "${current_sources[@]}" \
    -o "$comparison_tmpdir/current-benchmark"

output="$repo_root/benchmark_optimization_comparison.csv"
printf 'revision,seed,training_time_ms,validation_loss,mae,rmse,retained_training_bytes,training_allocations,training_allocated_bytes,peak_live_training_bytes,process_peak_rss_bytes,updates\n' > "$output"
for seed in 1234 2027 4242; do
    "$comparison_tmpdir/baseline-benchmark" "baseline_$baseline_commit" "$seed" >> "$output"
    "$comparison_tmpdir/current-benchmark" working_tree "$seed" >> "$output"
done
cat "$output"
