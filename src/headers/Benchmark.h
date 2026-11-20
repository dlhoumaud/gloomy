#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstddef>
#include <string>
#include <vector>

struct BenchmarkResult {
    std::string experiment = "memory_replay";
    std::string memory_strategy;
    std::string precision;
    std::string optimizer;
    std::string loss;
    int seed = 0;
    std::string aggregation = "run";
    size_t memory_capacity = 0;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double mae_stddev = 0.0;
    double rmse_stddev = 0.0;
    double mae_ci95 = 0.0;
    double rmse_ci95 = 0.0;
    double last_value_baseline_mae = 0.0;
    double mae_ratio_vs_full = 0.0;
    double training_time_ms = 0.0;
    double inference_time_us = 0.0;
    double samples_per_second = 0.0;
    double updates_per_second = 0.0;
    double memory_used_bytes = 0.0;
    size_t samples_stored = 0;
    size_t parameter_updates = 0;
    double forgetting = 0.0;
};

class BenchmarkCsv {
public:
    static void write(
        const std::string& path,
        const std::vector<BenchmarkResult>& results
    );
};

#endif // BENCHMARK_H
