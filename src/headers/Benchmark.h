#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstddef>
#include <string>
#include <vector>

struct BenchmarkResult {
    std::string memory_strategy;
    std::string precision;
    std::string optimizer;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double training_time_ms = 0.0;
    double inference_time_us = 0.0;
    size_t memory_used_bytes = 0;
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
