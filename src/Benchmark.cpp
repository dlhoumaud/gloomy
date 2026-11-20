#include "headers/Benchmark.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace {
void writeField(std::ofstream& stream, const std::string& value) {
    bool needs_quotes = false;
    for (char character : value) {
        if (character == ',' || character == '"' || character == '\n') {
            needs_quotes = true;
            break;
        }
    }
    if (!needs_quotes) {
        stream << value;
        return;
    }

    stream << '"';
    for (char character : value) {
        if (character == '"') {
            stream << '"';
        }
        stream << character;
    }
    stream << '"';
}
}

void BenchmarkCsv::write(
    const std::string& path,
    const std::vector<BenchmarkResult>& results
) {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Unable to open benchmark CSV for writing");
    }

    stream << "memory_strategy,precision,optimizer,experiment,loss,seed,aggregation,memory_capacity,"
              "training_loss,validation_loss,mae,rmse,mae_stddev,rmse_stddev,mae_ci95,rmse_ci95,"
              "last_value_baseline_mae,mae_ratio_vs_full,phase_a_mae_after_a,phase_a_mae_after_b,"
              "phase_a_mae_after_recovery,phase_b_mae_after_b,phase_b_mae_after_recovery,recovery,"
              "recovery_stddev,recovery_ci95,forgetting_stddev,forgetting_ci95,training_time_ms,inference_time_us,"
              "samples_per_second,updates_per_second,memory_used_bytes,samples_stored,"
              "parameter_updates,forgetting\n";
    stream << std::setprecision(17);
    for (const BenchmarkResult& result : results) {
        writeField(stream, result.memory_strategy);
        stream << ',';
        writeField(stream, result.precision);
        stream << ',';
        writeField(stream, result.optimizer);
        stream << ',';
        writeField(stream, result.experiment);
        stream << ',';
        writeField(stream, result.loss);
        stream << ',' << result.seed << ',';
        writeField(stream, result.aggregation);
        stream << ',' << result.memory_capacity
               << ',' << result.training_loss
               << ',' << result.validation_loss
               << ',' << result.mae
               << ',' << result.rmse
               << ',' << result.mae_stddev
               << ',' << result.rmse_stddev
               << ',' << result.mae_ci95
               << ',' << result.rmse_ci95
               << ',' << result.last_value_baseline_mae
               << ',' << result.mae_ratio_vs_full
               << ',' << result.phase_a_mae_after_a
               << ',' << result.phase_a_mae_after_b
               << ',' << result.phase_a_mae_after_recovery
               << ',' << result.phase_b_mae_after_b
               << ',' << result.phase_b_mae_after_recovery
               << ',' << result.recovery
               << ',' << result.recovery_stddev
               << ',' << result.recovery_ci95
               << ',' << result.forgetting_stddev
               << ',' << result.forgetting_ci95
               << ',' << result.training_time_ms
               << ',' << result.inference_time_us
               << ',' << result.samples_per_second
               << ',' << result.updates_per_second
               << ',' << result.memory_used_bytes
               << ',' << result.samples_stored
               << ',' << result.parameter_updates
               << ',' << result.forgetting << '\n';
    }
    if (!stream) {
        throw std::runtime_error("Unable to write benchmark CSV");
    }
}
