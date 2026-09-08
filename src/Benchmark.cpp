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

    stream << "memory_strategy,precision,optimizer,training_loss,validation_loss,mae,rmse,"
              "training_time_ms,inference_time_us,memory_used_bytes,samples_stored,"
              "parameter_updates,forgetting,memory_capacity,mae_ratio_to_full_dataset\n";
    stream << std::setprecision(17);
    for (const BenchmarkResult& result : results) {
        writeField(stream, result.memory_strategy);
        stream << ',';
        writeField(stream, result.precision);
        stream << ',';
        writeField(stream, result.optimizer);
        stream << ',' << result.training_loss
               << ',' << result.validation_loss
               << ',' << result.mae
               << ',' << result.rmse
               << ',' << result.training_time_ms
               << ',' << result.inference_time_us
               << ',' << result.memory_used_bytes
               << ',' << result.samples_stored
               << ',' << result.parameter_updates
               << ',' << result.forgetting
               << ',' << result.memory_capacity
               << ',' << result.mae_ratio_to_full_dataset << '\n';
    }
    if (!stream) {
        throw std::runtime_error("Unable to write benchmark CSV");
    }
}
