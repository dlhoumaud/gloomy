#include "headers/AdamOptimizer.h"
#include "headers/Benchmark.h"
#include "headers/LearningEngine.h"
#include "headers/Metrics.h"
#include "headers/MomentumOptimizer.h"
#include "headers/FIFOMemory.h"
#include "headers/ReservoirMemory.h"
#include "headers/PrioritizedMemory.h"
#include "headers/LearningMemory.h"
#include "headers/QuantizedFIFOMemory.h"
#include "headers/QuantizedInt8FIFOMemory.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <vector>

namespace {
std::vector<TrainingSample> makeDataset(size_t count, size_t start) {
    std::vector<TrainingSample> samples;
    samples.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const double x = static_cast<double>(index + start) / 10.0;
        samples.push_back({{x}, {2.0 * x + 1.0}});
    }
    return samples;
}

NeuralNetwork makeNetwork() {
    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 8);
    network.addLayer(8, 1);
    return network;
}

std::unique_ptr<Optimizer> makeOptimizer(const std::string& name) {
    if (name == "sgd") {
        return std::make_unique<SGDOptimizer>(0.01);
    }
    if (name == "momentum") {
        return std::make_unique<MomentumOptimizer>(0.001, 0.9);
    }
    return std::make_unique<AdamOptimizer>(0.01);
}

size_t optimizerStateBytes(const Optimizer& optimizer) {
    if (const auto* momentum = dynamic_cast<const MomentumOptimizer*>(&optimizer)) {
        return momentum->stateBytes();
    }
    if (const auto* adam = dynamic_cast<const AdamOptimizer*>(&optimizer)) {
        return adam->stateBytes();
    }
    return 0;
}

size_t parameterBytes(const NeuralNetwork& network) {
    size_t count = 0;
    for (const DenseLayer& layer : network.layers()) {
        for (const auto& row : layer.weights()) {
            count += row.size();
        }
        count += layer.bias().size();
    }
    return count * sizeof(double);
}

size_t sampleBytes(const TrainingSample& sample) {
    return (sample.input.size() + sample.target.size()) * sizeof(double) +
        6 * sizeof(double) + 2 * sizeof(std::size_t);
}

std::unique_ptr<LearningMemory> makeMemory(const std::string& name, size_t capacity) {
    if (name == "fifo") {
        return std::make_unique<FIFOMemory>(capacity);
    }
    if (name == "reservoir") {
        return std::make_unique<ReservoirMemory>(capacity, 1234u);
    }
    return std::make_unique<PrioritizedMemory>(capacity, 0.6, 1234u);
}

BenchmarkResult evaluate(
    NeuralNetwork& network,
    Optimizer& optimizer,
    const LossFunction& loss,
    const std::vector<TrainingSample>& validation,
    const std::string& memory_strategy,
    const std::string& optimizer_name,
    double training_loss,
    double training_time_ms,
    size_t samples_stored,
    size_t memory_used_bytes,
    size_t updates
) {
    double validation_loss = 0.0;
    double mae = 0.0;
    double squared_error = 0.0;
    const auto inference_start = std::chrono::steady_clock::now();
    for (const TrainingSample& sample : validation) {
        const std::vector<double> prediction = network.forward(sample.input);
        validation_loss += loss.compute(prediction, sample.target);
        const RegressionMetrics metric = Metrics::regression(prediction, sample.target);
        mae += metric.mae;
        squared_error += metric.rmse * metric.rmse;
    }
    const auto inference_end = std::chrono::steady_clock::now();

    const double validation_count = static_cast<double>(validation.size());
    BenchmarkResult result;
    result.memory_strategy = memory_strategy;
    result.precision = memory_strategy == "quantized_fifo_int16"
        ? "int16"
        : (memory_strategy == "quantized_fifo_int8" ? "int8" : "float64");
    result.optimizer = optimizer_name;
    result.training_loss = training_loss;
    result.validation_loss = validation_loss / validation_count;
    result.mae = mae / validation_count;
    result.rmse = std::sqrt(squared_error / validation_count);
    result.training_time_ms = training_time_ms;
    result.inference_time_us = std::chrono::duration<double, std::micro>(inference_end - inference_start).count() / validation_count;
    result.memory_used_bytes = memory_used_bytes + parameterBytes(network) + optimizerStateBytes(optimizer);
    result.samples_stored = samples_stored;
    result.parameter_updates = updates;
    return result;
}

double datasetLoss(
    NeuralNetwork& network,
    const LossFunction& loss,
    const std::vector<TrainingSample>& dataset
) {
    double total = 0.0;
    for (const TrainingSample& sample : dataset) {
        total += loss.compute(network.forward(sample.input), sample.target);
    }
    return total / static_cast<double>(dataset.size());
}
}

int main() {
    std::srand(1234);
    const std::vector<TrainingSample> training = makeDataset(80, 0);
    const std::vector<TrainingSample> validation = makeDataset(20, 80);
    const std::vector<std::string> optimizers = {"sgd", "momentum", "adam"};
    std::vector<BenchmarkResult> results;

    for (const std::string& optimizer_name : optimizers) {
        NeuralNetwork network = makeNetwork();
        MSELoss loss;
        std::unique_ptr<Optimizer> optimizer = makeOptimizer(optimizer_name);
        LearningEngine engine(network, loss, *optimizer);

        const auto start = std::chrono::steady_clock::now();
        const double training_loss = engine.train(training, 100, 8);
        const auto end = std::chrono::steady_clock::now();

        results.push_back(evaluate(
            network, *optimizer, loss, validation, "full_dataset", optimizer_name,
            training_loss, std::chrono::duration<double, std::milli>(end - start).count(),
            training.size(), training.empty() ? 0 : training.size() * sampleBytes(training.front()),
            100 * ((training.size() + 7) / 8)
        ));
    }

    const size_t memory_capacity = 16;
    for (const std::string memory_name : {"fifo", "reservoir", "prioritized"}) {
        for (const std::string& optimizer_name : optimizers) {
            NeuralNetwork network = makeNetwork();
            MSELoss loss;
            std::unique_ptr<Optimizer> optimizer = makeOptimizer(optimizer_name);
            std::unique_ptr<LearningMemory> memory = makeMemory(memory_name, memory_capacity);
            LearningEngine engine(network, loss, *optimizer);

            double training_loss = 0.0;
            const auto start = std::chrono::steady_clock::now();
            for (const TrainingSample& sample : training) {
                training_loss += engine.learn(*memory, sample, 8);
            }
            const auto end = std::chrono::steady_clock::now();

            results.push_back(evaluate(
                network, *optimizer, loss, validation, memory_name, optimizer_name,
                training_loss / static_cast<double>(training.size()),
                std::chrono::duration<double, std::milli>(end - start).count(),
                memory->size(), memory->size() * sampleBytes(training.front()),
                training.size()
            ));
        }
    }

    const TrainingSampleQuantizer int16_quantizer =
        TrainingSampleQuantizer::calibrate(training);
    const Int8TrainingSampleQuantizer int8_quantizer =
        Int8TrainingSampleQuantizer::calibrate(training);
    for (const std::string& optimizer_name : optimizers) {
        NeuralNetwork network = makeNetwork();
        MSELoss loss;
        std::unique_ptr<Optimizer> optimizer = makeOptimizer(optimizer_name);
        QuantizedFIFOMemory memory(memory_capacity, int16_quantizer);
        LearningEngine engine(network, loss, *optimizer);

        double training_loss = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (const TrainingSample& sample : training) {
            training_loss += engine.learn(memory, sample, 8);
        }
        const auto end = std::chrono::steady_clock::now();
        results.push_back(evaluate(
            network, *optimizer, loss, validation, "quantized_fifo_int16", optimizer_name,
            training_loss / static_cast<double>(training.size()),
            std::chrono::duration<double, std::milli>(end - start).count(),
            memory.size(), memory.memoryUsedBytes(), training.size()
        ));
    }

    for (const std::string& optimizer_name : optimizers) {
        NeuralNetwork network = makeNetwork();
        MSELoss loss;
        std::unique_ptr<Optimizer> optimizer = makeOptimizer(optimizer_name);
        QuantizedInt8FIFOMemory memory(memory_capacity, int8_quantizer);
        LearningEngine engine(network, loss, *optimizer);

        double training_loss = 0.0;
        const auto start = std::chrono::steady_clock::now();
        for (const TrainingSample& sample : training) {
            training_loss += engine.learn(memory, sample, 8);
        }
        const auto end = std::chrono::steady_clock::now();
        results.push_back(evaluate(
            network, *optimizer, loss, validation, "quantized_fifo_int8", optimizer_name,
            training_loss / static_cast<double>(training.size()),
            std::chrono::duration<double, std::milli>(end - start).count(),
            memory.size(), memory.memoryUsedBytes(), training.size()
        ));
    }

    const std::vector<TrainingSample> regime_a = makeDataset(40, 0);
    std::vector<TrainingSample> regime_b;
    regime_b.reserve(40);
    for (size_t index = 0; index < 40; ++index) {
        const double x = static_cast<double>(index + 40) / 10.0;
        regime_b.push_back({{x}, {-2.0 * x + 20.0}});
    }

    MSELoss forgetting_loss;
    {
        NeuralNetwork network = makeNetwork();
        SGDOptimizer optimizer(0.01);
        LearningEngine engine(network, forgetting_loss, optimizer);
        engine.train(regime_a, 80, 8);
        const double before_b = datasetLoss(network, forgetting_loss, regime_a);
        engine.train(regime_b, 80, 8);
        const double after_b = datasetLoss(network, forgetting_loss, regime_a);

        BenchmarkResult result;
        result.memory_strategy = "forgetting_no_replay";
        result.precision = "float64";
        result.optimizer = "sgd";
        result.validation_loss = after_b;
        result.forgetting = Metrics::forgetting(before_b, after_b);
        result.samples_stored = regime_b.size();
        result.parameter_updates = 160 * ((regime_a.size() + 7) / 8);
        result.memory_used_bytes = parameterBytes(network);
        results.push_back(result);
    }

    {
        NeuralNetwork network = makeNetwork();
        SGDOptimizer optimizer(0.01);
        LearningEngine engine(network, forgetting_loss, optimizer);
        FIFOMemory memory(regime_a.size());
        engine.train(regime_a, 80, 8);
        const double before_b = datasetLoss(network, forgetting_loss, regime_a);
        for (const TrainingSample& sample : regime_a) {
            memory.add(sample);
        }
        engine.train(regime_b, 80, 8);
        const double after_b = datasetLoss(network, forgetting_loss, regime_a);
        for (size_t update = 0; update < 20; ++update) {
            engine.trainFromMemory(memory, 8);
        }
        const double after_replay = datasetLoss(network, forgetting_loss, regime_a);

        BenchmarkResult result;
        result.memory_strategy = "forgetting_fifo_replay";
        result.precision = "float64";
        result.optimizer = "sgd";
        result.validation_loss = after_replay;
        result.forgetting = Metrics::forgetting(before_b, after_replay);
        result.samples_stored = memory.size();
        result.parameter_updates = 160 * ((regime_a.size() + 7) / 8) + 20;
        result.memory_used_bytes = memory.size() * sampleBytes(regime_a.front()) + parameterBytes(network);
        results.push_back(result);
        (void)after_b;
    }

    BenchmarkCsv::write("benchmark_results.csv", results);
    return 0;
}
