#include "headers/AdamOptimizer.h"
#include "headers/Benchmark.h"
#include "headers/FIFOMemory.h"
#include "headers/HybridMemory.h"
#include "headers/LearningEngine.h"
#include "headers/LearningMemory.h"
#include "headers/Metrics.h"
#include "headers/MomentumOptimizer.h"
#include "headers/NoveltyMemory.h"
#include "headers/PrioritizedMemory.h"
#include "headers/QuantizedFIFOMemory.h"
#include "headers/QuantizedInt8FIFOMemory.h"
#include "headers/ReservoirMemory.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace {
constexpr size_t training_count = 400;
constexpr size_t validation_count = 80;
constexpr size_t replay_batch = 8;
constexpr size_t train_every = 8;
const std::vector<size_t> capacities = {32, 64, 128, 256};
const std::vector<std::uint32_t> seeds = {1234u, 2027u, 4242u};
const std::vector<std::string> optimizer_names = {"sgd", "momentum", "adam"};
const std::vector<std::string> loss_names = {"mse", "mae", "huber"};

std::vector<TrainingSample> makeDataset(size_t count, size_t start) {
    std::vector<TrainingSample> samples;
    samples.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const double x = static_cast<double>(index + start) / 1000.0;
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
    if (name == "sgd") return std::make_unique<SGDOptimizer>(0.01);
    if (name == "momentum") return std::make_unique<MomentumOptimizer>(0.001, 0.9);
    return std::make_unique<AdamOptimizer>(0.01);
}

std::unique_ptr<LossFunction> makeLoss(const std::string& name) {
    if (name == "mae") return std::make_unique<MAELoss>();
    if (name == "huber") return std::make_unique<HuberLoss>(1.0);
    return std::make_unique<MSELoss>();
}

size_t optimizerStateBytes(const Optimizer& optimizer) {
    if (const auto* momentum = dynamic_cast<const MomentumOptimizer*>(&optimizer)) return momentum->stateBytes();
    if (const auto* adam = dynamic_cast<const AdamOptimizer*>(&optimizer)) return adam->stateBytes();
    return 0;
}

size_t parameterBytes(const NeuralNetwork& network) {
    size_t count = 0;
    for (const DenseLayer& layer : network.layers()) {
        for (const auto& row : layer.weights()) count += row.size();
        count += layer.bias().size();
    }
    return count * sizeof(double);
}

size_t sampleBytes(const TrainingSample& sample) {
    return (sample.input.size() + sample.target.size()) * sizeof(double) +
        6 * sizeof(double) + 2 * sizeof(std::size_t);
}

std::unique_ptr<LearningMemory> makeMemory(
    const std::string& name,
    const std::string& precision,
    size_t capacity,
    const std::vector<TrainingSample>& calibration
) {
    if (name == "fifo" && precision == "int16")
        return std::make_unique<QuantizedFIFOMemory>(capacity, TrainingSampleQuantizer::calibrate(calibration));
    if (name == "fifo" && precision == "int8")
        return std::make_unique<QuantizedInt8FIFOMemory>(capacity, Int8TrainingSampleQuantizer::calibrate(calibration));
    if (name == "fifo") return std::make_unique<FIFOMemory>(capacity);
    if (name == "reservoir") return std::make_unique<ReservoirMemory>(capacity, 1234u);
    if (name == "prioritized") return std::make_unique<PrioritizedMemory>(capacity, 0.6, 1234u);
    if (name == "novelty") return std::make_unique<NoveltyMemory>(capacity, 0.000025);
    HybridMemoryRatios ratios{0.2, 0.2, 0.2, 0.4};
    return std::make_unique<HybridMemory>(capacity, ratios, 0.000025, 1234u);
}

double datasetLoss(NeuralNetwork& network, const LossFunction& loss, const std::vector<TrainingSample>& data) {
    double total = 0.0;
    for (const auto& sample : data) total += loss.compute(network.forward(sample.input), sample.target);
    return total / static_cast<double>(data.size());
}

double lastValueBaselineMae(const std::vector<TrainingSample>& validation) {
    double total = 0.0;
    for (const auto& sample : validation)
        total += std::abs(sample.input.back() - sample.target.front());
    return total / static_cast<double>(validation.size());
}

BenchmarkResult evaluate(
    NeuralNetwork& network,
    Optimizer& optimizer,
    const LossFunction& loss,
    const std::vector<TrainingSample>& validation,
    const std::string& strategy,
    const std::string& precision,
    const std::string& optimizer_name,
    const std::string& loss_name,
    int seed,
    size_t capacity,
    double training_loss,
    double training_time_ms,
    size_t samples_processed,
    size_t samples_stored,
    size_t memory_bytes,
    size_t updates,
    double baseline_mae
) {
    double validation_loss = 0.0, mae = 0.0, squared_error = 0.0;
    const auto inference_start = std::chrono::steady_clock::now();
    for (const auto& sample : validation) {
        const auto prediction = network.forward(sample.input);
        validation_loss += loss.compute(prediction, sample.target);
        const auto metrics = Metrics::regression(prediction, sample.target);
        mae += metrics.mae;
        squared_error += metrics.rmse * metrics.rmse;
    }
    const auto inference_end = std::chrono::steady_clock::now();
    const double elapsed_seconds = training_time_ms / 1000.0;
    BenchmarkResult result;
    result.experiment = "memory_replay";
    result.memory_strategy = strategy;
    result.precision = precision;
    result.optimizer = optimizer_name;
    result.loss = loss_name;
    result.seed = seed;
    result.memory_capacity = capacity;
    result.training_loss = training_loss;
    result.validation_loss = validation_loss / static_cast<double>(validation.size());
    result.mae = mae / static_cast<double>(validation.size());
    result.rmse = std::sqrt(squared_error / static_cast<double>(validation.size()));
    result.last_value_baseline_mae = baseline_mae;
    result.training_time_ms = training_time_ms;
    result.inference_time_us = std::chrono::duration<double, std::micro>(inference_end - inference_start).count() /
        static_cast<double>(validation.size());
    result.samples_per_second = elapsed_seconds > 0.0 ? samples_processed / elapsed_seconds : 0.0;
    result.updates_per_second = elapsed_seconds > 0.0 ? updates / elapsed_seconds : 0.0;
    result.memory_used_bytes = memory_bytes + parameterBytes(network) + optimizerStateBytes(optimizer);
    result.samples_stored = samples_stored;
    result.parameter_updates = updates;
    return result;
}

using GroupKey = std::tuple<std::string, std::string, std::string, std::string, size_t>;
GroupKey groupKey(const BenchmarkResult& row) {
    return {row.memory_strategy, row.precision, row.optimizer, row.loss, row.memory_capacity};
}

std::vector<BenchmarkResult> addSeedSummaries(const std::vector<BenchmarkResult>& runs) {
    std::map<GroupKey, std::vector<BenchmarkResult>> groups;
    for (const auto& row : runs) groups[groupKey(row)].push_back(row);
    std::vector<BenchmarkResult> summaries;
    for (const auto& pair : groups) {
        const auto& rows = pair.second;
        if (rows.size() < 2) continue;
        BenchmarkResult mean = rows.front();
        mean.seed = -1;
        mean.aggregation = "mean";
        mean.mae_stddev = mean.rmse_stddev = 0.0;
        mean.mae_ci95 = mean.rmse_ci95 = 0.0;
        const double denominator = static_cast<double>(rows.size());
        mean.training_loss = mean.validation_loss = mean.mae = mean.rmse = 0.0;
        mean.last_value_baseline_mae = mean.mae_ratio_vs_full = 0.0;
        mean.training_time_ms = mean.inference_time_us = 0.0;
        mean.samples_per_second = mean.updates_per_second = mean.memory_used_bytes = 0.0;
        mean.samples_stored = mean.parameter_updates = 0;
        for (const auto& row : rows) {
            mean.training_loss += row.training_loss;
            mean.validation_loss += row.validation_loss;
            mean.mae += row.mae; mean.rmse += row.rmse;
            mean.last_value_baseline_mae += row.last_value_baseline_mae;
            mean.mae_ratio_vs_full += row.mae_ratio_vs_full;
            mean.training_time_ms += row.training_time_ms;
            mean.inference_time_us += row.inference_time_us;
            mean.samples_per_second += row.samples_per_second;
            mean.updates_per_second += row.updates_per_second;
            mean.memory_used_bytes += row.memory_used_bytes;
            mean.samples_stored += row.samples_stored;
            mean.parameter_updates += row.parameter_updates;
        }
        mean.training_loss /= denominator; mean.validation_loss /= denominator;
        mean.mae /= denominator; mean.rmse /= denominator;
        mean.last_value_baseline_mae /= denominator; mean.mae_ratio_vs_full /= denominator;
        mean.training_time_ms /= denominator; mean.inference_time_us /= denominator;
        mean.samples_per_second /= denominator; mean.updates_per_second /= denominator;
        mean.memory_used_bytes /= denominator;
        mean.samples_stored = static_cast<size_t>(std::llround(mean.samples_stored / denominator));
        mean.parameter_updates = static_cast<size_t>(std::llround(mean.parameter_updates / denominator));
        double mae_variance = 0.0, rmse_variance = 0.0;
        for (const auto& row : rows) {
            mae_variance += std::pow(row.mae - mean.mae, 2.0);
            rmse_variance += std::pow(row.rmse - mean.rmse, 2.0);
        }
        const double sample_denominator = static_cast<double>(rows.size() - 1);
        mean.mae_stddev = std::sqrt(mae_variance / sample_denominator);
        mean.rmse_stddev = std::sqrt(rmse_variance / sample_denominator);
        mean.mae_ci95 = 1.96 * mean.mae_stddev / std::sqrt(denominator);
        mean.rmse_ci95 = 1.96 * mean.rmse_stddev / std::sqrt(denominator);
        summaries.push_back(mean);
    }
    return summaries;
}

BenchmarkResult runBaseline(const std::vector<TrainingSample>& validation, int seed, double baseline_mae) {
    BenchmarkResult result;
    result.memory_strategy = "last_value_baseline";
    result.precision = "float64";
    result.optimizer = "none";
    result.loss = "none";
    result.seed = seed;
    result.experiment = "baseline";
    result.last_value_baseline_mae = baseline_mae;
    result.mae = baseline_mae;
    double squared = 0.0;
    for (const auto& sample : validation) {
        const double error = sample.input.back() - sample.target.front();
        squared += error * error;
    }
    result.rmse = std::sqrt(squared / validation.size());
    result.validation_loss = result.rmse * result.rmse;
    return result;
}

void runForgettingExperiment(std::vector<BenchmarkResult>& results) {
    const auto regime_a = makeDataset(40, 0);
    std::vector<TrainingSample> regime_b;
    regime_b.reserve(40);
    for (size_t index = 0; index < 40; ++index) {
        const double x = static_cast<double>(index + 40) / 10.0;
        regime_b.push_back({{x}, {-2.0 * x + 20.0}});
    }
    MSELoss loss;
    {
        std::srand(1234);
        NeuralNetwork network = makeNetwork(); SGDOptimizer optimizer(0.01);
        LearningEngine engine(network, loss, optimizer);
        engine.train(regime_a, 80, replay_batch);
        const double before = datasetLoss(network, loss, regime_a);
        engine.train(regime_b, 80, replay_batch);
        const double after = datasetLoss(network, loss, regime_a);
        BenchmarkResult row;
        row.experiment = "catastrophic_forgetting"; row.memory_strategy = "forgetting_no_replay";
        row.precision = "float64"; row.optimizer = "sgd"; row.loss = "mse";
        row.seed = 1234; row.validation_loss = after; row.forgetting = Metrics::forgetting(before, after);
        row.samples_stored = regime_b.size(); row.parameter_updates = 160 * ((regime_a.size() + replay_batch - 1) / replay_batch);
        row.memory_used_bytes = parameterBytes(network); results.push_back(row);
    }
    {
        std::srand(1234);
        NeuralNetwork network = makeNetwork(); SGDOptimizer optimizer(0.01);
        LearningEngine engine(network, loss, optimizer); FIFOMemory memory(regime_a.size());
        engine.train(regime_a, 80, replay_batch);
        const double before = datasetLoss(network, loss, regime_a);
        for (const auto& sample : regime_a) memory.add(sample);
        engine.train(regime_b, 80, replay_batch);
        for (size_t update = 0; update < 20; ++update) engine.trainFromMemory(memory, replay_batch);
        const double after = datasetLoss(network, loss, regime_a);
        BenchmarkResult row;
        row.experiment = "catastrophic_forgetting"; row.memory_strategy = "forgetting_fifo_replay";
        row.precision = "float64"; row.optimizer = "sgd"; row.loss = "mse";
        row.seed = 1234; row.validation_loss = after; row.forgetting = Metrics::forgetting(before, after);
        row.samples_stored = memory.size(); row.parameter_updates = 160 * ((regime_a.size() + replay_batch - 1) / replay_batch) + 20;
        row.memory_used_bytes = memory.size() * sampleBytes(regime_a.front()) + parameterBytes(network);
        results.push_back(row);
    }
}
}

int main() {
    const auto training = makeDataset(training_count, 0);
    const auto validation = makeDataset(validation_count, training_count);
    const double baseline_mae = lastValueBaselineMae(validation);
    std::vector<BenchmarkResult> all_results;
    std::map<std::tuple<int, std::string, std::string>, double> full_mae;
    std::vector<BenchmarkResult> baseline_runs;
    std::vector<BenchmarkResult> full_runs;

    for (std::uint32_t seed : seeds) {
        baseline_runs.push_back(runBaseline(validation, static_cast<int>(seed), baseline_mae));
        for (const auto& optimizer_name : optimizer_names) {
            for (const auto& loss_name : loss_names) {
                std::srand(seed);
                NeuralNetwork network = makeNetwork();
                auto optimizer = makeOptimizer(optimizer_name);
                auto loss = makeLoss(loss_name);
                LearningEngine engine(network, *loss, *optimizer);
                const auto start = std::chrono::steady_clock::now();
                engine.train(training, 1, replay_batch);
                const auto end = std::chrono::steady_clock::now();
                const size_t updates = (training.size() + replay_batch - 1) / replay_batch;
                const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                const size_t memory_bytes = training.size() * sampleBytes(training.front());
                BenchmarkResult row = evaluate(network, *optimizer, *loss, validation,
                    "full_dataset", "float64", optimizer_name, loss_name, static_cast<int>(seed),
                    training.size(), datasetLoss(network, *loss, training), elapsed_ms,
                    training.size(), training.size(), memory_bytes, updates, baseline_mae);
                full_mae[{static_cast<int>(seed), optimizer_name, loss_name}] = row.mae;
                full_runs.push_back(row);
            }
        }
    }

    all_results.insert(all_results.end(), baseline_runs.begin(), baseline_runs.end());
    const auto baseline_summaries = addSeedSummaries(baseline_runs);
    all_results.insert(all_results.end(), baseline_summaries.begin(), baseline_summaries.end());
    all_results.insert(all_results.end(), full_runs.begin(), full_runs.end());
    const std::vector<std::pair<std::string, std::string>> memory_variants = {
        {"fifo", "float64"}, {"reservoir", "float64"}, {"prioritized", "float64"},
        {"novelty", "float64"}, {"hybrid", "float64"}, {"fifo", "int16"}, {"fifo", "int8"}
    };
    for (size_t capacity : capacities) {
        for (const auto& variant : memory_variants) {
            for (const auto& optimizer_name : optimizer_names) {
                for (const auto& loss_name : loss_names) {
                    std::vector<BenchmarkResult> seed_runs;
                    for (std::uint32_t seed : seeds) {
                        std::srand(seed);
                        NeuralNetwork network = makeNetwork();
                        auto optimizer = makeOptimizer(optimizer_name);
                        auto loss = makeLoss(loss_name);
                        auto memory = makeMemory(variant.first, variant.second, capacity, training);
                        LearningEngine engine(network, *loss, *optimizer);
                        EveryNScheduler scheduler(train_every);
                        const auto start = std::chrono::steady_clock::now();
                        for (const auto& sample : training)
                            engine.learn(*memory, sample, replay_batch, scheduler);
                        const auto end = std::chrono::steady_clock::now();
                        const size_t updates = (training.size() + train_every - 1) / train_every;
                        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                        size_t memory_bytes = memory->size() * sampleBytes(training.front());
                        if (const auto* quantized = dynamic_cast<const QuantizedFIFOMemory*>(memory.get()))
                            memory_bytes = quantized->memoryUsedBytes();
                        if (const auto* quantized = dynamic_cast<const QuantizedInt8FIFOMemory*>(memory.get()))
                            memory_bytes = quantized->memoryUsedBytes();
                        BenchmarkResult row = evaluate(network, *optimizer, *loss, validation,
                            variant.first, variant.second, optimizer_name, loss_name, static_cast<int>(seed),
                            capacity, datasetLoss(network, *loss, training), elapsed_ms,
                            training.size(), memory->size(), memory_bytes, updates, baseline_mae);
                        const double reference_mae = full_mae.at({static_cast<int>(seed), optimizer_name, loss_name});
                        row.mae_ratio_vs_full = reference_mae > 1e-12 ? row.mae / reference_mae : 0.0;
                        seed_runs.push_back(row);
                    }
                    all_results.insert(all_results.end(), seed_runs.begin(), seed_runs.end());
                    const auto summaries = addSeedSummaries(seed_runs);
                    all_results.insert(all_results.end(), summaries.begin(), summaries.end());
                }
            }
        }
    }

    const auto full_summaries = addSeedSummaries(full_runs);
    all_results.insert(all_results.end(), full_summaries.begin(), full_summaries.end());
    runForgettingExperiment(all_results);

    std::vector<BenchmarkResult> memory_results, forgetting_results;
    for (const auto& row : all_results) {
        if (row.experiment == "memory_replay" || row.experiment == "baseline") memory_results.push_back(row);
        if (row.experiment == "catastrophic_forgetting") forgetting_results.push_back(row);
    }
    BenchmarkCsv::write("benchmark_results.csv", all_results);
    BenchmarkCsv::write("benchmark_memory.csv", memory_results);
    BenchmarkCsv::write("benchmark_forgetting.csv", forgetting_results);
    return 0;
}
