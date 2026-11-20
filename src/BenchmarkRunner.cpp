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
#include <new>
#include <tuple>
#include <vector>

namespace {
bool count_allocations = false;
size_t tracked_allocations = 0;
volatile double profile_sink = 0.0;
}

void* operator new(std::size_t size) {
    if (void* memory = std::malloc(size == 0 ? 1 : size)) {
        if (count_allocations) ++tracked_allocations;
        return memory;
    }
    throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
    if (void* memory = std::malloc(size == 0 ? 1 : size)) {
        if (count_allocations) ++tracked_allocations;
        return memory;
    }
    throw std::bad_alloc();
}

[[gnu::noinline]] void releaseAllocation(void* memory) noexcept { std::free(memory); }
void operator delete(void* memory) noexcept { releaseAllocation(memory); }
void operator delete[](void* memory) noexcept { releaseAllocation(memory); }
void operator delete(void* memory, std::size_t) noexcept { releaseAllocation(memory); }
void operator delete[](void* memory, std::size_t) noexcept { releaseAllocation(memory); }

namespace {
constexpr size_t training_count = 400;
constexpr size_t validation_count = 80;
constexpr size_t replay_batch = 8;
constexpr size_t train_every = 8;
const std::vector<size_t> capacities = {32, 64, 128, 256};
const std::vector<std::uint32_t> seeds = {1234u, 2027u, 4242u};
const std::vector<std::string> optimizer_names = {"sgd", "momentum", "adam"};
const std::vector<std::string> loss_names = {"mse", "mae", "huber"};

std::vector<TrainingSample> makeDataset(size_t count, size_t start);
NeuralNetwork makeNetwork();
std::unique_ptr<Optimizer> makeOptimizer(const std::string& name);
std::unique_ptr<LossFunction> makeLoss(const std::string& name);

template <typename Operation>
BenchmarkProfileResult profileOperation(
    const std::string& workload,
    const std::string& name,
    size_t iterations,
    Operation operation
) {
    for (size_t index = 0; index < 100; ++index) operation();

    std::vector<double> latencies;
    latencies.reserve(iterations);
    double total_us = 0.0;
    for (size_t index = 0; index < iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        operation();
        const auto end = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double, std::micro>(end - start).count();
        latencies.push_back(elapsed);
        total_us += elapsed;
    }

    tracked_allocations = 0;
    count_allocations = true;
    for (size_t index = 0; index < iterations; ++index) operation();
    count_allocations = false;

    std::sort(latencies.begin(), latencies.end());
    const size_t p50_index = (latencies.size() - 1) / 2;
    const size_t p95_index = static_cast<size_t>(std::ceil(0.95 * latencies.size())) - 1;
    return {workload, name, iterations, total_us / iterations,
        latencies[p50_index], latencies[p95_index],
        static_cast<double>(tracked_allocations) / iterations};
}

std::vector<BenchmarkProfileResult> profileCoreOperations() {
    const std::string workload = "dense_1_8_1_float64";
    NeuralNetwork network = makeNetwork();
    std::srand(1234u);
    auto loss = makeLoss("mse");
    auto optimizer = makeOptimizer("sgd");
    LearningEngine engine(network, *loss, *optimizer);
    const auto samples = makeDataset(replay_batch, 0);
    FIFOMemory memory(64);
    for (size_t index = 0; index < 64; ++index) memory.add(samples[index % samples.size()]);
    std::vector<double> gradient_buffer;

    std::vector<BenchmarkProfileResult> results;
    results.push_back(profileOperation(workload, "forward", 10000, [&]() {
        const auto output = network.forward(samples.front().input);
        profile_sink += output.front();
    }));
    results.push_back(profileOperation(workload, "forward_inference", 10000, [&]() {
        const auto& output = network.forwardInferenceRef(samples.front().input);
        profile_sink += output.front();
    }));
    results.push_back(profileOperation(workload, "loss_gradient", 10000, [&]() {
        loss->gradientInto(network.forwardRef(samples.front().input), samples.front().target, gradient_buffer);
        profile_sink += gradient_buffer.front();
    }));
    results.push_back(profileOperation(workload, "train_batch_update", 1000, [&]() {
        profile_sink += engine.trainBatch(samples);
    }));
    results.push_back(profileOperation(workload, "memory_replay", 1000, [&]() {
        profile_sink += engine.trainFromMemory(memory, replay_batch);
    }));
    results.push_back(profileOperation(workload, "minibatch_copy", 5000, [&]() {
        const auto copied_batch = samples;
        profile_sink += copied_batch.front().target.front();
    }));
    return results;
}

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
        count += layer.weightValues().size();
        count += layer.bias().size();
    }
    return count * sizeof(double);
}

size_t sampleBytes(const TrainingSample& sample) {
    return sizeof(sample) + trainingSampleDynamicBytes(sample);
}

std::unique_ptr<LearningMemory> makeMemory(
    const std::string& name,
    const std::string& precision,
    size_t capacity,
    const std::vector<TrainingSample>& calibration,
    std::uint32_t seed
) {
    if (name == "fifo" && precision == "int16")
        return std::make_unique<QuantizedFIFOMemory>(capacity, TrainingSampleQuantizer::calibrate(calibration));
    if (name == "fifo" && precision == "int8")
        return std::make_unique<QuantizedInt8FIFOMemory>(capacity, Int8TrainingSampleQuantizer::calibrate(calibration));
    if (name == "fifo") return std::make_unique<FIFOMemory>(capacity);
    if (name == "reservoir") return std::make_unique<ReservoirMemory>(capacity, seed);
    if (name == "prioritized") return std::make_unique<PrioritizedMemory>(capacity, 0.6, seed);
    if (name == "novelty") return std::make_unique<NoveltyMemory>(capacity, 0.000025);
    HybridMemoryRatios ratios{0.2, 0.2, 0.2, 0.4};
    return std::make_unique<HybridMemory>(capacity, ratios, 0.000025, seed);
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

using GroupKey = std::tuple<std::string, std::string, std::string, std::string, size_t, std::string>;
GroupKey groupKey(const BenchmarkResult& row) {
    return {row.memory_strategy, row.precision, row.optimizer, row.loss, row.memory_capacity, row.experiment};
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
        mean.phase_a_mae_after_a = mean.phase_a_mae_after_b = 0.0;
        mean.phase_a_mae_after_recovery = mean.phase_b_mae_after_b = 0.0;
        mean.phase_b_mae_after_recovery = mean.recovery = mean.forgetting = 0.0;
        mean.training_time_ms = mean.inference_time_us = 0.0;
        mean.samples_per_second = mean.updates_per_second = mean.memory_used_bytes = 0.0;
        mean.samples_stored = mean.parameter_updates = 0;
        for (const auto& row : rows) {
            mean.training_loss += row.training_loss;
            mean.validation_loss += row.validation_loss;
            mean.mae += row.mae; mean.rmse += row.rmse;
            mean.last_value_baseline_mae += row.last_value_baseline_mae;
            mean.mae_ratio_vs_full += row.mae_ratio_vs_full;
            mean.phase_a_mae_after_a += row.phase_a_mae_after_a;
            mean.phase_a_mae_after_b += row.phase_a_mae_after_b;
            mean.phase_a_mae_after_recovery += row.phase_a_mae_after_recovery;
            mean.phase_b_mae_after_b += row.phase_b_mae_after_b;
            mean.phase_b_mae_after_recovery += row.phase_b_mae_after_recovery;
            mean.recovery += row.recovery; mean.forgetting += row.forgetting;
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
        mean.phase_a_mae_after_a /= denominator; mean.phase_a_mae_after_b /= denominator;
        mean.phase_a_mae_after_recovery /= denominator; mean.phase_b_mae_after_b /= denominator;
        mean.phase_b_mae_after_recovery /= denominator;
        mean.recovery /= denominator; mean.forgetting /= denominator;
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
        double forgetting_variance = 0.0, recovery_variance = 0.0;
        for (const auto& row : rows) {
            forgetting_variance += std::pow(row.forgetting - mean.forgetting, 2.0);
            recovery_variance += std::pow(row.recovery - mean.recovery, 2.0);
        }
        mean.forgetting_stddev = std::sqrt(forgetting_variance / sample_denominator);
        mean.recovery_stddev = std::sqrt(recovery_variance / sample_denominator);
        mean.forgetting_ci95 = 1.96 * mean.forgetting_stddev / std::sqrt(denominator);
        mean.recovery_ci95 = 1.96 * mean.recovery_stddev / std::sqrt(denominator);
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
        row.memory_used_bytes = memory.allocatedBytes() + parameterBytes(network);
        results.push_back(row);
    }
}

std::vector<TrainingSample> makeConceptRegime(bool regime_a) {
    std::vector<TrainingSample> samples;
    samples.reserve(40);
    for (size_t index = 0; index < 40; ++index) {
        const double x = static_cast<double>(index) / 100.0;
        samples.push_back({{x}, {(regime_a ? 2.0 : -2.0) * x + 1.0}});
    }
    return samples;
}

double datasetMae(NeuralNetwork& network, const std::vector<TrainingSample>& data) {
    double total = 0.0;
    for (const auto& sample : data) {
        const auto prediction = network.forward(sample.input);
        total += Metrics::regression(prediction, sample.target).mae;
    }
    return total / static_cast<double>(data.size());
}

double datasetRmse(NeuralNetwork& network, const std::vector<TrainingSample>& data) {
    double squared_error = 0.0;
    size_t value_count = 0;
    for (const auto& sample : data) {
        const auto prediction = network.forward(sample.input);
        for (size_t index = 0; index < prediction.size(); ++index) {
            const double error = prediction[index] - sample.target[index];
            squared_error += error * error;
            ++value_count;
        }
    }
    return std::sqrt(squared_error / static_cast<double>(value_count));
}

void runConceptDriftExperiment(std::vector<BenchmarkResult>& results) {
    const auto regime_a = makeConceptRegime(true);
    const auto regime_b = makeConceptRegime(false);
    const std::vector<std::pair<std::string, std::string>> strategies = {
        {"no_replay", "float64"}, {"fifo", "float64"}, {"reservoir", "float64"},
        {"prioritized", "float64"}, {"novelty", "float64"}, {"hybrid", "float64"}
    };
    std::vector<BenchmarkResult> seed_runs;
    for (std::uint32_t seed : seeds) {
        for (const auto& strategy : strategies) {
            std::srand(seed);
            NeuralNetwork network = makeNetwork();
            SGDOptimizer optimizer(0.01);
            MSELoss loss;
            LearningEngine engine(network, loss, optimizer);
            std::unique_ptr<LearningMemory> memory;
            EveryNScheduler scheduler(train_every);
            if (strategy.first != "no_replay") {
                std::vector<TrainingSample> calibration = regime_a;
                calibration.insert(calibration.end(), regime_b.begin(), regime_b.end());
                memory = makeMemory(strategy.first, "float64", 64, calibration, seed);
            }
            auto trainPhase = [&](const std::vector<TrainingSample>& phase) {
                if (!memory) {
                    engine.train(phase, 10, replay_batch);
                    return;
                }
                for (size_t epoch = 0; epoch < 10; ++epoch)
                    for (const auto& sample : phase) engine.learn(*memory, sample, replay_batch, scheduler);
            };

            trainPhase(regime_a);
            const double a_after_a = datasetMae(network, regime_a);
            trainPhase(regime_b);
            const double a_after_b = datasetMae(network, regime_a);
            const double b_after_b = datasetMae(network, regime_b);
            trainPhase(regime_a);
            const double a_after_recovery = datasetMae(network, regime_a);
            const double b_after_recovery = datasetMae(network, regime_b);

            BenchmarkResult row;
            row.experiment = "concept_drift_a_b_a";
            row.memory_strategy = strategy.first;
            row.precision = strategy.second;
            row.optimizer = "sgd";
            row.loss = "mse";
            row.seed = static_cast<int>(seed);
            row.memory_capacity = memory ? memory->capacity() : 0;
            row.phase_a_mae_after_a = a_after_a;
            row.phase_a_mae_after_b = a_after_b;
            row.phase_a_mae_after_recovery = a_after_recovery;
            row.phase_b_mae_after_b = b_after_b;
            row.phase_b_mae_after_recovery = b_after_recovery;
            row.forgetting = a_after_a - a_after_b;
            row.recovery = a_after_b - a_after_recovery;
            row.validation_loss = a_after_recovery;
            row.mae = a_after_recovery;
            row.rmse = datasetRmse(network, regime_a);
            row.training_loss = datasetLoss(network, loss, regime_a);
            row.samples_stored = memory ? memory->size() : 0;
            row.parameter_updates = 3 * 10 * ((regime_a.size() + train_every - 1) / train_every);
            row.memory_used_bytes = parameterBytes(network) + optimizerStateBytes(optimizer);
            if (memory) row.memory_used_bytes += memory->allocatedBytes();
            seed_runs.push_back(row);
        }
    }
    results.insert(results.end(), seed_runs.begin(), seed_runs.end());
    const auto summaries = addSeedSummaries(seed_runs);
    results.insert(results.end(), summaries.begin(), summaries.end());
}
}

int main() {
    const auto profile_results = profileCoreOperations();
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
                        auto memory = makeMemory(variant.first, variant.second, capacity, training, seed);
                        LearningEngine engine(network, *loss, *optimizer);
                        EveryNScheduler scheduler(train_every);
                        const auto start = std::chrono::steady_clock::now();
                        for (const auto& sample : training)
                            engine.learn(*memory, sample, replay_batch, scheduler);
                        const auto end = std::chrono::steady_clock::now();
                        const size_t updates = (training.size() + train_every - 1) / train_every;
                        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
                        const size_t memory_bytes = memory->allocatedBytes();
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
    runConceptDriftExperiment(all_results);

    std::vector<BenchmarkResult> memory_results, forgetting_results, drift_results;
    for (const auto& row : all_results) {
        if (row.experiment == "memory_replay" || row.experiment == "baseline") memory_results.push_back(row);
        if (row.experiment == "catastrophic_forgetting" || row.experiment == "concept_drift_a_b_a") forgetting_results.push_back(row);
        if (row.experiment == "concept_drift_a_b_a") drift_results.push_back(row);
    }
    BenchmarkCsv::write("benchmark_results.csv", all_results);
    BenchmarkCsv::write("benchmark_memory.csv", memory_results);
    BenchmarkCsv::write("benchmark_forgetting.csv", forgetting_results);
    BenchmarkCsv::write("benchmark_drift.csv", drift_results);
    BenchmarkProfileCsv::write("benchmark_profile.csv", profile_results);
    return 0;
}
