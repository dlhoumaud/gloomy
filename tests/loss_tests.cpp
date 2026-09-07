#include "LossFunction.h"
#include "DenseLayer.h"
#include "LearningEngine.h"
#include "Optimizer.h"
#include "FIFOMemory.h"
#include "ReservoirMemory.h"
#include "PrioritizedMemory.h"
#include "NoveltyMemory.h"
#include "HybridMemory.h"
#include "ImportanceScorer.h"
#include "TrainingScheduler.h"
#include "Normalization.h"
#include "Quantization.h"
#include "TrainingSampleQuantization.h"
#include "QuantizedFIFOMemory.h"
#include "TrainingSampleSerialization.h"
#include "Metrics.h"
#include "Benchmark.h"
#include "Int8Quantization.h"
#include "Int8TrainingSampleQuantization.h"
#include "QuantizedInt8FIFOMemory.h"
#include "NetworkSerialization.h"
#include "NormalizationSerialization.h"
#include "MomentumOptimizer.h"
#include "AdamOptimizer.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {
void assertClose(double actual, double expected) {
    assert(std::abs(actual - expected) < 1e-12);
}

void testMSECompute() {
    MSELoss loss;
    assertClose(loss.compute({1.0, 3.0}, {0.0, 1.0}), 2.5);
}

void testMSEGradient() {
    MSELoss loss;
    const std::vector<double> gradient = loss.gradient({1.0, 3.0}, {0.0, 1.0});
    assert(gradient.size() == 2);
    assertClose(gradient[0], 1.0);
    assertClose(gradient[1], 2.0);
}

void testAdditionalLosses() {
    MAELoss mae;
    assertClose(mae.compute({1.0, 5.0}, {0.0, 1.0}), 2.5);
    const std::vector<double> mae_gradient = mae.gradient({1.0, 5.0}, {0.0, 1.0});
    assertClose(mae_gradient[0], 0.5);
    assertClose(mae_gradient[1], 0.5);

    HuberLoss huber(1.0);
    assertClose(huber.compute({0.5, 3.0}, {0.0, 0.0}), 1.3125);
    const std::vector<double> huber_gradient = huber.gradient({0.5, 3.0}, {0.0, 0.0});
    assertClose(huber_gradient[0], 0.25);
    assertClose(huber_gradient[1], 0.5);
}

void testInvalidSizes() {
    MSELoss loss;
    bool threw = false;
    try {
        loss.compute({1.0}, {1.0, 2.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testEmptyInputs() {
    MSELoss loss;
    bool threw = false;
    try {
        loss.gradient({}, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testDenseLayerGradient() {
    DenseLayer layer(2, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.3;
    layer.weights()[1][0] = -0.4;
    layer.bias()[0] = 0.1;

    const std::vector<double> input = {0.7, -1.2};
    const std::vector<double> target = {0.2};
    MSELoss loss;
    const double epsilon = 1e-6;

    const auto evaluate = [&]() {
        return loss.compute(layer.forward(input), target);
    };

    layer.forward(input);
    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));

    for (size_t input_index = 0; input_index < input.size(); ++input_index) {
        const double original = layer.weights()[input_index][0];
        layer.weights()[input_index][0] = original + epsilon;
        const double loss_plus = evaluate();
        layer.weights()[input_index][0] = original - epsilon;
        const double loss_minus = evaluate();
        layer.weights()[input_index][0] = original;

        const double numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        assert(std::abs(numerical - layer.weightGradients()[input_index][0]) < 1e-7);
    }

    const double original_bias = layer.bias()[0];
    layer.bias()[0] = original_bias + epsilon;
    const double loss_plus = evaluate();
    layer.bias()[0] = original_bias - epsilon;
    const double loss_minus = evaluate();
    layer.bias()[0] = original_bias;

    const double numerical_bias = (loss_plus - loss_minus) / (2.0 * epsilon);
    assert(std::abs(numerical_bias - layer.biasGradients()[0]) < 1e-7);
}

void testSGDUpdateReducesLoss() {
    DenseLayer layer(1, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.0;
    layer.bias()[0] = 0.0;

    const std::vector<double> input = {1.0};
    const std::vector<double> target = {2.0};
    MSELoss loss;
    const double lossBefore = loss.compute(layer.forward(input), target);

    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));
    std::vector<DenseLayer> layers;
    layers.push_back(layer);

    SGDOptimizer optimizer(0.1);
    optimizer.update(layers);

    const double lossAfter = loss.compute(layers[0].forward(input), target);
    assert(lossAfter < lossBefore);
}

void testLearningEngineBatchTraining() {
    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 1);
    network.layers()[0].weights()[0][0] = 0.0;
    network.layers()[0].bias()[0] = 0.0;

    MSELoss loss;
    SGDOptimizer optimizer(0.05);
    LearningEngine engine(network, loss, optimizer);
    const std::vector<TrainingSample> samples = {
        {{1.0}, {2.0}},
        {{2.0}, {4.0}}
    };

    const double initial_loss = loss.compute(network.forward({1.0}), {2.0});
    const double final_loss = engine.train(samples, 40, 2);
    assert(final_loss < initial_loss);
    assert(loss.compute(network.forward({1.0}), {2.0}) < initial_loss);
}

void testFIFOMemoryCapacity() {
    FIFOMemory memory(2);
    memory.add({{1.0}, {2.0}});
    memory.add({{2.0}, {4.0}});
    memory.add({{3.0}, {6.0}});

    assert(memory.capacity() == 2);
    assert(memory.size() == 2);
    const std::vector<TrainingSample> samples = memory.sample(2);
    assert(samples.size() == 2);
    assert(samples[0].input[0] == 2.0);
    assert(samples[1].input[0] == 3.0);

    memory.remove(0);
    assert(memory.size() == 1);
    memory.clear();
    assert(memory.size() == 0);
}

void testMemoryMetadata() {
    FIFOMemory memory(2);
    memory.add({{1.0}, {2.0}});
    memory.add({{2.0}, {4.0}});
    memory.advanceAges();
    const std::vector<TrainingSample> first_batch = memory.sample(1);
    assert(first_batch[0].age == 1);
    assert(first_batch[0].usage_count == 1);

    memory.advanceAges();
    const std::vector<TrainingSample> second_batch = memory.sample(1);
    assert(second_batch[0].age == 2);
    assert(second_batch[0].usage_count == 2);
}

void testLearningEngineMemoryReplay() {
    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 1);
    network.layers()[0].weights()[0][0] = 0.0;
    network.layers()[0].bias()[0] = 0.0;

    MSELoss loss;
    SGDOptimizer optimizer(0.05);
    LearningEngine engine(network, loss, optimizer);
    FIFOMemory memory(2);
    memory.add({{1.0}, {2.0}});
    memory.add({{2.0}, {4.0}});

    const double replay_loss = engine.trainFromMemory(memory, 2);
    assert(replay_loss > 0.0);

    engine.learn(memory, {{3.0}, {6.0}}, 2);
    const std::vector<TrainingSample> samples = memory.sample(2);
    assert(samples.size() == 2);
    assert(samples[0].input[0] == 2.0);
    assert(samples[1].input[0] == 3.0);
}

void testReservoirMemory() {
    ReservoirMemory first(3, 1234u);
    ReservoirMemory second(3, 1234u);
    for (double value = 1.0; value <= 100.0; ++value) {
        const TrainingSample sample = {{value}, {value * 2.0}};
        first.add(sample);
        second.add(sample);
    }

    assert(first.capacity() == 3);
    assert(first.size() == 3);
    const std::vector<TrainingSample> first_samples = first.sample(3);
    const std::vector<TrainingSample> second_samples = second.sample(3);
    for (size_t index = 0; index < first_samples.size(); ++index) {
        assert(first_samples[index].input[0] == second_samples[index].input[0]);
        assert(first_samples[index].input[0] >= 1.0);
        assert(first_samples[index].input[0] <= 100.0);
    }

    first.clear();
    assert(first.size() == 0);
}

void testPrioritizedMemory() {
    PrioritizedMemory memory(2, 0.6, 1234u);
    memory.add({{1.0}, {1.0}, 0.1});
    memory.add({{2.0}, {2.0}, 0.2});
    memory.add({{3.0}, {3.0}, 10.0});

    assert(memory.capacity() == 2);
    assert(memory.size() == 2);
    const std::vector<TrainingSample> samples = memory.sample(2);
    bool found_high_priority = false;
    for (const TrainingSample& sample : samples) {
        if (sample.input[0] == 3.0) {
            found_high_priority = true;
        }
    }
    assert(found_high_priority);

    const std::vector<MemoryEntry> entries = memory.sampleIndexed(1);
    assert(entries.size() == 1);
    TrainingSample updated = entries[0].sample;
    updated.priority = 42.0;
    memory.update(entries[0].index, updated);
    const std::vector<MemoryEntry> updated_entries = memory.sampleIndexed(2);
    bool found_updated_priority = false;
    for (const MemoryEntry& entry : updated_entries) {
        if (entry.sample.priority == 42.0) {
            found_updated_priority = true;
        }
    }
    assert(found_updated_priority);
}

void testNoveltyMemory() {
    NoveltyMemory memory(2, 0.25);
    memory.add({{0.0, 0.0}, {0.0}});
    memory.add({{0.1, 0.1}, {0.0}});
    assert(memory.size() == 1);

    memory.add({{1.0, 0.0}, {1.0}});
    assert(memory.size() == 2);
    memory.add({{2.0, 0.0}, {2.0}});
    assert(memory.size() == 2);

    const std::vector<TrainingSample> samples = memory.sample(2);
    assert(samples[0].input[0] == 2.0 || samples[1].input[0] == 2.0);

    bool threw = false;
    try {
        memory.add({{1.0}, {1.0}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testHybridMemory() {
    HybridMemoryRatios ratios;
    HybridMemory memory(8, ratios, 1.0, 1234u);
    for (double value = 0.0; value < 20.0; value += 1.0) {
        memory.add({{value}, {value}, 0.0});
    }

    assert(memory.capacity() == 8);
    assert(memory.size() <= memory.capacity());
    const std::vector<size_t> sizes = memory.partitionSizes();
    size_t partition_total = 0;
    for (size_t size : sizes) {
        partition_total += size;
    }
    assert(partition_total == memory.size());
    assert(sizes.size() == 4);

    const std::vector<TrainingSample> batch = memory.sample(3);
    assert(batch.size() == 3);
}

void testImportanceScorer() {
    ImportanceScorer error_scorer;
    assert(std::abs(error_scorer.score({0.5}) - 0.5) < 1e-12);

    ImportanceWeights weights;
    weights.error = 0.0;
    weights.novelty = 1.0;
    ImportanceScorer novelty_scorer(weights);
    assert(std::abs(novelty_scorer.score({0.0, 0.8}) - 0.8) < 1e-12);

    bool threw = false;
    try {
        ImportanceWeights invalid;
        invalid.error = -1.0;
        ImportanceScorer invalid_scorer(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testTrainingSchedulers() {
    EverySampleScheduler every_sample;
    assert(every_sample.shouldTrain(0.0));

    EveryNScheduler every_two(2);
    assert(!every_two.shouldTrain(1.0));
    assert(every_two.shouldTrain(1.0));
    assert(!every_two.shouldTrain(1.0));

    OnHighErrorScheduler high_error(0.5);
    assert(!high_error.shouldTrain(0.49));
    assert(high_error.shouldTrain(0.5));
}

void testStreamingNormalization() {
    StreamingNormalizer normalizer(2);
    normalizer.update({1.0, 10.0});
    normalizer.update({3.0, 14.0});
    normalizer.update({5.0, 18.0});

    assert(normalizer.count() == 3);
    assert(std::abs(normalizer.mean()[0] - 3.0) < 1e-12);
    assert(std::abs(normalizer.mean()[1] - 14.0) < 1e-12);
    assert(std::abs(normalizer.variance()[0] - (8.0 / 3.0)) < 1e-12);
    assert(normalizer.minimum()[0] == 1.0);
    assert(normalizer.maximum()[1] == 18.0);

    const std::vector<double> normalized = normalizer.normalize({3.0, 14.0});
    assert(std::abs(normalized[0]) < 1e-12);
    assert(std::abs(normalized[1]) < 1e-12);

    bool threw = false;
    try {
        normalizer.update({1.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testNormalizationSerialization() {
    const std::string path = "/tmp/gloomy_normalizer.bin";
    StreamingNormalizer original(2);
    original.update({1.0, 10.0});
    original.update({3.0, 14.0});
    original.update({5.0, 18.0});
    const std::vector<double> expected = original.normalize({4.0, 16.0});
    NormalizationSerialization::save(path, original);

    StreamingNormalizer restored(2);
    NormalizationSerialization::load(path, restored);
    const std::vector<double> actual = restored.normalize({4.0, 16.0});
    assert(restored.count() == original.count());
    assert(std::abs(actual[0] - expected[0]) < 1e-12);
    assert(std::abs(actual[1] - expected[1]) < 1e-12);
    std::remove(path.c_str());
}

void testInt16Quantization() {
    const std::vector<double> values = {-10.0, -1.5, 0.0, 2.5, 10.0};
    const QuantizationParameters parameters = Int16Quantizer::calibrate(values);
    const QuantizedVector quantized = Int16Quantizer::quantize(values, parameters);
    const std::vector<double> restored = Int16Quantizer::dequantize(quantized);

    assert(quantized.values.size() == values.size());
    assert(parameters.scale > 0.0);
    for (size_t index = 0; index < values.size(); ++index) {
        assert(std::abs(restored[index] - values[index]) <= parameters.scale);
    }

    const QuantizationParameters constant_parameters =
        Int16Quantizer::calibrate({3.0, 3.0});
    const QuantizedVector constant = Int16Quantizer::quantize({3.0}, constant_parameters);
    assert(std::abs(Int16Quantizer::dequantize(constant)[0] - 3.0) <= constant_parameters.scale);

    bool threw = false;
    try {
        Int16Quantizer::calibrate({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testInt8Quantization() {
    const std::vector<double> values = {-10.0, -1.5, 0.0, 2.5, 10.0};
    const QuantizationParameters parameters = Int8Quantizer::calibrate(values);
    const Int8QuantizedVector quantized = Int8Quantizer::quantize(values, parameters);
    const std::vector<double> restored = Int8Quantizer::dequantize(quantized);

    assert(quantized.values.size() == values.size());
    assert(parameters.scale > 0.0);
    for (size_t index = 0; index < values.size(); ++index) {
        assert(std::abs(restored[index] - values[index]) <= parameters.scale);
    }
}

void testQuantizedInt8FIFOMemory() {
    const std::vector<TrainingSample> calibration_samples = {
        {{{0.0, 1.0}}, {{0.0}}},
        {{{10.0, 11.0}}, {{10.0}}}
    };
    QuantizedInt8FIFOMemory memory(
        2,
        Int8TrainingSampleQuantizer::calibrate(calibration_samples)
    );
    memory.add(calibration_samples[0]);
    memory.add(calibration_samples[1]);
    QuantizedFIFOMemory int16_memory(
        2,
        TrainingSampleQuantizer::calibrate(calibration_samples)
    );
    int16_memory.add(calibration_samples[0]);
    int16_memory.add(calibration_samples[1]);
    assert(memory.size() == 2);
    assert(memory.memoryUsedBytes() == 2 * memory.bytesPerSample());
    assert(memory.bytesPerSample() < int16_memory.bytesPerSample());
    assert(!memory.sample(1).empty());
}

void testTrainingSampleQuantization() {
    const std::vector<TrainingSample> samples = {
        {{{-1.0, 0.0}}, {{0.0}}, 0.7},
        {{{1.0, 2.0}}, {{1.0}}, 0.3}
    };
    const TrainingSampleQuantizer quantizer = TrainingSampleQuantizer::calibrate(samples);
    const QuantizedTrainingSample encoded = quantizer.encode(samples[0]);
    const TrainingSample decoded = quantizer.decode(encoded);

    assert(decoded.input.size() == samples[0].input.size());
    assert(decoded.target.size() == samples[0].target.size());
    assert(std::abs(decoded.input[0] - samples[0].input[0]) <= quantizer.inputParameters().scale);
    assert(std::abs(decoded.target[0] - samples[0].target[0]) <= quantizer.targetParameters().scale);
    assert(decoded.priority == samples[0].priority);
}

void testQuantizedFIFOMemory() {
    const std::vector<TrainingSample> calibration_samples = {
        {{{0.0, 1.0}}, {{0.0}}},
        {{{10.0, 11.0}}, {{10.0}}}
    };
    QuantizedFIFOMemory memory(
        2,
        TrainingSampleQuantizer::calibrate(calibration_samples)
    );
    memory.add(calibration_samples[0]);
    memory.add(calibration_samples[1]);

    assert(memory.size() == 2);
    assert(memory.bytesPerSample() > 0);
    assert(memory.memoryUsedBytes() == 2 * memory.bytesPerSample());
    const std::vector<TrainingSample> restored = memory.sample(1);
    assert(std::abs(restored[0].input[0] - 0.0) <= 1.0);
}

void testTrainingSampleSerialization() {
    const std::string path = "/tmp/gloomy_training_samples.bin";
    const std::vector<TrainingSample> samples = {
        {{{1.0, 2.0}}, {{3.0}}, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 4, 2}
    };
    TrainingSampleSerialization::save(path, samples);
    const std::vector<TrainingSample> restored = TrainingSampleSerialization::load(path);
    assert(restored.size() == 1);
    assert(restored[0].input == samples[0].input);
    assert(restored[0].target == samples[0].target);
    assert(restored[0].priority == samples[0].priority);
    assert(restored[0].age == samples[0].age);
    assert(restored[0].usage_count == samples[0].usage_count);
    std::remove(path.c_str());
}

void testNetworkSerialization() {
    const std::string path = "/tmp/gloomy_network.bin";
    NeuralNetwork original;
    original.algorithm = "tanh";
    original.post_algorithm = "none";
    original.addLayer(2, 2);
    original.addLayer(2, 1);
    original.layers()[0].weights()[0][0] = 0.2;
    original.layers()[0].weights()[0][1] = -0.3;
    original.layers()[0].weights()[1][0] = 0.4;
    original.layers()[0].weights()[1][1] = 0.5;
    original.layers()[0].bias()[0] = 0.1;
    original.layers()[0].bias()[1] = -0.2;
    original.layers()[1].weights()[0][0] = 0.6;
    original.layers()[1].weights()[1][0] = -0.7;
    original.layers()[1].bias()[0] = 0.3;

    const std::vector<double> input = {0.5, -0.25};
    const std::vector<double> expected = original.forward(input);
    NetworkSerialization::save(path, original);

    NeuralNetwork restored;
    NetworkSerialization::load(path, restored);
    const std::vector<double> actual = restored.forward(input);
    assert(actual.size() == expected.size());
    assert(std::abs(actual[0] - expected[0]) < 1e-12);
    assert(restored.algorithm == original.algorithm);
    assert(restored.layers().size() == original.layers().size());

    std::fstream corrupt(path, std::ios::in | std::ios::out | std::ios::binary);
    corrupt.seekp(16);
    char byte = 0;
    corrupt.read(&byte, sizeof(byte));
    corrupt.seekp(16);
    byte ^= 1;
    corrupt.write(&byte, sizeof(byte));
    corrupt.close();
    bool checksum_failed = false;
    try {
        NetworkSerialization::load(path, restored);
    } catch (const std::runtime_error&) {
        checksum_failed = true;
    }
    assert(checksum_failed);
    std::remove(path.c_str());
}

void testMetrics() {
    const RegressionMetrics metrics = Metrics::regression({1.0, 3.0}, {0.0, 1.0});
    assert(std::abs(metrics.mae - 1.5) < 1e-12);
    assert(std::abs(metrics.rmse - std::sqrt(2.5)) < 1e-12);
    assert(std::abs(Metrics::forgetting(0.9, 0.7) - 0.2) < 1e-12);

    bool threw = false;
    try {
        Metrics::regression({1.0}, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testBenchmarkCsv() {
    const std::string path = "/tmp/gloomy_benchmark.csv";
    BenchmarkResult result;
    result.memory_strategy = "hybrid,8kb";
    result.precision = "int16";
    result.optimizer = "sgd";
    result.mae = 0.25;
    result.memory_used_bytes = 8192;
    result.samples_stored = 32;
    BenchmarkCsv::write(path, {result});

    std::ifstream stream(path);
    std::string content(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
    );
    assert(content.find("memory_strategy,precision,optimizer") == 0);
    assert(content.find("\"hybrid,8kb\",int16,sgd") != std::string::npos);
    std::remove(path.c_str());
}

void testMomentumOptimizer() {
    DenseLayer layer(1, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.0;
    layer.bias()[0] = 0.0;
    MSELoss loss;
    const std::vector<double> input = {1.0};
    const std::vector<double> target = {2.0};

    layer.forward(input);
    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));
    std::vector<DenseLayer> layers;
    layers.push_back(layer);
    MomentumOptimizer optimizer(0.1, 0.9);
    optimizer.update(layers);
    const double first_weight = layers[0].weights()[0][0];

    layers[0].forward(input);
    layers[0].zeroGradients();
    layers[0].backward(loss.gradient(layers[0].forward(input), target));
    optimizer.update(layers);
    assert(std::abs(layers[0].weights()[0][0]) > std::abs(first_weight));
    assert(optimizer.stateBytes() == sizeof(double) * 2);
}

void testAdamOptimizer() {
    DenseLayer layer(1, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.0;
    layer.bias()[0] = 0.0;
    MSELoss loss;
    const std::vector<double> input = {1.0};
    const std::vector<double> target = {2.0};

    layer.forward(input);
    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));
    std::vector<DenseLayer> layers;
    layers.push_back(layer);
    AdamOptimizer optimizer(0.05);
    optimizer.update(layers);
    assert(layers[0].weights()[0][0] > 0.0);
    assert(optimizer.stateBytes() == sizeof(double) * 4);
}

}

int main() {
    testMSECompute();
    testMSEGradient();
    testAdditionalLosses();
    testInvalidSizes();
    testEmptyInputs();
    testDenseLayerGradient();
    testSGDUpdateReducesLoss();
    testLearningEngineBatchTraining();
    testFIFOMemoryCapacity();
    testMemoryMetadata();
    testLearningEngineMemoryReplay();
    testReservoirMemory();
    testPrioritizedMemory();
    testNoveltyMemory();
    testHybridMemory();
    testImportanceScorer();
    testTrainingSchedulers();
    testStreamingNormalization();
    testNormalizationSerialization();
    testInt16Quantization();
    testInt8Quantization();
    testQuantizedInt8FIFOMemory();
    testTrainingSampleQuantization();
    testQuantizedFIFOMemory();
    testTrainingSampleSerialization();
    testNetworkSerialization();
    testMetrics();
    testBenchmarkCsv();
    testMomentumOptimizer();
    testAdamOptimizer();
    return 0;
}
