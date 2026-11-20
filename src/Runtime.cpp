#include "headers/Runtime.h"
#include "headers/AdamOptimizer.h"
#include "headers/FIFOMemory.h"
#include "headers/GloomyModelSerialization.h"
#include "headers/HybridMemory.h"
#include "headers/LearningEngine.h"
#include "headers/LossFunction.h"
#include "headers/MomentumOptimizer.h"
#include "headers/NoveltyMemory.h"
#include "headers/PrioritizedMemory.h"
#include "headers/QuantizedFIFOMemory.h"
#include "headers/QuantizedInt8FIFOMemory.h"
#include "headers/ReservoirMemory.h"
#include "headers/TrainingScheduler.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct RuntimeConfig {
    std::string runtime = "inference";
    std::string model_path = "model.gloomy";
    std::string dataset_path;
    std::string input;
    std::string activation = "tanh";
    std::string post_activation = "none";
    std::string loss = "mse";
    double huber_delta = 1.0;
    std::string optimizer = "sgd";
    double learning_rate = 0.01;
    double momentum = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;
    int hidden_layers = 2;
    int neurons = 16;
    size_t epochs = 1;
    size_t batch_size = 8;
    size_t train_every = 1;
    std::string memory_strategy = "fifo";
    std::string precision = "float64";
    size_t memory_capacity = 256;
    double novelty_threshold = 1.0;
    double recent_ratio = 0.25;
    double error_ratio = 0.25;
    double novelty_ratio = 0.25;
    double historical_ratio = 0.25;
    std::uint32_t seed = 1234;
};

std::string trim(const std::string& value) {
    const size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

double parseDouble(const std::string& text, const std::string& key) {
    size_t parsed = 0;
    const double value = std::stod(text, &parsed);
    if (parsed != text.size() || !std::isfinite(value))
        throw std::invalid_argument("Invalid numeric value for " + key);
    return value;
}

size_t parseSize(const std::string& text, const std::string& key) {
    if (text.empty() || text[0] == '-') throw std::invalid_argument("Invalid positive integer for " + key);
    size_t parsed = 0;
    const unsigned long long value = std::stoull(text, &parsed);
    if (parsed != text.size() || value == 0 || value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("Invalid positive integer for " + key);
    return static_cast<size_t>(value);
}

void applyValue(RuntimeConfig& config, const std::string& key, const std::string& value) {
    if (key == "runtime") config.runtime = value;
    else if (key == "model_path") config.model_path = value;
    else if (key == "dataset_path") config.dataset_path = value;
    else if (key == "input") config.input = value;
    else if (key == "activation") config.activation = value;
    else if (key == "post_activation") config.post_activation = value;
    else if (key == "loss") config.loss = value;
    else if (key == "huber_delta") config.huber_delta = parseDouble(value, key);
    else if (key == "optimizer") config.optimizer = value;
    else if (key == "learning_rate") config.learning_rate = parseDouble(value, key);
    else if (key == "momentum") config.momentum = parseDouble(value, key);
    else if (key == "beta1") config.beta1 = parseDouble(value, key);
    else if (key == "beta2") config.beta2 = parseDouble(value, key);
    else if (key == "epsilon") config.epsilon = parseDouble(value, key);
    else if (key == "hidden_layers") {
        size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != value.size() || parsed < 0) throw std::invalid_argument("hidden_layers must be a non-negative integer");
        config.hidden_layers = parsed;
    }
    else if (key == "neurons") {
        size_t consumed = 0;
        const int parsed = std::stoi(value, &consumed);
        if (consumed != value.size() || parsed <= 0) throw std::invalid_argument("neurons must be a positive integer");
        config.neurons = parsed;
    }
    else if (key == "epochs") config.epochs = parseSize(value, key);
    else if (key == "batch_size") config.batch_size = parseSize(value, key);
    else if (key == "train_every") config.train_every = parseSize(value, key);
    else if (key == "memory_strategy") config.memory_strategy = value;
    else if (key == "precision") config.precision = value;
    else if (key == "memory_capacity") config.memory_capacity = parseSize(value, key);
    else if (key == "novelty_threshold") config.novelty_threshold = parseDouble(value, key);
    else if (key == "recent_ratio") config.recent_ratio = parseDouble(value, key);
    else if (key == "error_ratio") config.error_ratio = parseDouble(value, key);
    else if (key == "novelty_ratio") config.novelty_ratio = parseDouble(value, key);
    else if (key == "historical_ratio") config.historical_ratio = parseDouble(value, key);
    else if (key == "seed") {
        if (value.empty() || value[0] == '-') throw std::invalid_argument("seed must be an unsigned integer");
        size_t consumed = 0;
        const unsigned long long parsed = std::stoull(value, &consumed);
        if (consumed != value.size() || parsed > std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("seed is out of range");
        config.seed = static_cast<std::uint32_t>(parsed);
    } else {
        throw std::invalid_argument("Unknown configuration key: " + key);
    }
}

RuntimeConfig readConfig(const std::string& path, int argc, char* argv[]) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Unable to open configuration file: " + path);
    RuntimeConfig config;
    std::map<std::string, bool> seen;
    std::string line;
    size_t line_number = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        const size_t separator = line.find('=');
        if (separator == std::string::npos) throw std::invalid_argument("Expected key=value at config line " + std::to_string(line_number));
        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        if (key.empty() || value.empty()) throw std::invalid_argument("Empty key or value at config line " + std::to_string(line_number));
        if (seen[key]) throw std::invalid_argument("Duplicate configuration key: " + key);
        seen[key] = true;
        applyValue(config, key, value);
    }

    for (int index = 3; index < argc; ++index) {
        const std::string option = argv[index];
        if (option == "--help" || option == "-h") {
            std::cout << "Usage: gloomy -f CONFIG [--runtime MODE] [--model PATH] [--data PATH] [--input VALUES]\n";
            std::exit(0);
        }
        if (index + 1 >= argc) throw std::invalid_argument("Missing value for option: " + option);
        const std::string value = argv[++index];
        if (option == "--runtime") applyValue(config, "runtime", value);
        else if (option == "--model") applyValue(config, "model_path", value);
        else if (option == "--data") applyValue(config, "dataset_path", value);
        else if (option == "--input") applyValue(config, "input", value);
        else if (option == "--epochs") applyValue(config, "epochs", value);
        else if (option == "--batch-size") applyValue(config, "batch_size", value);
        else throw std::invalid_argument("Unknown config-mode option: " + option);
    }
    return config;
}

void validateConfig(const RuntimeConfig& c) {
    if (c.runtime != "inference" && c.runtime != "training" && c.runtime != "online_learning")
        throw std::invalid_argument("runtime must be inference, training or online_learning");
    if (c.model_path.empty()) throw std::invalid_argument("model_path cannot be empty");
    if (c.hidden_layers > 1024 || c.neurons > 1'000'000 || c.memory_capacity > 10'000'000)
        throw std::invalid_argument("Architecture or memory capacity exceeds supported limits");
    const std::vector<std::string> activations = {"none", "sigmoid", "relu", "leaky_relu", "tanh"};
    if (std::find(activations.begin(), activations.end(), c.activation) == activations.end())
        throw std::invalid_argument("Unsupported activation: " + c.activation);
    if (c.post_activation != "none" && c.post_activation != "softmax")
        throw std::invalid_argument("Unsupported post_activation: " + c.post_activation);
    if (c.loss != "mse" && c.loss != "mae" && c.loss != "huber")
        throw std::invalid_argument("loss must be mse, mae or huber");
    if (c.optimizer != "sgd" && c.optimizer != "momentum" && c.optimizer != "adam")
        throw std::invalid_argument("optimizer must be sgd, momentum or adam");
    if (c.memory_strategy != "fifo" && c.memory_strategy != "reservoir" && c.memory_strategy != "prioritized" &&
        c.memory_strategy != "novelty" && c.memory_strategy != "hybrid")
        throw std::invalid_argument("Unsupported memory_strategy");
    if (c.precision != "float64" && c.precision != "int16" && c.precision != "int8")
        throw std::invalid_argument("precision must be float64, int16 or int8");
    if (c.precision != "float64" && c.memory_strategy != "fifo")
        throw std::invalid_argument("int16/int8 precision is currently supported with fifo memory only");
    if (c.learning_rate <= 0.0 || c.huber_delta <= 0.0 || c.momentum < 0.0 || c.momentum >= 1.0 ||
        c.beta1 < 0.0 || c.beta1 >= 1.0 || c.beta2 < 0.0 || c.beta2 >= 1.0 || c.epsilon <= 0.0)
        throw std::invalid_argument("Invalid optimizer or loss parameters");
    if (c.novelty_threshold < 0.0 || c.recent_ratio < 0.0 || c.error_ratio < 0.0 ||
        c.novelty_ratio < 0.0 || c.historical_ratio < 0.0)
        throw std::invalid_argument("Memory thresholds and ratios must be non-negative");
    if (c.runtime == "inference" && c.input.empty()) throw std::invalid_argument("input is required for inference");
    if (c.runtime != "inference" && c.dataset_path.empty()) throw std::invalid_argument("dataset_path is required for training");
}

std::vector<double> parseVector(const std::string& text, const std::string& context) {
    std::istringstream in(text);
    std::vector<double> values;
    double value = 0.0;
    while (in >> value) {
        if (!std::isfinite(value)) throw std::invalid_argument(context + " contains a non-finite number");
        values.push_back(value);
    }
    if (!in.eof()) throw std::invalid_argument(context + " contains a non-numeric value");
    if (values.empty()) throw std::invalid_argument(context + " cannot be empty");
    return values;
}

std::vector<TrainingSample> readDataset(const std::string& path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Unable to open dataset: " + path);
    std::vector<TrainingSample> samples;
    std::string line;
    size_t line_number = 0;
    size_t input_dimensions = 0, target_dimensions = 0;
    while (std::getline(file, line)) {
        ++line_number;
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t separator = line.find('|');
        if (separator == std::string::npos || line.find('|', separator + 1) != std::string::npos)
            throw std::invalid_argument("Expected 'inputs | targets' at dataset line " + std::to_string(line_number));
        TrainingSample sample;
        sample.input = parseVector(trim(line.substr(0, separator)), "Dataset inputs");
        sample.target = parseVector(trim(line.substr(separator + 1)), "Dataset targets");
        if (sample.input.size() > 1'000'000 || sample.target.size() > 1'000'000)
            throw std::invalid_argument("Dataset dimensions exceed the supported limit");
        if (!input_dimensions) { input_dimensions = sample.input.size(); target_dimensions = sample.target.size(); }
        if (sample.input.size() != input_dimensions || sample.target.size() != target_dimensions)
            throw std::invalid_argument("Dataset dimensions must be consistent (line " + std::to_string(line_number) + ")");
        samples.push_back(std::move(sample));
    }
    if (samples.empty()) throw std::invalid_argument("Dataset contains no samples");
    return samples;
}

void initializeNetwork(NeuralNetwork& network, const RuntimeConfig& c, size_t input_size, size_t target_size) {
    network.algorithm = c.activation;
    network.post_algorithm = c.post_activation;
    int previous = static_cast<int>(input_size);
    for (int layer = 0; layer < c.hidden_layers; ++layer) {
        network.addLayer(previous, c.neurons);
        previous = c.neurons;
    }
    network.addLayer(previous, static_cast<int>(target_size));
}

std::unique_ptr<Optimizer> createOptimizer(const RuntimeConfig& c) {
    if (c.optimizer == "sgd") return std::make_unique<SGDOptimizer>(c.learning_rate);
    if (c.optimizer == "momentum") return std::make_unique<MomentumOptimizer>(c.learning_rate, c.momentum);
    return std::make_unique<AdamOptimizer>(c.learning_rate, c.beta1, c.beta2, c.epsilon);
}

std::unique_ptr<LearningMemory> createMemory(
    const RuntimeConfig& c,
    const std::vector<TrainingSample>& calibration_samples
) {
    if (c.precision == "int16") {
        return std::make_unique<QuantizedFIFOMemory>(
            c.memory_capacity, TrainingSampleQuantizer::calibrate(calibration_samples));
    }
    if (c.precision == "int8") {
        return std::make_unique<QuantizedInt8FIFOMemory>(
            c.memory_capacity, Int8TrainingSampleQuantizer::calibrate(calibration_samples));
    }
    if (c.memory_strategy == "fifo") return std::make_unique<FIFOMemory>(c.memory_capacity);
    if (c.memory_strategy == "reservoir") return std::make_unique<ReservoirMemory>(c.memory_capacity, c.seed);
    if (c.memory_strategy == "prioritized") return std::make_unique<PrioritizedMemory>(c.memory_capacity, 0.6, c.seed);
    if (c.memory_strategy == "novelty") return std::make_unique<NoveltyMemory>(c.memory_capacity, c.novelty_threshold);
    HybridMemoryRatios ratios{c.recent_ratio, c.error_ratio, c.novelty_ratio, c.historical_ratio};
    return std::make_unique<HybridMemory>(c.memory_capacity, ratios, c.novelty_threshold, c.seed);
}

std::unique_ptr<LossFunction> createLoss(const RuntimeConfig& c) {
    if (c.loss == "mae") return std::make_unique<MAELoss>();
    if (c.loss == "huber") return std::make_unique<HuberLoss>(c.huber_delta);
    return std::make_unique<MSELoss>();
}

int run(const RuntimeConfig& config) {
    validateConfig(config);
    std::srand(config.seed);
    std::vector<double> inference_input;
    std::vector<TrainingSample> dataset;
    if (config.runtime == "inference") {
        inference_input = parseVector(config.input, "input");
        if (inference_input.size() > 1'000'000) throw std::invalid_argument("Input dimension exceeds the supported limit");
    }
    else dataset = readDataset(config.dataset_path);
    const size_t input_size = config.runtime == "inference" ? inference_input.size() : dataset.front().input.size();
    const size_t target_size = config.runtime == "inference" ? 1 : dataset.front().target.size();

    NeuralNetwork network;
    StreamingNormalizer normalizer(input_size);
    std::unique_ptr<Optimizer> optimizer = createOptimizer(config);
    std::unique_ptr<LearningMemory> memory;
    const bool exists = static_cast<bool>(std::ifstream(config.model_path, std::ios::binary));

    if (exists) {
        std::unique_ptr<LearningMemory> restored_memory;
        GloomyModelSerialization::load(config.model_path, network, normalizer, *optimizer,
            config.runtime == "inference" ? nullptr : &restored_memory);
        if (network.layers().front().weights().size() != input_size)
            throw std::invalid_argument("Input dimensions do not match the saved model");
        if (config.runtime != "inference" && network.layers().back().bias().size() != target_size)
            throw std::invalid_argument("Target dimensions do not match the saved model");
        memory = std::move(restored_memory);
    } else {
        if (config.runtime == "inference") throw std::runtime_error("Model file does not exist: " + config.model_path);
        initializeNetwork(network, config, input_size, target_size);
        for (const TrainingSample& sample : dataset) normalizer.update(sample.input);
    }

    if (config.runtime == "inference") {
        std::vector<double> normalized_input;
        normalizer.normalizeInto(inference_input, normalized_input);
        const std::vector<double>& prediction = network.forwardInferenceRef(normalized_input);
        for (double value : prediction) std::cout << value << '\n';
        return 0;
    }

    for (TrainingSample& sample : dataset) sample.input = normalizer.normalize(sample.input);
    if (!memory) memory = createMemory(config, dataset);
    auto loss = createLoss(config);
    LearningEngine engine(network, *loss, *optimizer);

    double final_loss = 0.0;
    if (config.runtime == "training") {
        for (const TrainingSample& sample : dataset) memory->add(sample);
        final_loss = engine.train(dataset, config.epochs, config.batch_size);
    } else {
        EveryNScheduler scheduler(config.train_every);
        double accumulated = 0.0;
        for (const TrainingSample& sample : dataset)
            accumulated += engine.learn(*memory, sample, config.batch_size, scheduler);
        final_loss = accumulated / static_cast<double>(dataset.size());
    }
    GloomyModelSerialization::save(config.model_path, network, normalizer, *optimizer, memory.get());
    std::cerr << "loss=" << final_loss << " samples=" << dataset.size() << '\n';
    return 0;
}
}

int runConfiguredRuntime(int argc, char* argv[]) {
    try {
        if (argc < 3) throw std::invalid_argument("Configuration path is required after -f/--config");
        const std::string option = argv[1];
        if (option != "-f" && option != "--config") throw std::invalid_argument("Expected -f or --config");
        RuntimeConfig config = readConfig(argv[2], argc, argv);
        return run(config);
    } catch (const std::exception& error) {
        std::cerr << "Runtime error: " << error.what() << '\n';
        return 1;
    }
}
