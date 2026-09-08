#include "headers/GloomyConfigFile.h"
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
std::string trim(const std::string& value) {
    const std::size_t begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const std::size_t end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

int parseInt(const std::string& key, const std::string& value) {
    try {
        std::size_t consumed = 0;
        const int result = std::stoi(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid integer value for '" + key + "': " + value);
    }
}

double parseDouble(const std::string& key, const std::string& value) {
    try {
        std::size_t consumed = 0;
        const double result = std::stod(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return result;
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid decimal value for '" + key + "': " + value);
    }
}

std::size_t parseSize(const std::string& key, const std::string& value) {
    if (value.empty() || value.front() == '-') {
        throw std::invalid_argument("Invalid non-negative integer value for '" + key + "': " + value);
    }
    try {
        std::size_t consumed = 0;
        const unsigned long long result = std::stoull(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return static_cast<std::size_t>(result);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid non-negative integer value for '" + key + "': " + value);
    }
}

std::uint32_t parseUint32(const std::string& key, const std::string& value) {
    const std::size_t parsed = parseSize(key, value);
    if (parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("Value for '" + key + "' does not fit in 32 bits: " + value);
    }
    return static_cast<std::uint32_t>(parsed);
}
}

GloomyConfig GloomyConfigFile::load(const std::string& path, const GloomyConfig& base) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Unable to open configuration file: " + path);

    GloomyConfig config = base;
    std::string line;
    std::size_t line_number = 0;
    while (std::getline(stream, line)) {
        ++line_number;
        const std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed.front() == '#' || trimmed.front() == ';') continue;

        const std::size_t separator = trimmed.find('=');
        if (separator == std::string::npos) {
            throw std::invalid_argument(
                "Malformed configuration line " + std::to_string(line_number) + " in " + path
            );
        }
        const std::string key = trim(trimmed.substr(0, separator));
        const std::string value = trim(trimmed.substr(separator + 1));
        if (key.empty()) {
            throw std::invalid_argument(
                "Empty configuration key on line " + std::to_string(line_number) + " in " + path
            );
        }

        if (key == "runtime") config.runtime = value;
        else if (key == "activation") config.activation = value;
        else if (key == "post_activation") config.post_activation = value;
        else if (key == "hidden_layers") config.hidden_layers = parseInt(key, value);
        else if (key == "neurons") config.neurons = parseInt(key, value);
        else if (key == "predictions") config.predictions = parseInt(key, value);
        else if (key == "window_size") config.window_size = parseSize(key, value);
        else if (key == "loss") config.loss = value;
        else if (key == "huber_delta") config.huber_delta = parseDouble(key, value);
        else if (key == "optimizer") config.optimizer = value;
        else if (key == "learning_rate") config.learning_rate = parseDouble(key, value);
        else if (key == "momentum") config.momentum = parseDouble(key, value);
        else if (key == "beta1") config.beta1 = parseDouble(key, value);
        else if (key == "beta2") config.beta2 = parseDouble(key, value);
        else if (key == "epsilon") config.epsilon = parseDouble(key, value);
        else if (key == "memory_strategy") config.memory_strategy = value;
        else if (key == "memory_capacity") config.memory_capacity = parseSize(key, value);
        else if (key == "recent_ratio") config.recent_ratio = parseDouble(key, value);
        else if (key == "error_ratio") config.error_ratio = parseDouble(key, value);
        else if (key == "novelty_ratio") config.novelty_ratio = parseDouble(key, value);
        else if (key == "historical_ratio") config.historical_ratio = parseDouble(key, value);
        else if (key == "novelty_threshold") config.novelty_threshold = parseDouble(key, value);
        else if (key == "prioritized_alpha") config.prioritized_alpha = parseDouble(key, value);
        else if (key == "prioritized_beta") config.prioritized_beta = parseDouble(key, value);
        else if (key == "seed") config.seed = parseUint32(key, value);
        else if (key == "precision") config.precision = value;
        else if (key == "train_every") config.train_every = parseSize(key, value);
        else if (key == "batch_size") config.batch_size = parseSize(key, value);
        else if (key == "epochs") config.epochs = parseSize(key, value);
        else if (key == "model_path") config.model_path = value;
        else if (key == "optimizer_path") config.optimizer_path = value;
        else if (key == "memory_path") config.memory_path = value;
        else if (key == "metrics_path") config.metrics_path = value;
        else {
            throw std::invalid_argument(
                "Unknown configuration key '" + key + "' on line " +
                std::to_string(line_number) + " in " + path
            );
        }
    }

    return config;
}
