#include "headers/OptimizerSerialization.h"
#include "headers/AdamOptimizer.h"
#include "headers/MomentumOptimizer.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
constexpr char magic[] = "GLOOMYOP";
constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t optimizer_type_sgd = 0;
constexpr std::uint32_t optimizer_type_momentum = 1;
constexpr std::uint32_t optimizer_type_adam = 2;
constexpr std::uint64_t maximum_layers = 1024;
constexpr std::uint64_t maximum_dimension = 1'000'000;
constexpr std::uint64_t checksum_offset = 1469598103934665603ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;

void writeBytes(std::ostream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write optimizer file");
}

void readBytes(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated optimizer file");
}

std::uint64_t checksum(const std::string& bytes) {
    std::uint64_t result = checksum_offset;
    for (unsigned char byte : bytes) {
        result ^= byte;
        result *= checksum_prime;
    }
    return result;
}

void writeUint32(std::ostream& stream, std::uint32_t value) {
    writeBytes(stream, &value, sizeof(value));
}

void writeUint64(std::ostream& stream, std::uint64_t value) {
    writeBytes(stream, &value, sizeof(value));
}

void writeDouble(std::ostream& stream, double value) {
    writeBytes(stream, &value, sizeof(value));
}

std::uint32_t readUint32(std::istream& stream) {
    std::uint32_t value = 0;
    readBytes(stream, &value, sizeof(value));
    return value;
}

std::uint64_t readUint64(std::istream& stream) {
    std::uint64_t value = 0;
    readBytes(stream, &value, sizeof(value));
    return value;
}

double readDouble(std::istream& stream) {
    double value = 0.0;
    readBytes(stream, &value, sizeof(value));
    return value;
}
}

void OptimizerSerialization::save(const std::string& path, const Optimizer& optimizer) {
    std::ostringstream payload(std::ios::binary);
    writeBytes(payload, magic, sizeof(magic) - 1);
    writeBytes(payload, &format_version, sizeof(format_version));

    if (const auto* sgd = dynamic_cast<const SGDOptimizer*>(&optimizer)) {
        writeUint32(payload, optimizer_type_sgd);
        writeDouble(payload, sgd->learningRate());
    } else if (const auto* momentum = dynamic_cast<const MomentumOptimizer*>(&optimizer)) {
        writeUint32(payload, optimizer_type_momentum);
        writeDouble(payload, momentum->learning_rate);
        writeDouble(payload, momentum->momentum_factor);
        const std::uint64_t layer_count = momentum->states.size();
        writeUint64(payload, layer_count);
        for (const MomentumOptimizer::LayerState& state : momentum->states) {
            const std::uint64_t input_count = state.weight_velocity.size();
            const std::uint64_t output_count = state.bias_velocity.size();
            writeUint64(payload, input_count);
            writeUint64(payload, output_count);
            for (const auto& row : state.weight_velocity) {
                writeBytes(payload, row.data(), row.size() * sizeof(double));
            }
            writeBytes(payload, state.bias_velocity.data(), state.bias_velocity.size() * sizeof(double));
        }
    } else if (const auto* adam = dynamic_cast<const AdamOptimizer*>(&optimizer)) {
        writeUint32(payload, optimizer_type_adam);
        writeDouble(payload, adam->learning_rate);
        writeDouble(payload, adam->first_decay);
        writeDouble(payload, adam->second_decay);
        writeDouble(payload, adam->epsilon);
        writeUint64(payload, static_cast<std::uint64_t>(adam->update_count));
        const std::uint64_t layer_count = adam->states.size();
        writeUint64(payload, layer_count);
        for (const AdamOptimizer::LayerState& state : adam->states) {
            const std::uint64_t input_count = state.first_moment.size();
            const std::uint64_t output_count = state.bias_first_moment.size();
            writeUint64(payload, input_count);
            writeUint64(payload, output_count);
            for (const auto& row : state.first_moment) {
                writeBytes(payload, row.data(), row.size() * sizeof(double));
            }
            for (const auto& row : state.second_moment) {
                writeBytes(payload, row.data(), row.size() * sizeof(double));
            }
            writeBytes(payload, state.bias_first_moment.data(), state.bias_first_moment.size() * sizeof(double));
            writeBytes(payload, state.bias_second_moment.data(), state.bias_second_moment.size() * sizeof(double));
        }
    } else {
        throw std::runtime_error("Unsupported optimizer type for serialization");
    }

    const std::string bytes = payload.str();
    const std::uint64_t file_checksum = checksum(bytes);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to open optimizer file for writing");
    writeBytes(stream, bytes.data(), bytes.size());
    writeBytes(stream, &file_checksum, sizeof(file_checksum));
}

std::unique_ptr<Optimizer> OptimizerSerialization::load(
    const std::string& path,
    const std::vector<DenseLayer>& layers
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open optimizer file for reading");
    const std::string file_bytes{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    };
    if (file_bytes.size() <= sizeof(std::uint64_t)) {
        throw std::runtime_error("Optimizer file is truncated");
    }

    const size_t payload_size = file_bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, file_bytes.data() + payload_size, sizeof(stored_checksum));
    const std::string payload = file_bytes.substr(0, payload_size);
    if (checksum(payload) != stored_checksum) {
        throw std::runtime_error("Optimizer file checksum mismatch");
    }
    std::istringstream stream(payload, std::ios::binary);

    char file_magic[sizeof(magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) {
        throw std::runtime_error("Invalid optimizer file magic");
    }
    std::uint32_t version = 0;
    readBytes(stream, &version, sizeof(version));
    if (version != format_version) throw std::runtime_error("Unsupported optimizer file version");

    const std::uint32_t type = readUint32(stream);

    if (type == optimizer_type_sgd) {
        const double learning_rate = readDouble(stream);
        return std::make_unique<SGDOptimizer>(learning_rate);
    }

    if (type == optimizer_type_momentum) {
        const double learning_rate = readDouble(stream);
        const double momentum_factor = readDouble(stream);
        auto optimizer = std::make_unique<MomentumOptimizer>(learning_rate, momentum_factor);

        const std::uint64_t layer_count = readUint64(stream);
        if (layer_count > maximum_layers || layer_count != layers.size()) {
            throw std::runtime_error("Optimizer state layer count does not match the network");
        }

        std::vector<MomentumOptimizer::LayerState> states;
        states.reserve(layer_count);
        for (std::uint64_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            const std::uint64_t input_count = readUint64(stream);
            const std::uint64_t output_count = readUint64(stream);
            if (input_count > maximum_dimension || output_count > maximum_dimension) {
                throw std::runtime_error("Optimizer state shape is invalid");
            }
            if (input_count != layers[layer_index].weights().size() ||
                output_count != layers[layer_index].bias().size()) {
                throw std::runtime_error("Optimizer state shape does not match the network layers");
            }
            MomentumOptimizer::LayerState state;
            state.weight_velocity.resize(input_count, std::vector<double>(output_count, 0.0));
            for (auto& row : state.weight_velocity) {
                readBytes(stream, row.data(), row.size() * sizeof(double));
            }
            state.bias_velocity.assign(output_count, 0.0);
            readBytes(stream, state.bias_velocity.data(), state.bias_velocity.size() * sizeof(double));
            states.push_back(std::move(state));
        }
        optimizer->states = std::move(states);
        return optimizer;
    }

    if (type == optimizer_type_adam) {
        const double learning_rate = readDouble(stream);
        const double beta1 = readDouble(stream);
        const double beta2 = readDouble(stream);
        const double epsilon = readDouble(stream);
        auto optimizer = std::make_unique<AdamOptimizer>(learning_rate, beta1, beta2, epsilon);

        const std::uint64_t update_count = readUint64(stream);
        const std::uint64_t layer_count = readUint64(stream);
        if (layer_count > maximum_layers || layer_count != layers.size()) {
            throw std::runtime_error("Optimizer state layer count does not match the network");
        }

        std::vector<AdamOptimizer::LayerState> states;
        states.reserve(layer_count);
        for (std::uint64_t layer_index = 0; layer_index < layer_count; ++layer_index) {
            const std::uint64_t input_count = readUint64(stream);
            const std::uint64_t output_count = readUint64(stream);
            if (input_count > maximum_dimension || output_count > maximum_dimension) {
                throw std::runtime_error("Optimizer state shape is invalid");
            }
            if (input_count != layers[layer_index].weights().size() ||
                output_count != layers[layer_index].bias().size()) {
                throw std::runtime_error("Optimizer state shape does not match the network layers");
            }
            AdamOptimizer::LayerState state;
            state.first_moment.resize(input_count, std::vector<double>(output_count, 0.0));
            for (auto& row : state.first_moment) {
                readBytes(stream, row.data(), row.size() * sizeof(double));
            }
            state.second_moment.resize(input_count, std::vector<double>(output_count, 0.0));
            for (auto& row : state.second_moment) {
                readBytes(stream, row.data(), row.size() * sizeof(double));
            }
            state.bias_first_moment.assign(output_count, 0.0);
            readBytes(stream, state.bias_first_moment.data(), state.bias_first_moment.size() * sizeof(double));
            state.bias_second_moment.assign(output_count, 0.0);
            readBytes(stream, state.bias_second_moment.data(), state.bias_second_moment.size() * sizeof(double));
            states.push_back(std::move(state));
        }
        optimizer->states = std::move(states);
        optimizer->update_count = static_cast<std::size_t>(update_count);
        return optimizer;
    }

    throw std::runtime_error("Unknown optimizer type in optimizer file");
}
