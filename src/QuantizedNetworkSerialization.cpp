#include "headers/QuantizedNetworkSerialization.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
constexpr char magic[] = "GLOOMYQN";
constexpr std::uint32_t format_version = 1;
constexpr std::uint64_t maximum_layers = 1024;
constexpr std::uint64_t maximum_dimension = 1'000'000;
constexpr std::uint64_t checksum_offset = 1469598103934665603ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;

void writeBytes(std::ostream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write quantized network file");
}

void readBytes(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated quantized network file");
}

std::uint64_t checksum(const std::string& bytes) {
    std::uint64_t result = checksum_offset;
    for (unsigned char byte : bytes) {
        result ^= byte;
        result *= checksum_prime;
    }
    return result;
}

void writeString(std::ostream& stream, const std::string& value) {
    const std::uint64_t size = value.size();
    writeBytes(stream, &size, sizeof(size));
    if (!value.empty()) writeBytes(stream, value.data(), value.size());
}

std::string readString(std::istream& stream) {
    std::uint64_t size = 0;
    readBytes(stream, &size, sizeof(size));
    if (size > maximum_dimension || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Quantized network string size is invalid");
    }
    std::string value(static_cast<size_t>(size), '\0');
    if (!value.empty()) readBytes(stream, value.data(), value.size());
    return value;
}

void writeQuantizedVector(std::ostream& stream, const Int8QuantizedVector& vector) {
    writeBytes(stream, &vector.parameters.scale, sizeof(vector.parameters.scale));
    writeBytes(stream, &vector.parameters.zero_point, sizeof(vector.parameters.zero_point));
    const std::uint64_t count = vector.values.size();
    writeBytes(stream, &count, sizeof(count));
    if (!vector.values.empty()) {
        writeBytes(stream, vector.values.data(), vector.values.size() * sizeof(std::int8_t));
    }
}

Int8QuantizedVector readQuantizedVector(std::istream& stream, std::uint64_t expected_count) {
    Int8QuantizedVector result;
    readBytes(stream, &result.parameters.scale, sizeof(result.parameters.scale));
    readBytes(stream, &result.parameters.zero_point, sizeof(result.parameters.zero_point));
    std::uint64_t count = 0;
    readBytes(stream, &count, sizeof(count));
    if (count != expected_count) {
        throw std::runtime_error("Quantized network vector size does not match its declared dimensions");
    }
    if (count > maximum_dimension) {
        throw std::runtime_error("Quantized network vector size is invalid");
    }
    result.values.resize(static_cast<size_t>(count));
    if (!result.values.empty()) {
        readBytes(stream, result.values.data(), result.values.size() * sizeof(std::int8_t));
    }
    return result;
}
}

void QuantizedNetworkSerialization::save(const std::string& path, const QuantizedNetwork& network) {
    std::ostringstream payload(std::ios::binary);
    writeBytes(payload, magic, sizeof(magic) - 1);
    writeBytes(payload, &format_version, sizeof(format_version));
    writeString(payload, network.algorithm);
    writeString(payload, network.post_algorithm);

    const std::uint64_t layer_count = network.layers.size();
    writeBytes(payload, &layer_count, sizeof(layer_count));
    for (const QuantizedDenseLayer& layer : network.layers) {
        const std::uint64_t input_size = layer.input_size;
        const std::uint64_t output_size = layer.output_size;
        writeBytes(payload, &input_size, sizeof(input_size));
        writeBytes(payload, &output_size, sizeof(output_size));
        writeQuantizedVector(payload, layer.weights);
        writeQuantizedVector(payload, layer.bias);
    }

    const std::string bytes = payload.str();
    const std::uint64_t file_checksum = checksum(bytes);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to open quantized network file for writing");
    writeBytes(stream, bytes.data(), bytes.size());
    writeBytes(stream, &file_checksum, sizeof(file_checksum));
}

QuantizedNetwork QuantizedNetworkSerialization::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open quantized network file for reading");
    const std::string file_bytes{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    };
    if (file_bytes.size() <= sizeof(std::uint64_t)) {
        throw std::runtime_error("Quantized network file is truncated");
    }

    const size_t payload_size = file_bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, file_bytes.data() + payload_size, sizeof(stored_checksum));
    const std::string payload = file_bytes.substr(0, payload_size);
    if (checksum(payload) != stored_checksum) {
        throw std::runtime_error("Quantized network file checksum mismatch");
    }
    std::istringstream stream(payload, std::ios::binary);

    char file_magic[sizeof(magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) {
        throw std::runtime_error("Invalid quantized network file magic");
    }
    std::uint32_t version = 0;
    readBytes(stream, &version, sizeof(version));
    if (version != format_version) throw std::runtime_error("Unsupported quantized network file version");

    QuantizedNetwork network;
    network.algorithm = readString(stream);
    network.post_algorithm = readString(stream);

    std::uint64_t layer_count = 0;
    readBytes(stream, &layer_count, sizeof(layer_count));
    if (layer_count == 0 || layer_count > maximum_layers) {
        throw std::runtime_error("Quantized network layer count is invalid");
    }

    network.layers.reserve(static_cast<size_t>(layer_count));
    for (std::uint64_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        std::uint64_t input_size = 0;
        std::uint64_t output_size = 0;
        readBytes(stream, &input_size, sizeof(input_size));
        readBytes(stream, &output_size, sizeof(output_size));
        if (input_size == 0 || output_size == 0 ||
            input_size > maximum_dimension || output_size > maximum_dimension) {
            throw std::runtime_error("Quantized network layer dimensions are invalid");
        }

        QuantizedDenseLayer layer;
        layer.input_size = static_cast<size_t>(input_size);
        layer.output_size = static_cast<size_t>(output_size);
        layer.weights = readQuantizedVector(stream, input_size * output_size);
        layer.bias = readQuantizedVector(stream, output_size);
        network.layers.push_back(std::move(layer));
    }

    return network;
}
