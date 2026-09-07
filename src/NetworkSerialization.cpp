#include "headers/NetworkSerialization.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
constexpr char magic[] = "GLOOMYNN";
constexpr std::uint32_t format_version = 1;
constexpr std::uint64_t maximum_layers = 1024;
constexpr std::uint64_t maximum_dimension = 1'000'000;
constexpr std::uint64_t checksum_offset = 1469598103934665603ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;

void writeBytes(std::ostream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write network file");
}

void readBytes(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated network file");
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
        throw std::runtime_error("Network string size is invalid");
    }
    std::string value(static_cast<size_t>(size), '\0');
    if (!value.empty()) readBytes(stream, value.data(), value.size());
    return value;
}
}

void NetworkSerialization::save(const std::string& path, const NeuralNetwork& network) {
    std::ostringstream payload(std::ios::binary);
    writeBytes(payload, magic, sizeof(magic) - 1);
    writeBytes(payload, &format_version, sizeof(format_version));
    writeString(payload, network.algorithm);
    writeString(payload, network.post_algorithm);

    const std::uint64_t layer_count = network.layers().size();
    writeBytes(payload, &layer_count, sizeof(layer_count));
    for (const DenseLayer& layer : network.layers()) {
        const std::uint64_t input_count = layer.weights().size();
        const std::uint64_t output_count = layer.bias().size();
        writeBytes(payload, &input_count, sizeof(input_count));
        writeBytes(payload, &output_count, sizeof(output_count));
        for (const auto& row : layer.weights()) {
            writeBytes(payload, row.data(), row.size() * sizeof(double));
        }
        writeBytes(payload, layer.bias().data(), layer.bias().size() * sizeof(double));
    }

    const std::string bytes = payload.str();
    const std::uint64_t file_checksum = checksum(bytes);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to open network file for writing");
    writeBytes(stream, bytes.data(), bytes.size());
    writeBytes(stream, &file_checksum, sizeof(file_checksum));
}

void NetworkSerialization::load(const std::string& path, NeuralNetwork& network) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open network file for reading");
    const std::string file_bytes{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    };
    if (file_bytes.size() <= sizeof(std::uint64_t)) {
        throw std::runtime_error("Network file is truncated");
    }

    const size_t payload_size = file_bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, file_bytes.data() + payload_size, sizeof(stored_checksum));
    const std::string payload = file_bytes.substr(0, payload_size);
    if (checksum(payload) != stored_checksum) {
        throw std::runtime_error("Network file checksum mismatch");
    }
    std::istringstream stream(payload, std::ios::binary);

    char file_magic[sizeof(magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) {
        throw std::runtime_error("Invalid network file magic");
    }
    std::uint32_t version = 0;
    readBytes(stream, &version, sizeof(version));
    if (version != format_version) throw std::runtime_error("Unsupported network file version");

    const std::string algorithm = readString(stream);
    const std::string post_algorithm = readString(stream);
    std::uint64_t layer_count = 0;
    readBytes(stream, &layer_count, sizeof(layer_count));
    if (layer_count == 0 || layer_count > maximum_layers) {
        throw std::runtime_error("Network layer count is invalid");
    }

    network.clear();
    network.algorithm = algorithm;
    network.post_algorithm = post_algorithm;
    for (std::uint64_t layer_index = 0; layer_index < layer_count; ++layer_index) {
        std::uint64_t input_count = 0;
        std::uint64_t output_count = 0;
        readBytes(stream, &input_count, sizeof(input_count));
        readBytes(stream, &output_count, sizeof(output_count));
        if (input_count == 0 || output_count == 0 ||
            input_count > maximum_dimension || output_count > maximum_dimension) {
            throw std::runtime_error("Network layer dimensions are invalid");
        }
        network.addLayer(static_cast<int>(input_count), static_cast<int>(output_count));
        DenseLayer& layer = network.layers().back();
        for (auto& row : layer.weights()) {
            readBytes(stream, row.data(), row.size() * sizeof(double));
        }
        readBytes(stream, layer.bias().data(), layer.bias().size() * sizeof(double));
    }
}
