#include "headers/NormalizationSerialization.h"
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
constexpr char magic[] = "GLOOMYNM";
constexpr std::uint32_t version = 1;
constexpr std::uint64_t maximum_dimensions = 1'000'000;

void writeBytes(std::ofstream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write normalizer file");
}

void readBytes(std::ifstream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated normalizer file");
}

void writeVector(std::ofstream& stream, const std::vector<double>& values) {
    writeBytes(stream, values.data(), values.size() * sizeof(double));
}

void readVector(std::ifstream& stream, std::vector<double>& values) {
    readBytes(stream, values.data(), values.size() * sizeof(double));
}
}

void NormalizationSerialization::save(
    const std::string& path,
    const StreamingNormalizer& normalizer
) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to open normalizer file for writing");
    writeBytes(stream, magic, sizeof(magic) - 1);
    writeBytes(stream, &version, sizeof(version));
    const std::uint64_t dimensions = normalizer.dimensions();
    const std::uint64_t count = normalizer.count();
    writeBytes(stream, &dimensions, sizeof(dimensions));
    writeBytes(stream, &count, sizeof(count));
    writeVector(stream, normalizer.mean());
    const std::vector<double> variances = normalizer.variance();
    writeVector(stream, variances);
    writeVector(stream, normalizer.minimum());
    writeVector(stream, normalizer.maximum());
}

void NormalizationSerialization::load(
    const std::string& path,
    StreamingNormalizer& normalizer
) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Unable to open normalizer file for reading");
    char file_magic[sizeof(magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) {
        throw std::runtime_error("Invalid normalizer file magic");
    }
    std::uint32_t file_version = 0;
    readBytes(stream, &file_version, sizeof(file_version));
    if (file_version != version) throw std::runtime_error("Unsupported normalizer file version");

    std::uint64_t dimensions = 0;
    std::uint64_t count = 0;
    readBytes(stream, &dimensions, sizeof(dimensions));
    readBytes(stream, &count, sizeof(count));
    if (dimensions == 0 || dimensions > maximum_dimensions ||
        count == 0 || count > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Normalizer dimensions or count are invalid");
    }

    std::vector<double> means(static_cast<size_t>(dimensions));
    std::vector<double> variances(static_cast<size_t>(dimensions));
    std::vector<double> minimum(static_cast<size_t>(dimensions));
    std::vector<double> maximum(static_cast<size_t>(dimensions));
    readVector(stream, means);
    readVector(stream, variances);
    readVector(stream, minimum);
    readVector(stream, maximum);
    normalizer.restore(static_cast<size_t>(count), means, variances, minimum, maximum);
}
