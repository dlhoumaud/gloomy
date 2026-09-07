#include "headers/TrainingSampleSerialization.h"
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
constexpr char magic[] = "GLOOMYSM";
constexpr std::uint32_t format_version = 1;
constexpr std::uint64_t maximum_vector_size = 1'000'000;

void writeBytes(std::ofstream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("Unable to write sample file");
    }
}

void readBytes(std::ifstream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("Invalid or truncated sample file");
    }
}

void writeVector(std::ofstream& stream, const std::vector<double>& values) {
    const std::uint64_t size = values.size();
    writeBytes(stream, &size, sizeof(size));
    if (!values.empty()) {
        writeBytes(stream, values.data(), values.size() * sizeof(double));
    }
}

std::vector<double> readVector(std::ifstream& stream) {
    std::uint64_t size = 0;
    readBytes(stream, &size, sizeof(size));
    if (size > maximum_vector_size || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Sample vector size is invalid");
    }

    std::vector<double> values(static_cast<size_t>(size));
    if (!values.empty()) {
        readBytes(stream, values.data(), values.size() * sizeof(double));
    }
    return values;
}
}

void TrainingSampleSerialization::save(
    const std::string& path,
    const std::vector<TrainingSample>& samples
) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Unable to open sample file for writing");
    }

    writeBytes(stream, magic, sizeof(magic) - 1);
    writeBytes(stream, &format_version, sizeof(format_version));
    const std::uint64_t count = samples.size();
    writeBytes(stream, &count, sizeof(count));

    for (const TrainingSample& sample : samples) {
        writeVector(stream, sample.input);
        writeVector(stream, sample.target);
        writeBytes(stream, &sample.priority, sizeof(sample.priority));
        writeBytes(stream, &sample.error, sizeof(sample.error));
        writeBytes(stream, &sample.novelty, sizeof(sample.novelty));
        writeBytes(stream, &sample.rarity, sizeof(sample.rarity));
        writeBytes(stream, &sample.recency, sizeof(sample.recency));
        writeBytes(stream, &sample.diversity, sizeof(sample.diversity));
        writeBytes(stream, &sample.age, sizeof(sample.age));
        writeBytes(stream, &sample.usage_count, sizeof(sample.usage_count));
    }
}

std::vector<TrainingSample> TrainingSampleSerialization::load(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("Unable to open sample file for reading");
    }

    char file_magic[sizeof(magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) {
        throw std::runtime_error("Invalid sample file magic");
    }

    std::uint32_t version = 0;
    readBytes(stream, &version, sizeof(version));
    if (version != format_version) {
        throw std::runtime_error("Unsupported sample file version");
    }

    std::uint64_t count = 0;
    readBytes(stream, &count, sizeof(count));
    if (count > maximum_vector_size || count > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Sample count is invalid");
    }

    std::vector<TrainingSample> samples;
    samples.reserve(static_cast<size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        TrainingSample sample;
        sample.input = readVector(stream);
        sample.target = readVector(stream);
        readBytes(stream, &sample.priority, sizeof(sample.priority));
        readBytes(stream, &sample.error, sizeof(sample.error));
        readBytes(stream, &sample.novelty, sizeof(sample.novelty));
        readBytes(stream, &sample.rarity, sizeof(sample.rarity));
        readBytes(stream, &sample.recency, sizeof(sample.recency));
        readBytes(stream, &sample.diversity, sizeof(sample.diversity));
        readBytes(stream, &sample.age, sizeof(sample.age));
        readBytes(stream, &sample.usage_count, sizeof(sample.usage_count));
        samples.push_back(std::move(sample));
    }
    return samples;
}
