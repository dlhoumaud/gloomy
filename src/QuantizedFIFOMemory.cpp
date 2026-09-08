#include "headers/QuantizedFIFOMemory.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {
constexpr size_t metadata_bytes = 6 * sizeof(double) + 2 * sizeof(std::size_t);
}

QuantizedFIFOMemory::QuantizedFIFOMemory(
    size_t capacity,
    TrainingSampleQuantizer quantizer
)
    : memory_capacity(capacity),
      sample_quantizer(std::move(quantizer)),
      input_dimensions(0),
      target_dimensions(0) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    samples.reserve(capacity);
}

void QuantizedFIFOMemory::add(const TrainingSample& sample) {
    validateTrainingSampleVectors(sample);
    if (input_dimensions == 0) {
        input_dimensions = sample.input.size();
        target_dimensions = sample.target.size();
    }
    if (sample.input.size() != input_dimensions || sample.target.size() != target_dimensions) {
        throw std::invalid_argument("Quantized sample dimensions must remain constant");
    }

    const QuantizedTrainingSample encoded = sample_quantizer.encode(sample);
    if (samples.size() == memory_capacity) {
        samples.erase(samples.begin());
    }
    samples.push_back(encoded);
}

void QuantizedFIFOMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> QuantizedFIFOMemory::sample(size_t batch_size) {
    const std::vector<MemoryEntry> entries = sampleIndexed(batch_size);
    std::vector<TrainingSample> result;
    result.reserve(entries.size());
    for (const MemoryEntry& entry : entries) {
        result.push_back(entry.sample);
    }
    return result;
}

std::vector<MemoryEntry> QuantizedFIFOMemory::sampleIndexed(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    const size_t count = std::min(batch_size, samples.size());
    std::vector<MemoryEntry> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        TrainingSample decoded = sample_quantizer.decode(samples[index]);
        ++decoded.usage_count;
        samples[index] = sample_quantizer.encode(decoded);
        result.push_back({index, decoded});
    }
    return result;
}

void QuantizedFIFOMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index] = sample_quantizer.encode(sample);
}

void QuantizedFIFOMemory::advanceAges() {
    for (QuantizedTrainingSample& stored : samples) {
        TrainingSample decoded = sample_quantizer.decode(stored);
        ++decoded.age;
        stored = sample_quantizer.encode(decoded);
    }
}

size_t QuantizedFIFOMemory::size() const {
    return samples.size();
}

size_t QuantizedFIFOMemory::capacity() const {
    return memory_capacity;
}

void QuantizedFIFOMemory::clear() {
    samples.clear();
}

size_t QuantizedFIFOMemory::bytesPerSample() const {
    return input_dimensions * sizeof(std::int16_t) +
        target_dimensions * sizeof(std::int16_t) + metadata_bytes;
}

size_t QuantizedFIFOMemory::memoryUsedBytes() const {
    return samples.size() * bytesPerSample();
}
