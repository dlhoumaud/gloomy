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
    if (sample.input.empty() || sample.target.empty()) {
        throw std::invalid_argument("Quantized samples cannot contain empty vectors");
    }
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
    std::vector<MemoryEntry> result;
    sampleIndexedInto(batch_size, result);
    return result;
}

void QuantizedFIFOMemory::sampleIndexedInto(size_t batch_size, std::vector<MemoryEntry>& output) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    const size_t count = std::min(batch_size, samples.size());
    output.resize(count);
    for (size_t index = 0; index < count; ++index) {
        TrainingSample& decoded = output[index].sample;
        sample_quantizer.decodeInto(samples[index], decoded);
        ++decoded.usage_count;
        sample_quantizer.encodeInto(decoded, samples[index]);
        output[index].index = index;
    }
}

void QuantizedFIFOMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    sample_quantizer.encodeInto(sample, samples[index]);
}

void QuantizedFIFOMemory::advanceAges() {
    TrainingSample decoded;
    for (QuantizedTrainingSample& stored : samples) {
        sample_quantizer.decodeInto(stored, decoded);
        ++decoded.age;
        sample_quantizer.encodeInto(decoded, stored);
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
