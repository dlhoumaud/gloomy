#include "headers/QuantizedInt8FIFOMemory.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {
constexpr size_t metadata_bytes = 6 * sizeof(double) + 2 * sizeof(std::size_t);
}

QuantizedInt8FIFOMemory::QuantizedInt8FIFOMemory(
    size_t capacity,
    Int8TrainingSampleQuantizer quantizer
)
    : memory_capacity(capacity), sample_quantizer(std::move(quantizer)) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    samples.reserve(capacity);
}

void QuantizedInt8FIFOMemory::add(const TrainingSample& sample) {
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
    if (samples.size() == memory_capacity) {
        samples.erase(samples.begin());
    }
    samples.push_back(sample_quantizer.encode(sample));
}

void QuantizedInt8FIFOMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> QuantizedInt8FIFOMemory::sample(size_t batch_size) {
    const std::vector<MemoryEntry> entries = sampleIndexed(batch_size);
    std::vector<TrainingSample> result;
    result.reserve(entries.size());
    for (const MemoryEntry& entry : entries) {
        result.push_back(entry.sample);
    }
    return result;
}

std::vector<MemoryEntry> QuantizedInt8FIFOMemory::sampleIndexed(size_t batch_size) {
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

void QuantizedInt8FIFOMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index] = sample_quantizer.encode(sample);
}

void QuantizedInt8FIFOMemory::advanceAges() {
    for (Int8QuantizedTrainingSample& stored : samples) {
        TrainingSample decoded = sample_quantizer.decode(stored);
        ++decoded.age;
        stored = sample_quantizer.encode(decoded);
    }
}

size_t QuantizedInt8FIFOMemory::size() const { return samples.size(); }
size_t QuantizedInt8FIFOMemory::capacity() const { return memory_capacity; }
void QuantizedInt8FIFOMemory::clear() { samples.clear(); }

size_t QuantizedInt8FIFOMemory::bytesPerSample() const {
    return input_dimensions + target_dimensions + metadata_bytes;
}

size_t QuantizedInt8FIFOMemory::memoryUsedBytes() const {
    return samples.size() * bytesPerSample();
}
