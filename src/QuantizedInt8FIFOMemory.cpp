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
    std::vector<MemoryEntry> result;
    sampleIndexedInto(batch_size, result);
    return result;
}

void QuantizedInt8FIFOMemory::sampleIndexedInto(size_t batch_size, std::vector<MemoryEntry>& output) {
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

void QuantizedInt8FIFOMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    sample_quantizer.encodeInto(sample, samples[index]);
}

void QuantizedInt8FIFOMemory::advanceAges() {
    TrainingSample decoded;
    for (Int8QuantizedTrainingSample& stored : samples) {
        sample_quantizer.decodeInto(stored, decoded);
        ++decoded.age;
        sample_quantizer.encodeInto(decoded, stored);
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

size_t QuantizedInt8FIFOMemory::allocatedBytes() const {
    size_t bytes = sizeof(*this) + samples.capacity() * sizeof(Int8QuantizedTrainingSample);
    for (const Int8QuantizedTrainingSample& sample : samples) {
        bytes += sample.input.values.capacity() * sizeof(std::int8_t);
        bytes += sample.target.values.capacity() * sizeof(std::int8_t);
    }
    return bytes;
}
