#include "headers/FIFOMemory.h"
#include <algorithm>
#include <stdexcept>

FIFOMemory::FIFOMemory(size_t capacity)
    : memory_capacity(capacity) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    samples.reserve(capacity);
}

void FIFOMemory::add(const TrainingSample& sample) {
    if (samples.size() == memory_capacity) {
        samples.erase(samples.begin());
    }
    samples.push_back(sample);
}

void FIFOMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> FIFOMemory::sample(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }

    const size_t count = std::min(batch_size, samples.size());
    std::vector<TrainingSample> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        ++samples[index].usage_count;
        result.push_back(samples[index]);
    }
    return result;
}

std::vector<MemoryEntry> FIFOMemory::sampleIndexed(size_t batch_size) {
    std::vector<MemoryEntry> result;
    sampleIndexedInto(batch_size, result);
    return result;
}

void FIFOMemory::sampleIndexedInto(size_t batch_size, std::vector<MemoryEntry>& output) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    const size_t count = std::min(batch_size, samples.size());
    output.resize(count);
    for (size_t index = 0; index < count; ++index) {
        ++samples[index].usage_count;
        output[index].index = index;
        output[index].sample = samples[index];
    }
}

void FIFOMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index] = sample;
}

void FIFOMemory::advanceAges() {
    for (TrainingSample& sample : samples) {
        ++sample.age;
    }
}

size_t FIFOMemory::size() const {
    return samples.size();
}

size_t FIFOMemory::capacity() const {
    return memory_capacity;
}

size_t FIFOMemory::allocatedBytes() const {
    return sizeof(*this) + trainingSamplesAllocatedBytes(samples);
}

void FIFOMemory::clear() {
    samples.clear();
}
