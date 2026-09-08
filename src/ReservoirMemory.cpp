#include "headers/ReservoirMemory.h"
#include <algorithm>
#include <stdexcept>

ReservoirMemory::ReservoirMemory(size_t capacity, std::uint32_t seed)
    : memory_capacity(capacity), generator(seed) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    samples.reserve(capacity);
}

void ReservoirMemory::add(const TrainingSample& sample) {
    validateTrainingSampleVectors(sample);
    ++seen_samples;
    if (samples.size() < memory_capacity) {
        samples.push_back(sample);
        return;
    }

    std::uniform_int_distribution<size_t> distribution(0, seen_samples - 1);
    const size_t replacement_index = distribution(generator);
    if (replacement_index < memory_capacity) {
        samples[replacement_index] = sample;
    }
}

void ReservoirMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> ReservoirMemory::sample(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }

    std::vector<size_t> indices(samples.size());
    for (size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }
    std::shuffle(indices.begin(), indices.end(), generator);

    const size_t count = std::min(batch_size, samples.size());
    std::vector<TrainingSample> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        ++samples[indices[index]].usage_count;
        result.push_back(samples[indices[index]]);
    }
    return result;
}

std::vector<MemoryEntry> ReservoirMemory::sampleIndexed(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    std::vector<size_t> indices(samples.size());
    for (size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }
    std::shuffle(indices.begin(), indices.end(), generator);
    const size_t count = std::min(batch_size, samples.size());
    std::vector<MemoryEntry> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        ++samples[indices[index]].usage_count;
        result.push_back({indices[index], samples[indices[index]]});
    }
    return result;
}

void ReservoirMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index] = sample;
}

void ReservoirMemory::advanceAges() {
    for (TrainingSample& sample : samples) {
        ++sample.age;
    }
}

size_t ReservoirMemory::size() const {
    return samples.size();
}

size_t ReservoirMemory::capacity() const {
    return memory_capacity;
}

void ReservoirMemory::clear() {
    samples.clear();
    seen_samples = 0;
}
