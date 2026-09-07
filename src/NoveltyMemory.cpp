#include "headers/NoveltyMemory.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

NoveltyMemory::NoveltyMemory(size_t capacity, double novelty_threshold)
    : memory_capacity(capacity), distance_threshold(novelty_threshold) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    if (novelty_threshold < 0.0 || !std::isfinite(novelty_threshold)) {
        throw std::invalid_argument("Novelty threshold must be finite and non-negative");
    }
    samples.reserve(capacity);
}

void NoveltyMemory::add(const TrainingSample& sample) {
    if (sample.input.empty()) {
        throw std::invalid_argument("Novelty sample input cannot be empty");
    }
    if (!samples.empty() && sample.input.size() != samples.front().input.size()) {
        throw std::invalid_argument("Novelty sample input dimensions must match");
    }

    if (samples.empty()) {
        samples.push_back(sample);
        return;
    }

    double nearest_distance = std::numeric_limits<double>::infinity();
    for (const TrainingSample& stored_sample : samples) {
        nearest_distance = std::min(
            nearest_distance,
            squaredDistance(sample.input, stored_sample.input)
        );
    }
    if (nearest_distance < distance_threshold) {
        return;
    }

    if (samples.size() < memory_capacity) {
        samples.push_back(sample);
        return;
    }

    samples[mostRedundantIndex()] = sample;
}

void NoveltyMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> NoveltyMemory::sample(size_t batch_size) {
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

std::vector<MemoryEntry> NoveltyMemory::sampleIndexed(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    const size_t count = std::min(batch_size, samples.size());
    std::vector<MemoryEntry> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        ++samples[index].usage_count;
        result.push_back({index, samples[index]});
    }
    return result;
}

void NoveltyMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index] = sample;
}

void NoveltyMemory::advanceAges() {
    for (TrainingSample& sample : samples) {
        ++sample.age;
    }
}

size_t NoveltyMemory::size() const {
    return samples.size();
}

size_t NoveltyMemory::capacity() const {
    return memory_capacity;
}

void NoveltyMemory::clear() {
    samples.clear();
}

double NoveltyMemory::noveltyThreshold() const {
    return distance_threshold;
}

double NoveltyMemory::squaredDistance(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    if (left.size() != right.size()) {
        throw std::invalid_argument("Novelty distance dimensions must match");
    }

    double distance = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        distance += difference * difference;
    }
    return distance;
}

size_t NoveltyMemory::mostRedundantIndex() const {
    size_t redundant_index = 0;
    double smallest_nearest_distance = std::numeric_limits<double>::infinity();

    for (size_t candidate = 0; candidate < samples.size(); ++candidate) {
        double nearest_distance = std::numeric_limits<double>::infinity();
        for (size_t neighbor = 0; neighbor < samples.size(); ++neighbor) {
            if (candidate == neighbor) {
                continue;
            }
            nearest_distance = std::min(
                nearest_distance,
                squaredDistance(samples[candidate].input, samples[neighbor].input)
            );
        }
        if (nearest_distance < smallest_nearest_distance) {
            smallest_nearest_distance = nearest_distance;
            redundant_index = candidate;
        }
    }
    return redundant_index;
}
