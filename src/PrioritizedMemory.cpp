#include "headers/PrioritizedMemory.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

PrioritizedMemory::PrioritizedMemory(
    size_t capacity,
    double alpha,
    std::uint32_t seed
)
    : memory_capacity(capacity), priority_exponent(alpha), generator(seed) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    if (alpha < 0.0 || !std::isfinite(alpha)) {
        throw std::invalid_argument("Priority exponent must be finite and non-negative");
    }
    samples.reserve(capacity);
}

void PrioritizedMemory::add(const TrainingSample& sample) {
    if (sample.priority < 0.0 || !std::isfinite(sample.priority)) {
        throw std::invalid_argument("Sample priority must be finite and non-negative");
    }

    if (samples.size() < memory_capacity) {
        samples.push_back(sample);
        return;
    }

    const auto lowest_priority = std::min_element(
        samples.begin(),
        samples.end(),
        [](const TrainingSample& left, const TrainingSample& right) {
            return left.priority < right.priority;
        }
    );
    if (sample.priority > lowest_priority->priority) {
        *lowest_priority = sample;
    }
}

void PrioritizedMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> PrioritizedMemory::sample(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }

    std::vector<size_t> indices(samples.size());
    for (size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }

    const size_t count = std::min(batch_size, samples.size());
    std::vector<TrainingSample> result;
    result.reserve(count);
    for (size_t selection = 0; selection < count; ++selection) {
        std::vector<double> weights;
        weights.reserve(indices.size());
        for (size_t index : indices) {
            const double priority = std::max(samples[index].priority, minimum_priority);
            weights.push_back(std::pow(priority, priority_exponent));
        }

        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        const size_t selected_position = distribution(generator);
        const size_t selected_index = indices[selected_position];
        ++samples[selected_index].usage_count;
        result.push_back(samples[selected_index]);
        indices.erase(indices.begin() + selected_position);
    }
    return result;
}

std::vector<MemoryEntry> PrioritizedMemory::sampleIndexed(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    std::vector<size_t> indices(samples.size());
    for (size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }
    const size_t count = std::min(batch_size, samples.size());
    std::vector<MemoryEntry> result;
    result.reserve(count);
    for (size_t selection = 0; selection < count; ++selection) {
        std::vector<double> weights;
        weights.reserve(indices.size());
        for (size_t index : indices) {
            weights.push_back(std::pow(std::max(samples[index].priority, minimum_priority), priority_exponent));
        }
        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        const size_t position = distribution(generator);
        const size_t selected = indices[position];
        ++samples[selected].usage_count;
        result.push_back({selected, samples[selected]});
        indices.erase(indices.begin() + position);
    }
    return result;
}

void PrioritizedMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index] = sample;
}

void PrioritizedMemory::advanceAges() {
    for (TrainingSample& sample : samples) {
        ++sample.age;
    }
}

size_t PrioritizedMemory::size() const {
    return samples.size();
}

size_t PrioritizedMemory::capacity() const {
    return memory_capacity;
}

void PrioritizedMemory::clear() {
    samples.clear();
}

double PrioritizedMemory::alpha() const {
    return priority_exponent;
}
