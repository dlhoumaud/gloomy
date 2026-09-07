#ifndef PRIORITIZED_MEMORY_H
#define PRIORITIZED_MEMORY_H

#include "LearningMemory.h"
#include <cstdint>
#include <random>
#include <vector>

class PrioritizedMemory final : public LearningMemory {
public:
    explicit PrioritizedMemory(
        size_t capacity,
        double alpha = 0.6,
        std::uint32_t seed = 5489u
    );

    void add(const TrainingSample& sample) override;
    void remove(size_t index) override;
    std::vector<TrainingSample> sample(size_t batch_size) override;
    std::vector<MemoryEntry> sampleIndexed(size_t batch_size) override;
    void update(size_t index, const TrainingSample& sample) override;
    void advanceAges() override;
    size_t size() const override;
    size_t capacity() const override;
    void clear() override;

    double alpha() const;

private:
    static constexpr double minimum_priority = 1e-12;

    size_t memory_capacity;
    double priority_exponent;
    std::mt19937 generator;
    std::vector<TrainingSample> samples;
};

#endif // PRIORITIZED_MEMORY_H
