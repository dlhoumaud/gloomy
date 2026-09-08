#ifndef RESERVOIR_MEMORY_H
#define RESERVOIR_MEMORY_H

#include "LearningMemory.h"
#include <cstdint>
#include <random>
#include <vector>

class ReservoirMemory final : public LearningMemory {
public:
    explicit ReservoirMemory(size_t capacity, std::uint32_t seed = 5489u);

    void add(const TrainingSample& sample) override;
    void remove(size_t index) override;
    std::vector<TrainingSample> sample(size_t batch_size) override;
    std::vector<MemoryEntry> sampleIndexed(size_t batch_size) override;
    void update(size_t index, const TrainingSample& sample) override;
    void advanceAges() override;
    size_t size() const override;
    size_t capacity() const override;
    void clear() override;

private:
    friend class LearningMemorySerialization;

    size_t memory_capacity;
    size_t seen_samples = 0;
    std::mt19937 generator;
    std::vector<TrainingSample> samples;
};

#endif // RESERVOIR_MEMORY_H
