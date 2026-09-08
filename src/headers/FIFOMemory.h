#ifndef FIFO_MEMORY_H
#define FIFO_MEMORY_H

#include "LearningMemory.h"
#include <vector>

class FIFOMemory final : public LearningMemory {
public:
    explicit FIFOMemory(size_t capacity);

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
    std::vector<TrainingSample> samples;
};

#endif // FIFO_MEMORY_H
