#ifndef LEARNING_MEMORY_H
#define LEARNING_MEMORY_H

#include "TrainingSample.h"
#include <cstddef>
#include <vector>

struct MemoryEntry {
    size_t index;
    TrainingSample sample;
};

inline size_t trainingSamplesAllocatedBytes(const std::vector<TrainingSample>& samples) {
    size_t bytes = samples.capacity() * sizeof(TrainingSample);
    for (const TrainingSample& sample : samples) bytes += trainingSampleDynamicBytes(sample);
    return bytes;
}

class LearningMemory {
public:
    virtual ~LearningMemory() = default;

    virtual void add(const TrainingSample& sample) = 0;
    virtual void remove(size_t index) = 0;
    virtual std::vector<TrainingSample> sample(size_t batch_size) = 0;
    virtual std::vector<MemoryEntry> sampleIndexed(size_t batch_size) = 0;
    virtual void sampleIndexedInto(size_t batch_size, std::vector<MemoryEntry>& output) {
        output = sampleIndexed(batch_size);
    }
    virtual void update(size_t index, const TrainingSample& sample) = 0;
    virtual void advanceAges() = 0;
    virtual size_t size() const = 0;
    virtual size_t capacity() const = 0;
    // Estimates retained allocations when supported; zero means unknown.
    virtual size_t allocatedBytes() const { return 0; }
    virtual void clear() = 0;
};

#endif // LEARNING_MEMORY_H
