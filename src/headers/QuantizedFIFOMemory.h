#ifndef QUANTIZED_FIFO_MEMORY_H
#define QUANTIZED_FIFO_MEMORY_H

#include "LearningMemory.h"
#include "TrainingSampleQuantization.h"
#include <cstddef>
#include <vector>

class QuantizedFIFOMemory final : public LearningMemory {
public:
    QuantizedFIFOMemory(
        size_t capacity,
        TrainingSampleQuantizer quantizer
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

    size_t bytesPerSample() const;
    size_t memoryUsedBytes() const;

private:
    size_t memory_capacity;
    TrainingSampleQuantizer sample_quantizer;
    size_t input_dimensions;
    size_t target_dimensions;
    std::vector<QuantizedTrainingSample> samples;
};

#endif // QUANTIZED_FIFO_MEMORY_H
