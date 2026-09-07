#ifndef QUANTIZED_INT8_FIFO_MEMORY_H
#define QUANTIZED_INT8_FIFO_MEMORY_H

#include "Int8TrainingSampleQuantization.h"
#include "LearningMemory.h"
#include <vector>

class QuantizedInt8FIFOMemory final : public LearningMemory {
public:
    QuantizedInt8FIFOMemory(size_t capacity, Int8TrainingSampleQuantizer quantizer);

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
    Int8TrainingSampleQuantizer sample_quantizer;
    size_t input_dimensions = 0;
    size_t target_dimensions = 0;
    std::vector<Int8QuantizedTrainingSample> samples;
};

#endif // QUANTIZED_INT8_FIFO_MEMORY_H
