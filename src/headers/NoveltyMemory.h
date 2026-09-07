#ifndef NOVELTY_MEMORY_H
#define NOVELTY_MEMORY_H

#include "LearningMemory.h"
#include <vector>

class NoveltyMemory final : public LearningMemory {
public:
    NoveltyMemory(size_t capacity, double novelty_threshold);

    void add(const TrainingSample& sample) override;
    void remove(size_t index) override;
    std::vector<TrainingSample> sample(size_t batch_size) override;
    std::vector<MemoryEntry> sampleIndexed(size_t batch_size) override;
    void update(size_t index, const TrainingSample& sample) override;
    void advanceAges() override;
    size_t size() const override;
    size_t capacity() const override;
    void clear() override;

    double noveltyThreshold() const;

private:
    static double squaredDistance(
        const std::vector<double>& left,
        const std::vector<double>& right
    );
    size_t mostRedundantIndex() const;

    size_t memory_capacity;
    double distance_threshold;
    std::vector<TrainingSample> samples;
};

#endif // NOVELTY_MEMORY_H
