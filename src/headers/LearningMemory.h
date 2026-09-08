#ifndef LEARNING_MEMORY_H
#define LEARNING_MEMORY_H

#include "TrainingSample.h"
#include <cstddef>
#include <vector>

struct MemoryEntry {
    size_t index;
    TrainingSample sample;
    // Poids de correction du biais d'échantillonnage (importance-sampling
    // weight), déjà normalisé à 1.0 au maximum. Vaut 1.0 (neutre) pour
    // toutes les stratégies à échantillonnage uniforme ; seule
    // PrioritizedMemory le calcule réellement (voir docs/memory.md).
    double importance_weight = 1.0;
};

class LearningMemory {
public:
    virtual ~LearningMemory() = default;

    virtual void add(const TrainingSample& sample) = 0;
    virtual void remove(size_t index) = 0;
    virtual std::vector<TrainingSample> sample(size_t batch_size) = 0;
    virtual std::vector<MemoryEntry> sampleIndexed(size_t batch_size) = 0;
    virtual void update(size_t index, const TrainingSample& sample) = 0;
    virtual void advanceAges() = 0;
    virtual size_t size() const = 0;
    virtual size_t capacity() const = 0;
    virtual void clear() = 0;
};

#endif // LEARNING_MEMORY_H
