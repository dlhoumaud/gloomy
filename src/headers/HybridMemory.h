#ifndef HYBRID_MEMORY_H
#define HYBRID_MEMORY_H

#include "LearningMemory.h"
#include <cstdint>
#include <random>
#include <vector>

struct HybridMemoryRatios {
    double recent = 0.25;
    double error = 0.25;
    double novelty = 0.25;
    double historical = 0.25;
};

class HybridMemory final : public LearningMemory {
public:
    HybridMemory(
        size_t capacity,
        HybridMemoryRatios ratios = {},
        double novelty_threshold = 0.0,
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

    HybridMemoryRatios ratios() const;
    std::vector<size_t> partitionSizes() const;

private:
    friend class LearningMemorySerialization;

    enum class Partition {
        Recent,
        Error,
        Novelty,
        Historical
    };

    struct StoredSample {
        TrainingSample sample;
        Partition partition;
    };

    static double squaredDistance(
        const std::vector<double>& left,
        const std::vector<double>& right
    );
    Partition choosePartition(const TrainingSample& sample) const;
    size_t partitionCapacity(Partition partition) const;
    size_t partitionSize(Partition partition) const;
    // Index, parmi `partition_indices`, du membre dont TrainingSample::age
    // est le plus grand (le plus ancien reellement, pas juste le premier
    // trouve dans le vecteur des echantillons).
    size_t oldestIndexInPartition(const std::vector<size_t>& partition_indices) const;
    // Fait entrer `sample` dans la partition Historical, en evincant au
    // besoin un membre choisi aleatoirement (politique Historical
    // inchangee). Utilise pour promouvoir le membre le plus ancien de
    // Recent lorsqu'il vieillit hors de Recent (vraie recence, voir add()).
    void demoteToHistorical(const TrainingSample& sample);

    size_t memory_capacity;
    HybridMemoryRatios memory_ratios;
    size_t recent_capacity;
    size_t error_capacity;
    size_t novelty_capacity;
    size_t historical_capacity;
    double distance_threshold;
    std::uint64_t seen_samples = 0;
    std::mt19937 generator;
    std::vector<StoredSample> samples;
};

#endif // HYBRID_MEMORY_H
