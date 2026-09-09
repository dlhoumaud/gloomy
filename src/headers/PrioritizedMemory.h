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
        std::uint32_t seed = 5489u,
        double beta = 0.4,
        double beta_annealing_rate = 0.0,
        double exploration_epsilon = 0.0
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
    double beta() const;
    double betaAnnealingRate() const;
    double explorationEpsilon() const;

private:
    friend class LearningMemorySerialization;

    static constexpr double minimum_priority = 1e-12;

    // Poids de tirage melangeant priorite et exploration uniforme sur un
    // sous-ensemble d'indices : P(i) = (1 - exploration_epsilon) * w_i / S
    // + exploration_epsilon / n, avec w_i = priority_i^alpha. epsilon = 0
    // (defaut) reproduit exactement la distribution priorisee pure (voir
    // docs/memory.md, « Exploration controlee »).
    std::vector<double> mixedWeights(const std::vector<size_t>& pool) const;

    size_t memory_capacity;
    double priority_exponent;
    double correction_exponent;
    // Incrementee (jusqu'a 1.0) apres chaque sampleIndexed(), pour renforcer
    // la correction de biais au fil de l'entrainement (pratique courante :
    // partir de 0.4 et tendre vers 1.0). 0.0 (defaut) desactive l'annealing
    // et laisse beta fixe, comme avant ce champ.
    double beta_annealing_rate;
    double exploration_epsilon;
    std::mt19937 generator;
    std::vector<TrainingSample> samples;
};

#endif // PRIORITIZED_MEMORY_H
