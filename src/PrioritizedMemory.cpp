#include "headers/PrioritizedMemory.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

PrioritizedMemory::PrioritizedMemory(
    size_t capacity,
    double alpha,
    std::uint32_t seed,
    double beta,
    double beta_annealing_rate,
    double exploration_epsilon
)
    : memory_capacity(capacity),
      priority_exponent(alpha),
      correction_exponent(beta),
      beta_annealing_rate(beta_annealing_rate),
      exploration_epsilon(exploration_epsilon),
      generator(seed) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    if (alpha < 0.0 || !std::isfinite(alpha)) {
        throw std::invalid_argument("Priority exponent must be finite and non-negative");
    }
    if (beta < 0.0 || !std::isfinite(beta)) {
        throw std::invalid_argument("Correction exponent (beta) must be finite and non-negative");
    }
    if (beta_annealing_rate < 0.0 || !std::isfinite(beta_annealing_rate)) {
        throw std::invalid_argument("Beta annealing rate must be finite and non-negative");
    }
    if (exploration_epsilon < 0.0 || exploration_epsilon > 1.0 || !std::isfinite(exploration_epsilon)) {
        throw std::invalid_argument("Exploration epsilon must be finite and in [0, 1]");
    }
    samples.reserve(capacity);
}

std::vector<double> PrioritizedMemory::mixedWeights(const std::vector<size_t>& pool) const {
    std::vector<double> weights(pool.size());
    double total = 0.0;
    for (size_t position = 0; position < pool.size(); ++position) {
        weights[position] = std::pow(std::max(samples[pool[position]].priority, minimum_priority), priority_exponent);
        total += weights[position];
    }
    if (exploration_epsilon <= 0.0 || pool.empty() || total <= 0.0) {
        return weights;
    }
    const double uniform_share = exploration_epsilon / static_cast<double>(pool.size());
    for (double& weight : weights) {
        weight = (1.0 - exploration_epsilon) * (weight / total) + uniform_share;
    }
    return weights;
}

void PrioritizedMemory::add(const TrainingSample& sample) {
    validateTrainingSampleVectors(sample);
    if (sample.priority < 0.0 || !std::isfinite(sample.priority)) {
        throw std::invalid_argument("Sample priority must be finite and non-negative");
    }

    if (samples.size() < memory_capacity) {
        samples.push_back(sample);
        return;
    }

    const auto lowest_priority = std::min_element(
        samples.begin(),
        samples.end(),
        [](const TrainingSample& left, const TrainingSample& right) {
            return left.priority < right.priority;
        }
    );
    if (sample.priority > lowest_priority->priority) {
        *lowest_priority = sample;
    }
}

void PrioritizedMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> PrioritizedMemory::sample(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }

    std::vector<size_t> indices(samples.size());
    for (size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }

    const size_t count = std::min(batch_size, samples.size());
    std::vector<TrainingSample> result;
    result.reserve(count);
    for (size_t selection = 0; selection < count; ++selection) {
        const std::vector<double> weights = mixedWeights(indices);
        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        const size_t selected_position = distribution(generator);
        const size_t selected_index = indices[selected_position];
        ++samples[selected_index].usage_count;
        result.push_back(samples[selected_index]);
        indices.erase(indices.begin() + selected_position);
    }
    return result;
}

std::vector<MemoryEntry> PrioritizedMemory::sampleIndexed(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    std::vector<size_t> indices(samples.size());
    for (size_t index = 0; index < indices.size(); ++index) {
        indices[index] = index;
    }

    // Distribution de reference sur l'ensemble de la memoire au moment du
    // tirage : P(i) = priority_i^alpha / somme_j(priority_j^alpha). Calculee
    // une seule fois, avant tout tirage sans remise, pour que le poids de
    // correction reflete la vraie probabilite de selection sous la
    // distribution priorisee plutot que la distribution residuelle apres
    // retrait des elements deja tires (voir docs/memory.md).
    std::vector<double> pool_weights(samples.size());
    double total_weight = 0.0;
    for (size_t index = 0; index < samples.size(); ++index) {
        pool_weights[index] = std::pow(std::max(samples[index].priority, minimum_priority), priority_exponent);
        total_weight += pool_weights[index];
    }

    const size_t count = std::min(batch_size, samples.size());
    std::vector<MemoryEntry> result;
    result.reserve(count);
    double max_raw_weight = 0.0;

    for (size_t selection = 0; selection < count; ++selection) {
        const std::vector<double> weights = mixedWeights(indices);
        std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
        const size_t position = distribution(generator);
        const size_t selected = indices[position];
        ++samples[selected].usage_count;

        // Poids d'importance-sampling non normalise : (N * P(i))^(-beta).
        // P(i) melange priorite et exploration uniforme, comme les poids de
        // tirage ci-dessus (voir mixedWeights et docs/memory.md).
        const double raw_probability = pool_weights[selected] / total_weight;
        const double probability = exploration_epsilon > 0.0
            ? (1.0 - exploration_epsilon) * raw_probability + exploration_epsilon / static_cast<double>(samples.size())
            : raw_probability;
        const double raw_weight = std::pow(static_cast<double>(samples.size()) * probability, -correction_exponent);
        max_raw_weight = std::max(max_raw_weight, raw_weight);

        result.push_back({selected, samples[selected], raw_weight});
        indices.erase(indices.begin() + position);
    }

    // Normalisation par le maximum du batch : le plus grand poids vaut 1,
    // pour ne jamais amplifier une mise a jour (stabilise l'entrainement).
    if (max_raw_weight > 0.0) {
        for (MemoryEntry& entry : result) {
            entry.importance_weight /= max_raw_weight;
        }
    }

    // Annealing : beta se rapproche de 1.0 au fil des replays si
    // beta_annealing_rate > 0 (defaut 0.0 = beta fixe, comportement
    // inchange). N'affecte que les tirages suivants, pas celui-ci.
    if (beta_annealing_rate > 0.0) {
        correction_exponent = std::min(1.0, correction_exponent + beta_annealing_rate);
    }

    return result;
}

void PrioritizedMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index] = sample;
}

void PrioritizedMemory::advanceAges() {
    for (TrainingSample& sample : samples) {
        ++sample.age;
    }
}

size_t PrioritizedMemory::size() const {
    return samples.size();
}

size_t PrioritizedMemory::capacity() const {
    return memory_capacity;
}

void PrioritizedMemory::clear() {
    samples.clear();
}

double PrioritizedMemory::alpha() const {
    return priority_exponent;
}

double PrioritizedMemory::beta() const {
    return correction_exponent;
}

double PrioritizedMemory::betaAnnealingRate() const {
    return beta_annealing_rate;
}

double PrioritizedMemory::explorationEpsilon() const {
    return exploration_epsilon;
}
