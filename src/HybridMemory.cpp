#include "headers/HybridMemory.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

HybridMemory::HybridMemory(
    size_t capacity,
    HybridMemoryRatios ratios,
    double novelty_threshold,
    std::uint32_t seed
)
    : memory_capacity(capacity), memory_ratios(ratios), distance_threshold(novelty_threshold), generator(seed) {
    if (capacity == 0) {
        throw std::invalid_argument("Memory capacity must be positive");
    }
    if (novelty_threshold < 0.0 || !std::isfinite(novelty_threshold)) {
        throw std::invalid_argument("Novelty threshold must be finite and non-negative");
    }
    const double ratio_sum = ratios.recent + ratios.error + ratios.novelty + ratios.historical;
    if (!std::isfinite(ratio_sum) || ratio_sum <= 0.0 ||
        ratios.recent < 0.0 || ratios.error < 0.0 ||
        ratios.novelty < 0.0 || ratios.historical < 0.0) {
        throw std::invalid_argument("Hybrid memory ratios must be finite and non-negative");
    }

    const double scale = 1.0 / ratio_sum;
    memory_ratios.recent *= scale;
    memory_ratios.error *= scale;
    memory_ratios.novelty *= scale;
    memory_ratios.historical *= scale;

    recent_capacity = static_cast<size_t>(memory_capacity * memory_ratios.recent);
    error_capacity = static_cast<size_t>(memory_capacity * memory_ratios.error);
    novelty_capacity = static_cast<size_t>(memory_capacity * memory_ratios.novelty);
    historical_capacity = static_cast<size_t>(memory_capacity * memory_ratios.historical);
    size_t assigned = recent_capacity + error_capacity + novelty_capacity + historical_capacity;
    recent_capacity += memory_capacity - assigned;
    samples.reserve(capacity);
}

void HybridMemory::add(const TrainingSample& sample) {
    validateTrainingSampleVectors(sample);
    ++seen_samples;
    const Partition partition = choosePartition(sample);
    const size_t limit = partitionCapacity(partition);
    if (limit == 0) {
        return;
    }

    std::vector<size_t> partition_indices;
    for (size_t index = 0; index < samples.size(); ++index) {
        if (samples[index].partition == partition) {
            partition_indices.push_back(index);
        }
    }

    if (partition_indices.size() < limit) {
        samples.push_back({sample, partition});
        return;
    }

    if (partition == Partition::Historical) {
        std::uniform_int_distribution<size_t> distribution(0, partition_indices.size() - 1);
        samples[partition_indices[distribution(generator)]] = {sample, partition};
        return;
    }

    // Recent/Error/Novelty : evincer le membre le plus ancien (age le plus
    // eleve), pas le premier trouve dans le vecteur — une position seule ne
    // reflete plus l'age reel une fois qu'un remplacement en place a deja
    // eu lieu dans cette partition (voir docs/memory.md).
    const size_t oldest_index = oldestIndexInPartition(partition_indices);
    if (partition == Partition::Recent) {
        // Vraie recence : le membre le plus ancien de Recent n'est pas
        // perdu, il est promu vers Historical, qui se peuple ainsi par
        // vieillissement reel plutot que par une alternance a l'admission.
        const TrainingSample demoted = samples[oldest_index].sample;
        samples[oldest_index] = {sample, Partition::Recent};
        demoteToHistorical(demoted);
        return;
    }
    samples[oldest_index] = {sample, partition};
}

void HybridMemory::remove(size_t index) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples.erase(samples.begin() + index);
}

std::vector<TrainingSample> HybridMemory::sample(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }

    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), generator);

    const size_t count = std::min(batch_size, samples.size());
    std::vector<TrainingSample> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        ++samples[indices[index]].sample.usage_count;
        result.push_back(samples[indices[index]].sample);
    }
    return result;
}

std::vector<MemoryEntry> HybridMemory::sampleIndexed(size_t batch_size) {
    if (batch_size == 0) {
        throw std::invalid_argument("Sample batch size must be positive");
    }
    std::vector<size_t> indices(samples.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), generator);
    const size_t count = std::min(batch_size, samples.size());
    std::vector<MemoryEntry> result;
    result.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        ++samples[indices[index]].sample.usage_count;
        result.push_back({indices[index], samples[indices[index]].sample});
    }
    return result;
}

void HybridMemory::update(size_t index, const TrainingSample& sample) {
    if (index >= samples.size()) {
        throw std::out_of_range("Memory sample index is out of range");
    }
    samples[index].sample = sample;
}

void HybridMemory::advanceAges() {
    for (StoredSample& stored : samples) {
        ++stored.sample.age;
    }
}

size_t HybridMemory::size() const {
    return samples.size();
}

size_t HybridMemory::capacity() const {
    return memory_capacity;
}

void HybridMemory::clear() {
    samples.clear();
    seen_samples = 0;
}

HybridMemoryRatios HybridMemory::ratios() const {
    return memory_ratios;
}

std::vector<size_t> HybridMemory::partitionSizes() const {
    return {
        partitionSize(Partition::Recent),
        partitionSize(Partition::Error),
        partitionSize(Partition::Novelty),
        partitionSize(Partition::Historical)
    };
}

double HybridMemory::squaredDistance(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    if (left.size() != right.size()) {
        return std::numeric_limits<double>::infinity();
    }
    double distance = 0.0;
    for (size_t index = 0; index < left.size(); ++index) {
        const double difference = left[index] - right[index];
        distance += difference * difference;
    }
    return distance;
}

HybridMemory::Partition HybridMemory::choosePartition(const TrainingSample& sample) const {
    if (error_capacity > 0 && sample.priority > 0.5) {
        return Partition::Error;
    }

    if (novelty_capacity > 0 && distance_threshold > 0.0) {
        for (const StoredSample& stored : samples) {
            if (squaredDistance(sample.input, stored.sample.input) >= distance_threshold) {
                return Partition::Novelty;
            }
        }
    }

    // Toute observation generique (ni erreur, ni nouveaute) est par
    // definition la plus recente au moment de son arrivee : elle rejoint
    // Recent directement, sans alternance. Recent evince ensuite son membre
    // le plus ancien (par age reel) vers Historical au lieu de le perdre
    // (voir add()), pour que Historical reste un veritable reservoir
    // d'anciens elements plutot qu'une seconde file alimentee par
    // alternance a l'admission.
    if (recent_capacity > 0) {
        return Partition::Recent;
    }
    if (historical_capacity > 0) {
        return Partition::Historical;
    }
    if (novelty_capacity > 0) {
        return Partition::Novelty;
    }
    return Partition::Error;
}

size_t HybridMemory::partitionCapacity(Partition partition) const {
    switch (partition) {
    case Partition::Recent: return recent_capacity;
    case Partition::Error: return error_capacity;
    case Partition::Novelty: return novelty_capacity;
    case Partition::Historical: return historical_capacity;
    }
    return 0;
}

size_t HybridMemory::partitionSize(Partition partition) const {
    return static_cast<size_t>(std::count_if(
        samples.begin(),
        samples.end(),
        [partition](const StoredSample& stored) { return stored.partition == partition; }
    ));
}

size_t HybridMemory::oldestIndexInPartition(const std::vector<size_t>& partition_indices) const {
    size_t oldest_index = partition_indices.front();
    std::size_t oldest_age = samples[oldest_index].sample.age;
    for (size_t index : partition_indices) {
        if (samples[index].sample.age > oldest_age) {
            oldest_age = samples[index].sample.age;
            oldest_index = index;
        }
    }
    return oldest_index;
}

void HybridMemory::demoteToHistorical(const TrainingSample& sample) {
    if (historical_capacity == 0) {
        // Pas de partition Historical configuree : l'element qui vieillit
        // hors de Recent est perdu, comme avant ce changement.
        return;
    }

    std::vector<size_t> historical_indices;
    for (size_t index = 0; index < samples.size(); ++index) {
        if (samples[index].partition == Partition::Historical) {
            historical_indices.push_back(index);
        }
    }

    if (historical_indices.size() < historical_capacity) {
        samples.push_back({sample, Partition::Historical});
        return;
    }

    std::uniform_int_distribution<size_t> distribution(0, historical_indices.size() - 1);
    samples[historical_indices[distribution(generator)]] = {sample, Partition::Historical};
}
