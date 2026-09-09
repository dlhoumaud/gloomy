#include "headers/LearningEngine.h"
#include "headers/ImportanceScorer.h"
#include <cmath>
#include <stdexcept>

namespace {
// Distance euclidienne au carre entre deux vecteurs de meme dimension
// (meme convention que NoveltyMemory, dupliquee ici car privee a cette
// classe). Utilisee pour les composantes novelty/diversity ci-dessous.
double squaredDistance(const std::vector<double>& left, const std::vector<double>& right) {
    double total = 0.0;
    for (size_t index = 0; index < left.size() && index < right.size(); ++index) {
        const double delta = left[index] - right[index];
        total += delta * delta;
    }
    return total;
}

// Ramene une distance (au carre, non bornee) dans [0, 1) : 0 pour une
// distance nulle, tend vers 1 pour une distance tres grande. Evite d'avoir
// a choisir une echelle de normalisation propre a chaque jeu de donnees.
double normalizeDistance(double raw_distance) {
    return raw_distance / (1.0 + raw_distance);
}
}

LearningEngine::LearningEngine(
    NeuralNetwork& network,
    const LossFunction& loss,
    Optimizer& optimizer,
    ImportanceWeights importance_weights
)
    : network(network), loss(loss), optimizer(optimizer), scorer(importance_weights) {}

double LearningEngine::trainBatch(const std::vector<TrainingSample>& batch) {
    return trainWeightedBatch(batch, std::vector<double>(batch.size(), 1.0));
}

double LearningEngine::trainWeightedBatch(
    const std::vector<TrainingSample>& batch,
    const std::vector<double>& sample_weights
) {
    if (batch.empty()) {
        throw std::invalid_argument("Training batch cannot be empty");
    }
    if (batch.size() != sample_weights.size()) {
        throw std::invalid_argument("Sample weights must match the batch size");
    }

    network.zeroGradients();
    double total_loss = 0.0;
    for (size_t index = 0; index < batch.size(); ++index) {
        const TrainingSample& sample = batch[index];
        const double weight = sample_weights[index];
        if (!std::isfinite(weight) || weight < 0.0) {
            throw std::invalid_argument("Sample weights must be finite and non-negative");
        }

        const std::vector<double> prediction = network.forward(sample.input);
        total_loss += loss.compute(prediction, sample.target);

        std::vector<double> gradient = loss.gradient(prediction, sample.target);
        for (double& value : gradient) {
            value *= weight;
        }
        network.backward(gradient);
    }

    optimizer.update(network.layers(), 1.0 / static_cast<double>(batch.size()));
    return total_loss / static_cast<double>(batch.size());
}

double LearningEngine::trainFromMemory(LearningMemory& memory, size_t batch_size) {
    if (memory.size() == 0) {
        throw std::invalid_argument("Cannot train from an empty learning memory");
    }

    memory.advanceAges();
    const std::vector<MemoryEntry> entries = memory.sampleIndexed(batch_size);
    std::vector<TrainingSample> batch;
    std::vector<double> sample_weights;
    batch.reserve(entries.size());
    sample_weights.reserve(entries.size());
    for (const MemoryEntry& entry : entries) {
        batch.push_back(entry.sample);
        sample_weights.push_back(entry.importance_weight);
    }

    // entry.importance_weight vaut 1.0 (neutre) pour toute strategie a
    // echantillonnage uniforme ; seule PrioritizedMemory le calcule
    // reellement, ce qui reproduit ici la correction de biais
    // d'echantillonnage du prioritized replay.
    const double batch_loss = trainWeightedBatch(batch, sample_weights);

    for (size_t index = 0; index < entries.size(); ++index) {
        const MemoryEntry& entry = entries[index];
        TrainingSample updated_sample = entry.sample;
        const std::vector<double> prediction = network.forward(updated_sample.input);
        updated_sample.error = 0.0;
        for (size_t output_index = 0; output_index < prediction.size(); ++output_index) {
            updated_sample.error = std::max(
                updated_sample.error,
                std::abs(prediction[output_index] - updated_sample.target[output_index])
            );
        }

        // recency/rarity : fonctions monotones bornees de l'age et du
        // nombre d'utilisations, deja mis a jour par advanceAges()/
        // sampleIndexed() ci-dessus (voir docs/memory.md).
        updated_sample.recency = 1.0 / (1.0 + static_cast<double>(updated_sample.age));
        updated_sample.rarity = 1.0 / (1.0 + static_cast<double>(updated_sample.usage_count));

        // novelty/diversity : calculees par rapport aux AUTRES echantillons
        // du meme batch (pas de toute la memoire, pour eviter d'elargir
        // l'interface LearningMemory) — novelty = distance au plus proche
        // voisin du batch, diversity = distance moyenne aux autres membres
        // du batch. Seul cas particulier : un batch d'un seul echantillon,
        // qui n'a personne a comparer (novelty maximale par convention,
        // diversite nulle par convention).
        if (entries.size() <= 1) {
            updated_sample.novelty = 1.0;
            updated_sample.diversity = 0.0;
        } else {
            double nearest = -1.0;
            double distance_sum = 0.0;
            for (size_t other_index = 0; other_index < entries.size(); ++other_index) {
                if (other_index == index) continue;
                const double raw_distance = squaredDistance(
                    updated_sample.input, entries[other_index].sample.input
                );
                distance_sum += raw_distance;
                if (nearest < 0.0 || raw_distance < nearest) nearest = raw_distance;
            }
            updated_sample.novelty = normalizeDistance(nearest);
            updated_sample.diversity = normalizeDistance(
                distance_sum / static_cast<double>(entries.size() - 1)
            );
        }

        updated_sample.priority = scorer.score({
            updated_sample.error,
            updated_sample.novelty,
            updated_sample.rarity,
            updated_sample.recency,
            updated_sample.diversity
        });
        memory.update(entry.index, updated_sample);
    }
    return batch_loss;
}

double LearningEngine::learn(
    LearningMemory& memory,
    const TrainingSample& sample,
    size_t batch_size
) {
    EverySampleScheduler scheduler;
    return learn(memory, sample, batch_size, scheduler);
}

double LearningEngine::learn(
    LearningMemory& memory,
    const TrainingSample& sample,
    size_t batch_size,
    TrainingScheduler& scheduler
) {
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    TrainingSample prioritized_sample = sample;
    const std::vector<double> prediction = network.forward(sample.input);
    const double sample_loss = loss.compute(prediction, sample.target);
    prioritized_sample.error = 0.0;
    for (size_t index = 0; index < prediction.size(); ++index) {
        prioritized_sample.error = std::max(
            prioritized_sample.error,
            std::abs(prediction[index] - sample.target[index])
        );
    }

    // recency/rarity : une observation qui vient d'arriver est par
    // definition la plus recente (age par defaut 0) et, si jamais utilisee
    // (usage_count par defaut 0), la plus rare possible. novelty/diversity
    // ne sont pas calculees ici : elles exigeraient d'interroger tout le
    // contenu de la memoire (pas seulement d'y ajouter), ce que
    // LearningMemory n'expose pas ; trainFromMemory() ci-dessus les calcule
    // par rapport au batch de replay, une fois l'echantillon deja stocke
    // (voir docs/memory.md, « Score d'importance »).
    prioritized_sample.recency = 1.0 / (1.0 + static_cast<double>(prioritized_sample.age));
    prioritized_sample.rarity = 1.0 / (1.0 + static_cast<double>(prioritized_sample.usage_count));
    prioritized_sample.priority = scorer.score({
        prioritized_sample.error,
        0.0,
        prioritized_sample.rarity,
        prioritized_sample.recency,
        0.0
    });

    memory.add(prioritized_sample);
    if (!scheduler.shouldTrain(prioritized_sample.error)) {
        return sample_loss;
    }
    return trainFromMemory(memory, batch_size);
}

double LearningEngine::train(
    const std::vector<TrainingSample>& samples,
    size_t epochs,
    size_t batch_size
) {
    if (samples.empty()) {
        throw std::invalid_argument("Training samples cannot be empty");
    }
    if (epochs == 0) {
        throw std::invalid_argument("Epoch count must be positive");
    }
    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be positive");
    }

    double epoch_loss = 0.0;
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        epoch_loss = 0.0;
        size_t processed_samples = 0;
        while (processed_samples < samples.size()) {
            const size_t remaining = samples.size() - processed_samples;
            const size_t current_batch_size = remaining < batch_size ? remaining : batch_size;
            const std::vector<TrainingSample> batch(
                samples.begin() + processed_samples,
                samples.begin() + processed_samples + current_batch_size
            );
            epoch_loss += trainBatch(batch) * static_cast<double>(current_batch_size);
            processed_samples += current_batch_size;
        }
        epoch_loss /= static_cast<double>(samples.size());
    }

    return epoch_loss;
}
