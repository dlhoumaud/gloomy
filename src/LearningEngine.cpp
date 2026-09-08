#include "headers/LearningEngine.h"
#include "headers/ImportanceScorer.h"
#include <cmath>
#include <stdexcept>

LearningEngine::LearningEngine(
    NeuralNetwork& network,
    const LossFunction& loss,
    Optimizer& optimizer
)
    : network(network), loss(loss), optimizer(optimizer) {}

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
    ImportanceScorer scorer;
    for (const MemoryEntry& entry : entries) {
        TrainingSample updated_sample = entry.sample;
        const std::vector<double> prediction = network.forward(updated_sample.input);
        updated_sample.error = 0.0;
        for (size_t index = 0; index < prediction.size(); ++index) {
            updated_sample.error = std::max(
                updated_sample.error,
                std::abs(prediction[index] - updated_sample.target[index])
            );
        }
        updated_sample.priority = scorer.score({updated_sample.error});
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
    ImportanceScorer scorer;
    prioritized_sample.priority = scorer.score({prioritized_sample.error});

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
