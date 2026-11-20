#include "headers/LearningEngine.h"
#include "headers/ImportanceScorer.h"
#include <cmath>
#include <stdexcept>

namespace {
template <typename SampleAt>
double trainBatchSamples(
    NeuralNetwork& network,
    const LossFunction& loss,
    Optimizer& optimizer,
    size_t sample_count,
    SampleAt sample_at
) {
    if (sample_count == 0) {
        throw std::invalid_argument("Training batch cannot be empty");
    }

    network.zeroGradients();
    double total_loss = 0.0;
    for (size_t index = 0; index < sample_count; ++index) {
        const TrainingSample& sample = sample_at(index);
        const std::vector<double>& prediction = network.forwardRef(sample.input);
        total_loss += loss.compute(prediction, sample.target);
        network.backwardRef(loss.gradient(prediction, sample.target));
    }
    optimizer.update(network.layers(), 1.0 / static_cast<double>(sample_count));
    return total_loss / static_cast<double>(sample_count);
}
}

LearningEngine::LearningEngine(
    NeuralNetwork& network,
    const LossFunction& loss,
    Optimizer& optimizer
)
    : network(network), loss(loss), optimizer(optimizer) {}

double LearningEngine::trainBatch(const std::vector<TrainingSample>& batch) {
    return trainBatchSamples(network, loss, optimizer, batch.size(),
        [&batch](size_t index) -> const TrainingSample& { return batch[index]; });
}

double LearningEngine::trainBatchRange(
    const std::vector<TrainingSample>& samples,
    size_t offset,
    size_t count
) {
    if (offset > samples.size() || count > samples.size() - offset) {
        throw std::out_of_range("Training batch range is out of bounds");
    }
    return trainBatchSamples(network, loss, optimizer, count,
        [&samples, offset](size_t index) -> const TrainingSample& { return samples[offset + index]; });
}

double LearningEngine::trainFromMemory(LearningMemory& memory, size_t batch_size) {
    if (memory.size() == 0) {
        throw std::invalid_argument("Cannot train from an empty learning memory");
    }

    memory.advanceAges();
    const std::vector<MemoryEntry> entries = memory.sampleIndexed(batch_size);
    const double batch_loss = trainBatchSamples(network, loss, optimizer, entries.size(),
        [&entries](size_t index) -> const TrainingSample& { return entries[index].sample; });
    ImportanceScorer scorer;
    for (const MemoryEntry& entry : entries) {
        TrainingSample updated_sample = entry.sample;
        const std::vector<double>& prediction = network.forwardRef(updated_sample.input);
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
    const std::vector<double>& prediction = network.forwardRef(sample.input);
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
            epoch_loss += trainBatchRange(samples, processed_samples, current_batch_size) *
                static_cast<double>(current_batch_size);
            processed_samples += current_batch_size;
        }
        epoch_loss /= static_cast<double>(samples.size());
    }

    return epoch_loss;
}
