#include "headers/OnlineLearningRuntime.h"
#include "headers/NeuralNetwork.h"
#include "headers/LossFunction.h"
#include "headers/Optimizer.h"
#include "headers/MomentumOptimizer.h"
#include "headers/AdamOptimizer.h"
#include "headers/LearningMemory.h"
#include "headers/FIFOMemory.h"
#include "headers/ReservoirMemory.h"
#include "headers/PrioritizedMemory.h"
#include "headers/NoveltyMemory.h"
#include "headers/HybridMemory.h"
#include "headers/TrainingScheduler.h"
#include "headers/LearningEngine.h"
#include "headers/Normalization.h"
#include "headers/ModelSerialization.h"
#include <fstream>
#include <memory>
#include <stdexcept>

namespace {
std::unique_ptr<LossFunction> makeLoss(const GloomyConfig& config) {
    if (config.loss == "mse") return std::make_unique<MSELoss>();
    if (config.loss == "mae") return std::make_unique<MAELoss>();
    if (config.loss == "huber") return std::make_unique<HuberLoss>(config.huber_delta);
    throw std::invalid_argument("Unknown loss function: " + config.loss);
}

std::unique_ptr<Optimizer> makeOptimizer(const GloomyConfig& config) {
    if (config.optimizer == "sgd") return std::make_unique<SGDOptimizer>(config.learning_rate);
    if (config.optimizer == "momentum") {
        return std::make_unique<MomentumOptimizer>(config.learning_rate, config.momentum);
    }
    if (config.optimizer == "adam") {
        return std::make_unique<AdamOptimizer>(config.learning_rate, config.beta1, config.beta2, config.epsilon);
    }
    throw std::invalid_argument("Unknown optimizer: " + config.optimizer);
}

std::unique_ptr<LearningMemory> makeMemory(const GloomyConfig& config) {
    if (config.memory_strategy == "fifo") {
        return std::make_unique<FIFOMemory>(config.memory_capacity);
    }
    if (config.memory_strategy == "reservoir") {
        return std::make_unique<ReservoirMemory>(config.memory_capacity, config.seed);
    }
    if (config.memory_strategy == "prioritized") {
        return std::make_unique<PrioritizedMemory>(
            config.memory_capacity, config.prioritized_alpha, config.seed, config.prioritized_beta
        );
    }
    if (config.memory_strategy == "novelty") {
        return std::make_unique<NoveltyMemory>(config.memory_capacity, config.novelty_threshold);
    }
    if (config.memory_strategy == "hybrid") {
        const HybridMemoryRatios ratios{
            config.recent_ratio, config.error_ratio, config.novelty_ratio, config.historical_ratio
        };
        return std::make_unique<HybridMemory>(
            config.memory_capacity, ratios, config.novelty_threshold, config.seed
        );
    }
    throw std::invalid_argument("Unknown memory strategy: " + config.memory_strategy);
}

std::unique_ptr<TrainingScheduler> makeScheduler(const GloomyConfig& config) {
    if (config.train_every <= 1) return std::make_unique<EverySampleScheduler>();
    return std::make_unique<EveryNScheduler>(config.train_every);
}
}

OnlineLearningResult runOnlineLearning(
    const GloomyConfig& config,
    const std::vector<double>& sequence
) {
    return runOnlineLearning(config, sequence, "");
}

TrainingResult runTraining(
    const GloomyConfig& config,
    const std::vector<double>& sequence
) {
    if (sequence.size() < 2) {
        throw std::invalid_argument(
            "Training requires at least two values in the input sequence"
        );
    }
    if (config.batch_size == 0) {
        throw std::invalid_argument("batch_size must be positive");
    }
    if (config.epochs == 0) {
        throw std::invalid_argument("epochs must be positive");
    }

    const std::unique_ptr<LossFunction> loss = makeLoss(config);
    auto network = std::make_unique<NeuralNetwork>();
    network->algorithm = config.activation;
    network->post_algorithm = config.post_activation;
    if (config.hidden_layers <= 0) {
        network->addLayer(1, 1);
    } else {
        network->addLayer(1, config.neurons);
        for (int index = 1; index < config.hidden_layers; ++index) {
            network->addLayer(config.neurons, config.neurons);
        }
        network->addLayer(config.neurons, 1);
    }

    auto normalizer = std::make_unique<StreamingNormalizer>(1);
    auto optimizer = makeOptimizer(config);
    auto memory = makeMemory(config);

    std::vector<TrainingSample> samples;
    samples.reserve(sequence.size() - 1);
    for (std::size_t index = 0; index + 1 < sequence.size(); ++index) {
        const std::vector<double> raw_observation = {sequence[index]};
        const std::vector<double> raw_target = {sequence[index + 1]};

        normalizer->update(raw_observation);
        const std::vector<double> observation = normalizer->normalize(raw_observation);
        const std::vector<double> target = normalizer->normalize(raw_target);
        samples.push_back({observation, target});
    }

    LearningEngine engine(*network, *loss, *optimizer);
    const double average_loss = engine.train(samples, config.epochs, config.batch_size);

    TrainingResult result;
    result.average_loss = average_loss;
    result.network = std::move(*network);
    result.normalizer = std::move(normalizer);
    result.optimizer = std::move(optimizer);
    result.memory = std::move(memory);
    return result;
}

OnlineLearningResult runOnlineLearning(
    const GloomyConfig& config,
    const std::vector<double>& sequence,
    const std::string& model_path
) {
    if (sequence.size() < 2) {
        throw std::invalid_argument(
            "Online learning requires at least two values in the input sequence"
        );
    }
    if (config.batch_size == 0) {
        throw std::invalid_argument("batch_size must be positive");
    }

    const std::unique_ptr<LossFunction> loss = makeLoss(config);
    const std::unique_ptr<TrainingScheduler> scheduler = makeScheduler(config);

    auto network = std::make_unique<NeuralNetwork>();
    auto normalizer = std::make_unique<StreamingNormalizer>(1);
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<LearningMemory> memory;

    if (!model_path.empty()) {
        std::ifstream input(model_path, std::ios::binary);
        if (input.good()) {
            ModelSerialization::LoadedModel loaded = ModelSerialization::load(model_path);
            network = std::make_unique<NeuralNetwork>(std::move(loaded.network));
            normalizer = std::move(loaded.normalizer);
            optimizer = std::move(loaded.optimizer);
            memory = std::move(loaded.memory);
        }
    }

    if (!optimizer || !memory || !normalizer) {
        network->algorithm = config.activation;
        network->post_algorithm = config.post_activation;
        if (config.hidden_layers <= 0) {
            network->addLayer(1, 1);
        } else {
            network->addLayer(1, config.neurons);
            for (int index = 1; index < config.hidden_layers; ++index) {
                network->addLayer(config.neurons, config.neurons);
            }
            network->addLayer(config.neurons, 1);
        }

        optimizer = makeOptimizer(config);
        memory = makeMemory(config);
        normalizer = std::make_unique<StreamingNormalizer>(1);
    }

    LearningEngine engine(*network, *loss, *optimizer);

    OnlineLearningResult result;
    result.steps.reserve(sequence.size() - 1);
    double total_loss = 0.0;

    for (std::size_t index = 0; index + 1 < sequence.size(); ++index) {
        const std::vector<double> raw_observation = {sequence[index]};
        const std::vector<double> raw_target = {sequence[index + 1]};

        normalizer->update(raw_observation);
        const std::vector<double> observation = normalizer->normalize(raw_observation);
        const std::vector<double> target = normalizer->normalize(raw_target);

        const std::vector<double> prediction = network->forward(observation);

        TrainingSample sample;
        sample.input = observation;
        sample.target = target;
        const double step_loss = engine.learn(*memory, sample, config.batch_size, *scheduler);

        OnlineLearningStep step;
        step.observation = raw_observation[0];
        step.target = raw_target[0];
        step.prediction_before_update = prediction[0];
        step.loss_before_update = step_loss;
        result.steps.push_back(step);
        total_loss += step_loss;
    }

    result.network = std::move(*network);
    result.normalizer = std::move(normalizer);
    result.optimizer = std::move(optimizer);
    result.memory = std::move(memory);

    result.average_loss = total_loss / static_cast<double>(result.steps.size());
    result.memory_size = result.memory->size();
    return result;
}
