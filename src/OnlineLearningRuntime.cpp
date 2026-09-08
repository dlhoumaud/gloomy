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
    if (sequence.size() < 2) {
        throw std::invalid_argument(
            "Online learning requires at least two values in the input sequence"
        );
    }
    if (config.batch_size == 0) {
        throw std::invalid_argument("batch_size must be positive");
    }

    // Réseau scalaire : chaque observation et chaque cible sont un unique
    // double, adapté à un flux d'apprentissage en continu.
    NeuralNetwork network;
    network.algorithm = config.activation;
    network.post_algorithm = config.post_activation;
    if (config.hidden_layers <= 0) {
        network.addLayer(1, 1);
    } else {
        network.addLayer(1, config.neurons);
        for (int index = 1; index < config.hidden_layers; ++index) {
            network.addLayer(config.neurons, config.neurons);
        }
        network.addLayer(config.neurons, 1);
    }

    const std::unique_ptr<LossFunction> loss = makeLoss(config);
    const std::unique_ptr<Optimizer> optimizer = makeOptimizer(config);
    const std::unique_ptr<LearningMemory> memory = makeMemory(config);
    const std::unique_ptr<TrainingScheduler> scheduler = makeScheduler(config);

    LearningEngine engine(network, *loss, *optimizer);
    StreamingNormalizer normalizer(1);

    OnlineLearningResult result;
    result.steps.reserve(sequence.size() - 1);
    double total_loss = 0.0;

    for (std::size_t index = 0; index + 1 < sequence.size(); ++index) {
        const std::vector<double> raw_observation = {sequence[index]};
        const std::vector<double> raw_target = {sequence[index + 1]};

        normalizer.update(raw_observation);
        const std::vector<double> observation = normalizer.normalize(raw_observation);
        const std::vector<double> target = normalizer.normalize(raw_target);

        const std::vector<double> prediction = network.forward(observation);

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

    result.average_loss = total_loss / static_cast<double>(result.steps.size());
    result.memory_size = memory->size();
    return result;
}
