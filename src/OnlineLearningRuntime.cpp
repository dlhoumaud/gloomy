#include "headers/OnlineLearningRuntime.h"
#include "headers/NeuralNetwork.h"
#include "headers/LossFunction.h"
#include "headers/Optimizer.h"
#include "headers/MomentumOptimizer.h"
#include "headers/AdamOptimizer.h"
#include "headers/CompressedAdamOptimizer.h"
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
#include "headers/OptimizerSerialization.h"
#include "headers/LearningMemorySerialization.h"
#include "headers/ConceptDriftDetector.h"
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
    if (config.optimizer == "compressed_adam") {
        return std::make_unique<CompressedAdamOptimizer>(config.learning_rate, config.beta1, config.beta2, config.epsilon);
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
            config.memory_capacity, config.prioritized_alpha, config.seed, config.prioritized_beta,
            config.prioritized_beta_annealing_rate, config.prioritized_exploration_epsilon
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

ImportanceWeights makeImportanceWeights(const GloomyConfig& config) {
    return ImportanceWeights{
        config.importance_weight_error,
        config.importance_weight_novelty,
        config.importance_weight_rarity,
        config.importance_weight_recency,
        config.importance_weight_diversity
    };
}
}

OnlineLearningResult runOnlineLearning(
    const GloomyConfig& config,
    const std::vector<double>& sequence
) {
    return runOnlineLearning(config, sequence, "");
}

namespace {
// Verifie que la dimension d'entree du reseau (neuf ou repris d'un modele
// sauvegarde) correspond bien a config.window_size, pour eviter de melanger
// silencieusement un window_size de configuration avec un reseau entraine
// avec une autre fenetre (voir docs/roadmap.md, « Priorité haute »).
void validateNetworkWindowSize(const NeuralNetwork& network, const GloomyConfig& config) {
    if (network.layers().empty()) {
        throw std::invalid_argument("Network has no layers");
    }
    const std::size_t actual_input_size = network.layers().front().weights().size();
    if (actual_input_size != config.window_size) {
        throw std::invalid_argument(
            "Resumed model's input window size does not match config.window_size"
        );
    }
}
}

void saveTrainingArtifacts(
    const GloomyConfig& config,
    const TrainingResult& result
) {
    if (!config.model_path.empty()) {
        ModelSerialization::save(
            config.model_path,
            result.network,
            *result.normalizer,
            *result.optimizer,
            *result.memory
        );
    }
    if (!config.optimizer_path.empty()) {
        OptimizerSerialization::save(config.optimizer_path, *result.optimizer);
    }
    if (!config.memory_path.empty()) {
        LearningMemorySerialization::save(config.memory_path, *result.memory);
    }
    if (!config.metrics_path.empty()) {
        std::ofstream metrics(config.metrics_path, std::ios::trunc);
        if (!metrics) {
            throw std::runtime_error("Unable to open metrics file for writing: " + config.metrics_path);
        }
        metrics << "average_loss=" << result.average_loss << '\n';
    }
}

void saveOnlineArtifacts(
    const GloomyConfig& config,
    const OnlineLearningResult& result
) {
    if (!config.model_path.empty()) {
        ModelSerialization::save(
            config.model_path,
            result.network,
            *result.normalizer,
            *result.optimizer,
            *result.memory
        );
    }
    if (!config.optimizer_path.empty()) {
        OptimizerSerialization::save(config.optimizer_path, *result.optimizer);
    }
    if (!config.memory_path.empty()) {
        LearningMemorySerialization::save(config.memory_path, *result.memory);
    }
    if (!config.metrics_path.empty()) {
        std::ofstream metrics(config.metrics_path, std::ios::trunc);
        if (!metrics) {
            throw std::runtime_error("Unable to open metrics file for writing: " + config.metrics_path);
        }
        metrics << "average_loss=" << result.average_loss
                << " memory_size=" << result.memory_size << '\n';
    }
}

TrainingResult runTraining(
    const GloomyConfig& config,
    const std::vector<double>& sequence
) {
    return runTraining(config, sequence, "");
}

TrainingResult runTraining(
    const GloomyConfig& config,
    const std::vector<double>& sequence,
    const std::string& model_path
) {
    if (config.window_size == 0) {
        throw std::invalid_argument("window_size must be positive");
    }
    if (sequence.size() <= config.window_size) {
        throw std::invalid_argument(
            "Training requires more values in the input sequence than window_size"
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
        const int window_size = static_cast<int>(config.window_size);
        if (config.hidden_layers <= 0) {
            network->addLayer(window_size, 1);
        } else {
            network->addLayer(window_size, config.neurons);
            for (int index = 1; index < config.hidden_layers; ++index) {
                network->addLayer(config.neurons, config.neurons);
            }
            network->addLayer(config.neurons, 1);
        }

        optimizer = makeOptimizer(config);
        memory = makeMemory(config);
        normalizer = std::make_unique<StreamingNormalizer>(1);
    }
    validateNetworkWindowSize(*network, config);

    // observation/target/scratch sont reutilises d'une iteration a l'autre
    // (voir docs/roadmap.md, « Priorité moyenne : compression et embarqué ») :
    // une fois leur capacite etablie a la premiere iteration, l'affectation
    // d'un element ou normalize(..., out) ne reallouent plus. Seule la copie
    // finale dans `samples` (necessaire, le dataset doit survivre a la
    // boucle) alloue encore.
    std::vector<double> observation(config.window_size);
    std::vector<double> raw_target(1);
    std::vector<double> target(1);
    std::vector<double> scratch(1);
    std::vector<double> normalized_scratch(1);

    // Bootstrap : sequence[0 .. window_size-2] n'apparaissent jamais comme
    // la valeur la plus recente entrant dans une fenetre (ce role commence a
    // sequence[window_size-1], mise a jour au debut de l'iteration 0
    // ci-dessous) — elles ne seraient donc jamais vues par le normaliseur
    // sans cette boucle prealable (voir docs/roadmap.md, « Priorité
    // haute »).
    for (std::size_t index = 0; index + 1 < config.window_size; ++index) {
        scratch[0] = sequence[index];
        normalizer->update(scratch);
    }

    std::vector<TrainingSample> samples;
    samples.reserve(sequence.size() - config.window_size);
    for (std::size_t step = 0; step + config.window_size < sequence.size(); ++step) {
        scratch[0] = sequence[step + config.window_size - 1];
        normalizer->update(scratch);

        for (std::size_t position = 0; position < config.window_size; ++position) {
            scratch[0] = sequence[step + position];
            normalizer->normalize(scratch, normalized_scratch);
            observation[position] = normalized_scratch[0];
        }

        raw_target[0] = sequence[step + config.window_size];
        normalizer->normalize(raw_target, target);
        samples.push_back({observation, target});
    }

    LearningEngine engine(*network, *loss, *optimizer, makeImportanceWeights(config));
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
    if (config.window_size == 0) {
        throw std::invalid_argument("window_size must be positive");
    }
    if (sequence.size() <= config.window_size) {
        throw std::invalid_argument(
            "Online learning requires more values in the input sequence than window_size"
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
        const int window_size = static_cast<int>(config.window_size);
        if (config.hidden_layers <= 0) {
            network->addLayer(window_size, 1);
        } else {
            network->addLayer(window_size, config.neurons);
            for (int index = 1; index < config.hidden_layers; ++index) {
                network->addLayer(config.neurons, config.neurons);
            }
            network->addLayer(config.neurons, 1);
        }

        optimizer = makeOptimizer(config);
        memory = makeMemory(config);
        normalizer = std::make_unique<StreamingNormalizer>(1);
    }
    validateNetworkWindowSize(*network, config);

    LearningEngine engine(*network, *loss, *optimizer, makeImportanceWeights(config));

    OnlineLearningResult result;
    result.steps.reserve(sequence.size() - config.window_size);
    double total_loss = 0.0;

    // Buffers reutilises d'une iteration a l'autre plutot que reconstruits :
    // une fois leur capacite etablie, plus aucune allocation pour eux dans
    // la boucle (voir docs/roadmap.md, « Priorité moyenne : compression et
    // embarqué »). `sample` est repris a l'identique par
    // LearningEngine::learn(), qui en fait immediatement sa propre copie
    // avant de la modifier — le reutiliser ici ne fait donc courir aucun
    // risque d'aliasing avec ce qui est stocke en memoire.
    std::vector<double> observation(config.window_size);
    std::vector<double> raw_target(1);
    std::vector<double> target(1);
    std::vector<double> scratch(1);
    std::vector<double> normalized_scratch(1);
    TrainingSample sample;

    // Detection active de concept drift, opt-in (voir GloomyConfig et
    // docs/roadmap.md, « Priorité moyenne : mémoire et continual learning »).
    // Desactivee par defaut : drift_detector reste nul et le comportement
    // est identique a avant ce mecanisme.
    std::unique_ptr<ConceptDriftDetector> drift_detector;
    if (config.concept_drift_detection) {
        drift_detector = std::make_unique<ConceptDriftDetector>(
            config.concept_drift_recent_window,
            config.concept_drift_minimum_history,
            config.concept_drift_std_devs
        );
    }

    // Bootstrap : voir le commentaire equivalent dans runTraining ci-dessus.
    for (std::size_t index = 0; index + 1 < config.window_size; ++index) {
        scratch[0] = sequence[index];
        normalizer->update(scratch);
    }

    for (std::size_t step = 0; step + config.window_size < sequence.size(); ++step) {
        scratch[0] = sequence[step + config.window_size - 1];
        normalizer->update(scratch);

        for (std::size_t position = 0; position < config.window_size; ++position) {
            scratch[0] = sequence[step + position];
            normalizer->normalize(scratch, normalized_scratch);
            observation[position] = normalized_scratch[0];
        }

        raw_target[0] = sequence[step + config.window_size];
        normalizer->normalize(raw_target, target);

        const std::vector<double> prediction = network->forward(observation);

        sample.input = observation;
        sample.target = target;
        const double step_loss = engine.learn(*memory, sample, config.batch_size, *scheduler);

        OnlineLearningStep online_step;
        online_step.observation = sequence[step + config.window_size - 1];
        online_step.target = sequence[step + config.window_size];
        online_step.prediction_before_update = prediction[0];
        online_step.loss_before_update = step_loss;

        if (drift_detector) {
            online_step.drift_detected = drift_detector->update(step_loss);
            if (online_step.drift_detected && memory->size() > 0) {
                // Action declenchee par le signal : un replay immediat,
                // en plus de la mise a jour normalement planifiee par le
                // scheduler ci-dessus (voir docs/roadmap.md).
                engine.trainFromMemory(*memory, config.batch_size);
            }
        }

        result.steps.push_back(online_step);
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
