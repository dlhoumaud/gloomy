#include "headers/AdamOptimizer.h"
#include "headers/Benchmark.h"
#include "headers/DenseLayer.h"
#include "headers/LearningEngine.h"
#include "headers/Metrics.h"
#include "headers/MomentumOptimizer.h"
#include "headers/FIFOMemory.h"
#include "headers/ReservoirMemory.h"
#include "headers/PrioritizedMemory.h"
#include "headers/NoveltyMemory.h"
#include "headers/HybridMemory.h"
#include "headers/LearningMemory.h"
#include "headers/QuantizedFIFOMemory.h"
#include "headers/QuantizedInt8FIFOMemory.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<TrainingSample> makeDataset(size_t count, size_t start) {
    std::vector<TrainingSample> samples;
    samples.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const double x = static_cast<double>(index + start) / 10.0;
        samples.push_back({{x}, {2.0 * x + 1.0}});
    }
    return samples;
}

NeuralNetwork makeNetwork() {
    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 8);
    network.addLayer(8, 1);
    return network;
}

std::unique_ptr<Optimizer> makeOptimizer(const std::string& name) {
    if (name == "sgd") {
        return std::make_unique<SGDOptimizer>(0.01);
    }
    if (name == "momentum") {
        return std::make_unique<MomentumOptimizer>(0.001, 0.9);
    }
    return std::make_unique<AdamOptimizer>(0.01);
}

std::unique_ptr<LossFunction> makeLoss(const std::string& name) {
    if (name == "mse") return std::make_unique<MSELoss>();
    if (name == "mae") return std::make_unique<MAELoss>();
    if (name == "huber") return std::make_unique<HuberLoss>(1.0);
    throw std::invalid_argument("Unknown benchmark loss: " + name);
}

std::unique_ptr<LearningMemory> makeMemory(const std::string& name, size_t capacity, std::uint32_t seed = 1234u) {
    if (name == "fifo") {
        return std::make_unique<FIFOMemory>(capacity);
    }
    if (name == "reservoir") {
        return std::make_unique<ReservoirMemory>(capacity, seed);
    }
    if (name == "prioritized") {
        return std::make_unique<PrioritizedMemory>(capacity, 0.6, seed);
    }
    if (name == "novelty") {
        return std::make_unique<NoveltyMemory>(capacity, 1.0);
    }
    if (name == "hybrid") {
        const HybridMemoryRatios ratios{0.25, 0.25, 0.25, 0.25};
        return std::make_unique<HybridMemory>(capacity, ratios, 1.0, seed);
    }
    throw std::invalid_argument("Unknown benchmark memory strategy: " + name);
}

size_t optimizerStateBytes(const Optimizer& optimizer) {
    if (const auto* momentum = dynamic_cast<const MomentumOptimizer*>(&optimizer)) {
        return momentum->stateBytes();
    }
    if (const auto* adam = dynamic_cast<const AdamOptimizer*>(&optimizer)) {
        return adam->stateBytes();
    }
    return 0;
}

size_t parameterBytes(const NeuralNetwork& network) {
    size_t count = 0;
    for (const DenseLayer& layer : network.layers()) {
        for (const auto& row : layer.weights()) {
            count += row.size();
        }
        count += layer.bias().size();
    }
    return count * sizeof(double);
}

size_t sampleBytes(const TrainingSample& sample) {
    return (sample.input.size() + sample.target.size()) * sizeof(double) +
        6 * sizeof(double) + 2 * sizeof(std::size_t);
}

// Nombre approximatif d'operations multiplication-accumulation (MAC) d'un
// seul passage avant du reseau : somme, sur chaque couche dense, de
// entrees * sorties. Grossier par construction (voir approximate_macs dans
// BenchmarkResult).
double macsPerForwardPass(const NeuralNetwork& network) {
    double macs = 0.0;
    for (const DenseLayer& layer : network.layers()) {
        macs += static_cast<double>(layer.weights().size()) * static_cast<double>(layer.bias().size());
    }
    return macs;
}

BenchmarkResult evaluate(
    NeuralNetwork& network,
    Optimizer& optimizer,
    const LossFunction& loss,
    const std::vector<TrainingSample>& validation,
    const std::string& memory_strategy,
    const std::string& optimizer_name,
    const std::string& loss_name,
    double training_loss,
    double training_time_ms,
    size_t samples_stored,
    size_t memory_used_bytes,
    size_t updates,
    size_t memory_capacity = 0,
    size_t samples_processed = 0
) {
    double validation_loss = 0.0;
    double mae = 0.0;
    double squared_error = 0.0;
    const auto inference_start = std::chrono::steady_clock::now();
    for (const TrainingSample& sample : validation) {
        const std::vector<double> prediction = network.forward(sample.input);
        validation_loss += loss.compute(prediction, sample.target);
        const RegressionMetrics metric = Metrics::regression(prediction, sample.target);
        mae += metric.mae;
        squared_error += metric.rmse * metric.rmse;
    }
    const auto inference_end = std::chrono::steady_clock::now();

    // Par defaut (pas de forward/backward distincts, ex. quand
    // samples_processed n'est pas fourni), on assimile le debit et le cout
    // MAC au nombre de mises a jour de poids.
    const size_t effective_samples_processed = samples_processed > 0 ? samples_processed : updates;

    const double validation_count = static_cast<double>(validation.size());
    BenchmarkResult result;
    result.memory_strategy = memory_strategy;
    result.precision = memory_strategy == "quantized_fifo_int16"
        ? "int16"
        : (memory_strategy == "quantized_fifo_int8" ? "int8" : "float64");
    result.optimizer = optimizer_name;
    result.loss_function = loss_name;
    result.training_loss = training_loss;
    result.validation_loss = validation_loss / validation_count;
    result.mae = mae / validation_count;
    result.rmse = std::sqrt(squared_error / validation_count);
    result.training_time_ms = training_time_ms;
    result.inference_time_us = std::chrono::duration<double, std::micro>(inference_end - inference_start).count() / validation_count;
    result.memory_used_bytes = memory_used_bytes + parameterBytes(network) + optimizerStateBytes(optimizer);
    result.samples_stored = samples_stored;
    result.parameter_updates = updates;
    result.memory_capacity = memory_capacity;
    // Forward + backward (gradient par rapport aux entrees et aux poids)
    // approxime a un facteur x3 les MAC du seul forward — regle empirique
    // courante, volontairement grossiere (voir BenchmarkResult).
    result.approximate_macs = macsPerForwardPass(network) * 3.0 * static_cast<double>(effective_samples_processed);
    if (training_time_ms > 0.0) {
        const double training_time_s = training_time_ms / 1000.0;
        result.samples_per_second = static_cast<double>(effective_samples_processed) / training_time_s;
        result.updates_per_second = static_cast<double>(updates) / training_time_s;
    }
    return result;
}

double datasetLoss(
    NeuralNetwork& network,
    const LossFunction& loss,
    const std::vector<TrainingSample>& dataset
) {
    double total = 0.0;
    for (const TrainingSample& sample : dataset) {
        total += loss.compute(network.forward(sample.input), sample.target);
    }
    return total / static_cast<double>(dataset.size());
}

// Valeur critique de Student (bilaterale, 95%) pour df = 2, soit
// exactement le cas a 3 seeds traite ci-dessous. Voir le commentaire sur
// mae_ci95_margin dans BenchmarkResult.
constexpr double t_critical_95_df2 = 4.302652729911275;

// Calcule mae_mean/mae_stddev/mae_ci95_margin sur un groupe de resultats
// partageant le meme scenario (capacite x strategie x optimiseur x perte x
// precision) et ne differant que par la seed, puis les reporte sur chaque
// resultat du groupe. Factorise entre le balayage float64 et les balayages
// quantifies (int16/int8) ci-dessous, qui partagent exactement cette
// logique (voir docs/roadmap.md, « Priorité moyenne : benchmark
// scientifique »).
void applyMeanStddevAndCi95(std::vector<BenchmarkResult>& seed_results) {
    double mae_sum = 0.0;
    for (const BenchmarkResult& seed_result : seed_results) {
        mae_sum += seed_result.mae;
    }
    const double mae_mean = mae_sum / static_cast<double>(seed_results.size());

    double squared_deviation_sum = 0.0;
    for (const BenchmarkResult& seed_result : seed_results) {
        const double difference = seed_result.mae - mae_mean;
        squared_deviation_sum += difference * difference;
    }
    // Ecart-type population (diviseur N) pour la statistique descriptive
    // mae_stddev ; ecart-type d'echantillon (diviseur N-1) pour
    // l'intervalle de confiance, comme l'exige la formule de Student.
    const double mae_stddev_population = std::sqrt(
        squared_deviation_sum / static_cast<double>(seed_results.size())
    );
    double mae_ci95_margin = 0.0;
    if (seed_results.size() == 3) {
        const double mae_stddev_sample = std::sqrt(squared_deviation_sum / 2.0);
        mae_ci95_margin = t_critical_95_df2 * mae_stddev_sample / std::sqrt(3.0);
    }

    for (BenchmarkResult& seed_result : seed_results) {
        seed_result.mae_mean = mae_mean;
        seed_result.mae_stddev = mae_stddev_population;
        seed_result.mae_ci95_margin = mae_ci95_margin;
    }
}
}

int main() {
    DenseLayer::seedWeightInitialization(1234);
    const std::vector<TrainingSample> training = makeDataset(80, 0);
    const std::vector<TrainingSample> validation = makeDataset(20, 80);
    const std::vector<std::string> optimizers = {"sgd", "momentum", "adam"};
    const std::vector<std::string> losses = {"mse", "mae", "huber"};

    std::vector<BenchmarkResult> baseline_results;
    std::vector<BenchmarkResult> full_dataset_results;
    std::vector<BenchmarkResult> memory_capacity_results;
    std::vector<BenchmarkResult> quantization_results;
    std::vector<BenchmarkResult> forgetting_results;

    // Baseline naive : predire pour toute la validation la cible du dernier
    // echantillon d'entrainement connu, sans aucun apprentissage. Sert de
    // reference minimale — tout modele entraine doit au moins faire mieux.
    // Evaluee avec chaque perte, pour rester comparable aux scenarios qui
    // varient la perte plus bas.
    for (const std::string& loss_name : losses) {
        const std::unique_ptr<LossFunction> loss = makeLoss(loss_name);
        const std::vector<double>& last_known_target = training.back().target;
        double validation_loss = 0.0;
        double mae = 0.0;
        double squared_error = 0.0;
        for (const TrainingSample& sample : validation) {
            validation_loss += loss->compute(last_known_target, sample.target);
            const RegressionMetrics metric = Metrics::regression(last_known_target, sample.target);
            mae += metric.mae;
            squared_error += metric.rmse * metric.rmse;
        }
        const double validation_count = static_cast<double>(validation.size());

        BenchmarkResult result;
        result.memory_strategy = "baseline_last_value";
        result.precision = "float64";
        result.optimizer = "none";
        result.loss_function = loss_name;
        result.validation_loss = validation_loss / validation_count;
        result.mae = mae / validation_count;
        result.rmse = std::sqrt(squared_error / validation_count);
        baseline_results.push_back(result);
    }

    // MAE du dataset complet par (optimiseur, perte), pour calculer plus
    // bas le ratio mae_memoire_bornee / mae_dataset_complet de chaque
    // scenario partageant la meme perte.
    std::map<std::string, double> full_dataset_mae;
    const auto fullDatasetKey = [](const std::string& optimizer_name, const std::string& loss_name) {
        return optimizer_name + "|" + loss_name;
    };
    for (const std::string& optimizer_name : optimizers) {
        for (const std::string& loss_name : losses) {
            NeuralNetwork network = makeNetwork();
            const std::unique_ptr<LossFunction> loss = makeLoss(loss_name);
            std::unique_ptr<Optimizer> optimizer = makeOptimizer(optimizer_name);
            LearningEngine engine(network, *loss, *optimizer);

            const auto start = std::chrono::steady_clock::now();
            const double training_loss = engine.train(training, 100, 8);
            const auto end = std::chrono::steady_clock::now();

            BenchmarkResult result = evaluate(
                network, *optimizer, *loss, validation, "full_dataset", optimizer_name, loss_name,
                training_loss, std::chrono::duration<double, std::milli>(end - start).count(),
                training.size(), training.empty() ? 0 : training.size() * sampleBytes(training.front()),
                100 * ((training.size() + 7) / 8), training.size(), 100 * training.size()
            );
            full_dataset_mae[fullDatasetKey(optimizer_name, loss_name)] = result.mae;
            full_dataset_results.push_back(result);
        }
    }

    const auto applyFullDatasetRatio = [&](BenchmarkResult& result, const std::string& optimizer_name, const std::string& loss_name) {
        const auto full_dataset_it = full_dataset_mae.find(fullDatasetKey(optimizer_name, loss_name));
        if (full_dataset_it != full_dataset_mae.end() && full_dataset_it->second > 0.0) {
            result.mae_ratio_to_full_dataset = result.mae / full_dataset_it->second;
        }
    };

    // Seeds repetees pour chaque scenario borne : la seed pilote a la fois
    // l'initialisation des poids (DenseLayer::seedWeightInitialization) et
    // le generateur de la memoire d'apprentissage (Reservoir, Prioritized,
    // Hybrid), pour caracteriser la variance reelle du scenario plutot
    // qu'un unique tirage.
    const std::vector<std::uint32_t> seeds = {1234u, 2345u, 3456u};

    // Capacites de memoire bornee comparees au dataset complet (voir
    // docs/benchmark.md, section « Baseline obligatoire »).
    for (const size_t memory_capacity : {32u, 64u, 128u, 256u}) {
        for (const std::string memory_name : {"fifo", "reservoir", "prioritized", "novelty", "hybrid"}) {
            for (const std::string& optimizer_name : optimizers) {
                for (const std::string& loss_name : losses) {
                    std::vector<BenchmarkResult> seed_results;
                    seed_results.reserve(seeds.size());

                    for (const std::uint32_t seed : seeds) {
                        DenseLayer::seedWeightInitialization(seed);
                        NeuralNetwork network = makeNetwork();
                        const std::unique_ptr<LossFunction> loss = makeLoss(loss_name);
                        std::unique_ptr<Optimizer> optimizer = makeOptimizer(optimizer_name);
                        std::unique_ptr<LearningMemory> memory = makeMemory(memory_name, memory_capacity, seed);
                        LearningEngine engine(network, *loss, *optimizer);

                        double training_loss = 0.0;
                        const auto start = std::chrono::steady_clock::now();
                        for (const TrainingSample& sample : training) {
                            training_loss += engine.learn(*memory, sample, 8);
                        }
                        const auto end = std::chrono::steady_clock::now();

                        BenchmarkResult result = evaluate(
                            network, *optimizer, *loss, validation, memory_name, optimizer_name, loss_name,
                            training_loss / static_cast<double>(training.size()),
                            std::chrono::duration<double, std::milli>(end - start).count(),
                            memory->size(), memory->size() * sampleBytes(training.front()),
                            training.size(), memory_capacity, training.size()
                        );
                        result.seed = seed;
                        applyFullDatasetRatio(result, optimizer_name, loss_name);
                        seed_results.push_back(result);
                    }

                    applyMeanStddevAndCi95(seed_results);
                    for (const BenchmarkResult& seed_result : seed_results) {
                        memory_capacity_results.push_back(seed_result);
                    }
                }
            }
        }
    }

    // Balayage de capacites ET de seeds pour les precisions quantifiees
    // (int16, int8), au meme titre que le balayage float64 ci-dessus — fait
    // pour combler l'angle mort documente dans docs/roadmap.md, « Priorité
    // moyenne : benchmark scientifique ». Les memoires quantifiees sont
    // deterministes (pas de tirage aleatoire interne, comme FIFO) : la seed
    // ne fait donc varier ici que l'initialisation des poids du reseau,
    // exactement comme "fifo" dans le balayage float64 ci-dessus.
    const TrainingSampleQuantizer int16_quantizer =
        TrainingSampleQuantizer::calibrate(training);
    const Int8TrainingSampleQuantizer int8_quantizer =
        Int8TrainingSampleQuantizer::calibrate(training);
    for (const size_t memory_capacity : {32u, 64u, 128u, 256u}) {
        for (const std::string& optimizer_name : optimizers) {
            for (const std::string& loss_name : losses) {
                std::vector<BenchmarkResult> int16_seed_results;
                std::vector<BenchmarkResult> int8_seed_results;
                int16_seed_results.reserve(seeds.size());
                int8_seed_results.reserve(seeds.size());

                for (const std::uint32_t seed : seeds) {
                    DenseLayer::seedWeightInitialization(seed);
                    NeuralNetwork int16_network = makeNetwork();
                    const std::unique_ptr<LossFunction> int16_loss = makeLoss(loss_name);
                    std::unique_ptr<Optimizer> int16_optimizer = makeOptimizer(optimizer_name);
                    QuantizedFIFOMemory int16_memory(memory_capacity, int16_quantizer);
                    LearningEngine int16_engine(int16_network, *int16_loss, *int16_optimizer);

                    double int16_training_loss = 0.0;
                    const auto int16_start = std::chrono::steady_clock::now();
                    for (const TrainingSample& sample : training) {
                        int16_training_loss += int16_engine.learn(int16_memory, sample, 8);
                    }
                    const auto int16_end = std::chrono::steady_clock::now();
                    BenchmarkResult int16_result = evaluate(
                        int16_network, *int16_optimizer, *int16_loss, validation,
                        "quantized_fifo_int16", optimizer_name, loss_name,
                        int16_training_loss / static_cast<double>(training.size()),
                        std::chrono::duration<double, std::milli>(int16_end - int16_start).count(),
                        int16_memory.size(), int16_memory.memoryUsedBytes(), training.size(),
                        memory_capacity, training.size()
                    );
                    int16_result.seed = seed;
                    applyFullDatasetRatio(int16_result, optimizer_name, loss_name);
                    int16_seed_results.push_back(int16_result);

                    // Meme seed reutilisee pour int8, comme le balayage
                    // float64 reutilise chaque seed pour toutes les
                    // strategies de memoire a capacite egale.
                    DenseLayer::seedWeightInitialization(seed);
                    NeuralNetwork int8_network = makeNetwork();
                    const std::unique_ptr<LossFunction> int8_loss = makeLoss(loss_name);
                    std::unique_ptr<Optimizer> int8_optimizer = makeOptimizer(optimizer_name);
                    QuantizedInt8FIFOMemory int8_memory(memory_capacity, int8_quantizer);
                    LearningEngine int8_engine(int8_network, *int8_loss, *int8_optimizer);

                    double int8_training_loss = 0.0;
                    const auto int8_start = std::chrono::steady_clock::now();
                    for (const TrainingSample& sample : training) {
                        int8_training_loss += int8_engine.learn(int8_memory, sample, 8);
                    }
                    const auto int8_end = std::chrono::steady_clock::now();
                    BenchmarkResult int8_result = evaluate(
                        int8_network, *int8_optimizer, *int8_loss, validation,
                        "quantized_fifo_int8", optimizer_name, loss_name,
                        int8_training_loss / static_cast<double>(training.size()),
                        std::chrono::duration<double, std::milli>(int8_end - int8_start).count(),
                        int8_memory.size(), int8_memory.memoryUsedBytes(), training.size(),
                        memory_capacity, training.size()
                    );
                    int8_result.seed = seed;
                    applyFullDatasetRatio(int8_result, optimizer_name, loss_name);
                    int8_seed_results.push_back(int8_result);
                }

                applyMeanStddevAndCi95(int16_seed_results);
                applyMeanStddevAndCi95(int8_seed_results);
                for (const BenchmarkResult& seed_result : int16_seed_results) {
                    quantization_results.push_back(seed_result);
                }
                for (const BenchmarkResult& seed_result : int8_seed_results) {
                    quantization_results.push_back(seed_result);
                }
            }
        }
    }

    // Point de controle explicite avant l'experience de forgetting
    // ci-dessous : stabilise l'etat du generateur partage
    // (DenseLayer::seedWeightInitialization) independamment de tout ce qui
    // a pu changer au-dessus (voir docs/roadmap.md, « RNG partage » /
    // docs/examples.md pour l'impact sur les valeurs documentees).
    DenseLayer::seedWeightInitialization(1234);

    const std::vector<TrainingSample> regime_a = makeDataset(40, 0);
    std::vector<TrainingSample> regime_b;
    regime_b.reserve(40);
    for (size_t index = 0; index < 40; ++index) {
        const double x = static_cast<double>(index + 40) / 10.0;
        regime_b.push_back({{x}, {-2.0 * x + 20.0}});
    }

    MSELoss forgetting_loss;
    {
        NeuralNetwork network = makeNetwork();
        SGDOptimizer optimizer(0.01);
        LearningEngine engine(network, forgetting_loss, optimizer);
        engine.train(regime_a, 80, 8);
        const double before_b = datasetLoss(network, forgetting_loss, regime_a);
        engine.train(regime_b, 80, 8);
        const double after_b = datasetLoss(network, forgetting_loss, regime_a);

        BenchmarkResult result;
        result.memory_strategy = "forgetting_no_replay";
        result.precision = "float64";
        result.optimizer = "sgd";
        result.loss_function = "mse";
        result.validation_loss = after_b;
        result.forgetting = Metrics::forgetting(before_b, after_b);
        result.samples_stored = regime_b.size();
        result.parameter_updates = 160 * ((regime_a.size() + 7) / 8);
        result.memory_used_bytes = parameterBytes(network);
        forgetting_results.push_back(result);
    }

    {
        NeuralNetwork network = makeNetwork();
        SGDOptimizer optimizer(0.01);
        LearningEngine engine(network, forgetting_loss, optimizer);
        FIFOMemory memory(regime_a.size());
        engine.train(regime_a, 80, 8);
        const double before_b = datasetLoss(network, forgetting_loss, regime_a);
        for (const TrainingSample& sample : regime_a) {
            memory.add(sample);
        }
        engine.train(regime_b, 80, 8);
        const double after_b = datasetLoss(network, forgetting_loss, regime_a);
        for (size_t update = 0; update < 20; ++update) {
            engine.trainFromMemory(memory, 8);
        }
        const double after_replay = datasetLoss(network, forgetting_loss, regime_a);

        BenchmarkResult result;
        result.memory_strategy = "forgetting_fifo_replay";
        result.precision = "float64";
        result.optimizer = "sgd";
        result.loss_function = "mse";
        result.validation_loss = after_replay;
        result.forgetting = Metrics::forgetting(before_b, after_replay);
        result.samples_stored = memory.size();
        result.parameter_updates = 160 * ((regime_a.size() + 7) / 8) + 20;
        result.memory_used_bytes = memory.size() * sampleBytes(regime_a.front()) + parameterBytes(network);
        forgetting_results.push_back(result);
        (void)after_b;
    }

    // Un fichier CSV par experience, plus un fichier combine pour garder la
    // compatibilite avec les usages existants de benchmark_results.csv
    // (voir docs/benchmark.md et docs/examples.md).
    BenchmarkCsv::write("benchmark_baseline.csv", baseline_results);
    BenchmarkCsv::write("benchmark_full_dataset.csv", full_dataset_results);
    BenchmarkCsv::write("benchmark_memory_capacity.csv", memory_capacity_results);
    BenchmarkCsv::write("benchmark_quantization.csv", quantization_results);
    BenchmarkCsv::write("benchmark_forgetting.csv", forgetting_results);

    std::vector<BenchmarkResult> all_results;
    all_results.reserve(
        baseline_results.size() + full_dataset_results.size() + memory_capacity_results.size() +
        quantization_results.size() + forgetting_results.size()
    );
    all_results.insert(all_results.end(), baseline_results.begin(), baseline_results.end());
    all_results.insert(all_results.end(), full_dataset_results.begin(), full_dataset_results.end());
    all_results.insert(all_results.end(), memory_capacity_results.begin(), memory_capacity_results.end());
    all_results.insert(all_results.end(), quantization_results.begin(), quantization_results.end());
    all_results.insert(all_results.end(), forgetting_results.begin(), forgetting_results.end());
    BenchmarkCsv::write("benchmark_results.csv", all_results);

    return 0;
}
