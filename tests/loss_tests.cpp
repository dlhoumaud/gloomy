#include "LossFunction.h"
#include "DenseLayer.h"
#include "LearningEngine.h"
#include "Optimizer.h"
#include "FIFOMemory.h"
#include "ReservoirMemory.h"
#include "PrioritizedMemory.h"
#include "NoveltyMemory.h"
#include "HybridMemory.h"
#include "ImportanceScorer.h"
#include "TrainingScheduler.h"
#include "Normalization.h"
#include "Quantization.h"
#include "TrainingSampleQuantization.h"
#include "QuantizedFIFOMemory.h"
#include "TrainingSampleSerialization.h"
#include "Metrics.h"
#include "Benchmark.h"
#include "Int8Quantization.h"
#include "Int8TrainingSampleQuantization.h"
#include "QuantizedInt8FIFOMemory.h"
#include "NetworkSerialization.h"
#include "NormalizationSerialization.h"
#include "MomentumOptimizer.h"
#include "AdamOptimizer.h"
#include "OptimizerSerialization.h"
#include "LearningMemorySerialization.h"
#include "GloomyConfig.h"
#include "GloomyConfigFile.h"
#include "OnlineLearningRuntime.h"
#include <cstdio>
#include <fstream>
#include <iterator>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
void assertClose(double actual, double expected) {
    assert(std::abs(actual - expected) < 1e-12);
}

void testMSECompute() {
    MSELoss loss;
    assertClose(loss.compute({1.0, 3.0}, {0.0, 1.0}), 2.5);
}

void testMSEGradient() {
    MSELoss loss;
    const std::vector<double> gradient = loss.gradient({1.0, 3.0}, {0.0, 1.0});
    assert(gradient.size() == 2);
    assertClose(gradient[0], 1.0);
    assertClose(gradient[1], 2.0);
}

void testAdditionalLosses() {
    MAELoss mae;
    assertClose(mae.compute({1.0, 5.0}, {0.0, 1.0}), 2.5);
    const std::vector<double> mae_gradient = mae.gradient({1.0, 5.0}, {0.0, 1.0});
    assertClose(mae_gradient[0], 0.5);
    assertClose(mae_gradient[1], 0.5);

    HuberLoss huber(1.0);
    assertClose(huber.compute({0.5, 3.0}, {0.0, 0.0}), 1.3125);
    const std::vector<double> huber_gradient = huber.gradient({0.5, 3.0}, {0.0, 0.0});
    assertClose(huber_gradient[0], 0.25);
    assertClose(huber_gradient[1], 0.5);
}

void testInvalidSizes() {
    MSELoss loss;
    bool threw = false;
    try {
        loss.compute({1.0}, {1.0, 2.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testEmptyInputs() {
    MSELoss loss;
    bool threw = false;
    try {
        loss.gradient({}, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testDenseLayerGradient() {
    DenseLayer layer(2, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.3;
    layer.weights()[1][0] = -0.4;
    layer.bias()[0] = 0.1;

    const std::vector<double> input = {0.7, -1.2};
    const std::vector<double> target = {0.2};
    MSELoss loss;
    const double epsilon = 1e-6;

    const auto evaluate = [&]() {
        return loss.compute(layer.forward(input), target);
    };

    layer.forward(input);
    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));

    for (size_t input_index = 0; input_index < input.size(); ++input_index) {
        const double original = layer.weights()[input_index][0];
        layer.weights()[input_index][0] = original + epsilon;
        const double loss_plus = evaluate();
        layer.weights()[input_index][0] = original - epsilon;
        const double loss_minus = evaluate();
        layer.weights()[input_index][0] = original;

        const double numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
        assert(std::abs(numerical - layer.weightGradients()[input_index][0]) < 1e-7);
    }

    const double original_bias = layer.bias()[0];
    layer.bias()[0] = original_bias + epsilon;
    const double loss_plus = evaluate();
    layer.bias()[0] = original_bias - epsilon;
    const double loss_minus = evaluate();
    layer.bias()[0] = original_bias;

    const double numerical_bias = (loss_plus - loss_minus) / (2.0 * epsilon);
    assert(std::abs(numerical_bias - layer.biasGradients()[0]) < 1e-7);
}

void testDenseLayerSeededInitialization() {
    DenseLayer::seedWeightInitialization(42u);
    const DenseLayer first(3, 2);
    DenseLayer::seedWeightInitialization(42u);
    const DenseLayer second(3, 2);
    assert(first.weights() == second.weights());
    assert(first.bias() == second.bias());

    DenseLayer::seedWeightInitialization(43u);
    const DenseLayer third(3, 2);
    assert(first.weights() != third.weights());

    for (const auto& row : first.weights()) {
        for (double weight : row) {
            assert(weight >= -0.5 && weight <= 0.5);
        }
    }
}

// Verifie par difference centree que les gradients analytiques de `network`
// (poids et biais de chaque couche) correspondent aux gradients numeriques
// de `loss` sur (`input`, `target`). Couvre tout le reseau, y compris un
// eventuel post_algorithm (softmax) applique sur la derniere couche.
void checkNetworkGradient(
    NeuralNetwork& network,
    const LossFunction& loss,
    const std::vector<double>& input,
    const std::vector<double>& target
) {
    const double epsilon = 1e-6;
    const double tolerance = 1e-6;

    network.zeroGradients();
    network.backward(loss.gradient(network.forward(input), target));

    for (DenseLayer& layer : network.layers()) {
        auto& weights = layer.weights();
        const auto& weight_gradients = layer.weightGradients();
        for (size_t input_index = 0; input_index < weights.size(); ++input_index) {
            for (size_t output_index = 0; output_index < weights[input_index].size(); ++output_index) {
                const double original = weights[input_index][output_index];
                weights[input_index][output_index] = original + epsilon;
                const double loss_plus = loss.compute(network.forward(input), target);
                weights[input_index][output_index] = original - epsilon;
                const double loss_minus = loss.compute(network.forward(input), target);
                weights[input_index][output_index] = original;

                const double numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
                assert(std::abs(numerical - weight_gradients[input_index][output_index]) < tolerance);
            }
        }

        auto& biases = layer.bias();
        const auto& bias_gradients = layer.biasGradients();
        for (size_t output_index = 0; output_index < biases.size(); ++output_index) {
            const double original = biases[output_index];
            biases[output_index] = original + epsilon;
            const double loss_plus = loss.compute(network.forward(input), target);
            biases[output_index] = original - epsilon;
            const double loss_minus = loss.compute(network.forward(input), target);
            biases[output_index] = original;

            const double numerical = (loss_plus - loss_minus) / (2.0 * epsilon);
            assert(std::abs(numerical - bias_gradients[output_index]) < tolerance);
        }
    }
}

void testGradientCheckingAllActivations() {
    // sigmoid_derivative/tanh_derivative sont volontairement exclues :
    // DenseLayer::backward les rejette explicitement comme activations
    // d'entrainement (voir docs/configurations.md).
    const std::vector<std::string> activations = {"none", "sigmoid", "relu", "leaky_relu", "tanh"};
    const std::vector<double> input = {0.6, -0.9, 0.3};
    const std::vector<double> target = {0.4};
    MSELoss loss;

    for (const std::string& activation : activations) {
        DenseLayer::seedWeightInitialization(7u);
        NeuralNetwork network;
        network.algorithm = activation;
        network.addLayer(3, 4);
        network.addLayer(4, 4);
        network.addLayer(4, 1);
        checkNetworkGradient(network, loss, input, target);
    }
}

void testSoftmaxGradientCheck() {
    // Sortie a 3 neurones avec softmax : exerce le terme croise du gradient
    // softmax dans DenseLayer::backward (jamais couvert par les tests a une
    // seule sortie).
    DenseLayer::seedWeightInitialization(11u);
    NeuralNetwork network;
    network.algorithm = "none";
    network.post_algorithm = "softmax";
    network.addLayer(3, 4);
    network.addLayer(4, 3);

    const std::vector<double> input = {0.5, -0.2, 0.8};
    const std::vector<double> target = {1.0, 0.0, 0.0};
    MSELoss loss;
    checkNetworkGradient(network, loss, input, target);
}

void testActivationStabilityWithLargeValues() {
    const std::vector<std::string> activations = {"sigmoid", "tanh", "relu", "leaky_relu", "none"};
    const std::vector<double> large_input = {1e8, -1e8};
    MSELoss loss;
    const std::vector<double> target = {0.5};

    for (const std::string& activation : activations) {
        DenseLayer::seedWeightInitialization(13u);
        NeuralNetwork network;
        network.algorithm = activation;
        network.addLayer(2, 3);
        network.addLayer(3, 1);

        const std::vector<double> output = network.forward(large_input);
        for (double value : output) {
            assert(std::isfinite(value));
        }

        network.zeroGradients();
        network.backward(loss.gradient(output, target));
        for (const DenseLayer& layer : network.layers()) {
            for (const auto& row : layer.weightGradients()) {
                for (double gradient : row) {
                    assert(std::isfinite(gradient));
                }
            }
            for (double gradient : layer.biasGradients()) {
                assert(std::isfinite(gradient));
            }
        }
    }
}

void testNonFiniteValuesRejected() {
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    const double inf_value = std::numeric_limits<double>::infinity();

    MSELoss loss;
    bool threw = false;
    try {
        loss.compute({nan_value}, {0.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    threw = false;
    try {
        loss.gradient({0.0}, {inf_value});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    DenseLayer layer(2, 1);
    threw = false;
    try {
        layer.forward({nan_value, 0.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    layer.forward({0.0, 0.0});
    layer.zeroGradients();
    threw = false;
    try {
        layer.backward({inf_value});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testSGDUpdateReducesLoss() {
    DenseLayer layer(1, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.0;
    layer.bias()[0] = 0.0;

    const std::vector<double> input = {1.0};
    const std::vector<double> target = {2.0};
    MSELoss loss;
    const double lossBefore = loss.compute(layer.forward(input), target);

    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));
    std::vector<DenseLayer> layers;
    layers.push_back(layer);

    SGDOptimizer optimizer(0.1);
    optimizer.update(layers);

    const double lossAfter = loss.compute(layers[0].forward(input), target);
    assert(lossAfter < lossBefore);
}

void testLearningEngineBatchTraining() {
    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 1);
    network.layers()[0].weights()[0][0] = 0.0;
    network.layers()[0].bias()[0] = 0.0;

    MSELoss loss;
    SGDOptimizer optimizer(0.05);
    LearningEngine engine(network, loss, optimizer);
    const std::vector<TrainingSample> samples = {
        {{1.0}, {2.0}},
        {{2.0}, {4.0}}
    };

    const double initial_loss = loss.compute(network.forward({1.0}), {2.0});
    const double final_loss = engine.train(samples, 40, 2);
    assert(final_loss < initial_loss);
    assert(loss.compute(network.forward({1.0}), {2.0}) < initial_loss);
}

void testFIFOMemoryCapacity() {
    FIFOMemory memory(2);
    memory.add({{1.0}, {2.0}});
    memory.add({{2.0}, {4.0}});
    memory.add({{3.0}, {6.0}});

    assert(memory.capacity() == 2);
    assert(memory.size() == 2);
    const std::vector<TrainingSample> samples = memory.sample(2);
    assert(samples.size() == 2);
    assert(samples[0].input[0] == 2.0);
    assert(samples[1].input[0] == 3.0);

    memory.remove(0);
    assert(memory.size() == 1);
    memory.clear();
    assert(memory.size() == 0);
}

void testMemoryMetadata() {
    FIFOMemory memory(2);
    memory.add({{1.0}, {2.0}});
    memory.add({{2.0}, {4.0}});
    memory.advanceAges();
    const std::vector<TrainingSample> first_batch = memory.sample(1);
    assert(first_batch[0].age == 1);
    assert(first_batch[0].usage_count == 1);

    memory.advanceAges();
    const std::vector<TrainingSample> second_batch = memory.sample(1);
    assert(second_batch[0].age == 2);
    assert(second_batch[0].usage_count == 2);
}

void testLearningEngineMemoryReplay() {
    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 1);
    network.layers()[0].weights()[0][0] = 0.0;
    network.layers()[0].bias()[0] = 0.0;

    MSELoss loss;
    SGDOptimizer optimizer(0.05);
    LearningEngine engine(network, loss, optimizer);
    FIFOMemory memory(2);
    memory.add({{1.0}, {2.0}});
    memory.add({{2.0}, {4.0}});

    const double replay_loss = engine.trainFromMemory(memory, 2);
    assert(replay_loss > 0.0);

    engine.learn(memory, {{3.0}, {6.0}}, 2);
    const std::vector<TrainingSample> samples = memory.sample(2);
    assert(samples.size() == 2);
    assert(samples[0].input[0] == 2.0);
    assert(samples[1].input[0] == 3.0);
}

void testReservoirMemory() {
    ReservoirMemory first(3, 1234u);
    ReservoirMemory second(3, 1234u);
    for (double value = 1.0; value <= 100.0; ++value) {
        const TrainingSample sample = {{value}, {value * 2.0}};
        first.add(sample);
        second.add(sample);
    }

    assert(first.capacity() == 3);
    assert(first.size() == 3);
    const std::vector<TrainingSample> first_samples = first.sample(3);
    const std::vector<TrainingSample> second_samples = second.sample(3);
    for (size_t index = 0; index < first_samples.size(); ++index) {
        assert(first_samples[index].input[0] == second_samples[index].input[0]);
        assert(first_samples[index].input[0] >= 1.0);
        assert(first_samples[index].input[0] <= 100.0);
    }

    first.clear();
    assert(first.size() == 0);
}

void testPrioritizedMemory() {
    PrioritizedMemory memory(2, 0.6, 1234u);
    memory.add({{1.0}, {1.0}, 0.1});
    memory.add({{2.0}, {2.0}, 0.2});
    memory.add({{3.0}, {3.0}, 10.0});

    assert(memory.capacity() == 2);
    assert(memory.size() == 2);
    const std::vector<TrainingSample> samples = memory.sample(2);
    bool found_high_priority = false;
    for (const TrainingSample& sample : samples) {
        if (sample.input[0] == 3.0) {
            found_high_priority = true;
        }
    }
    assert(found_high_priority);

    const std::vector<MemoryEntry> entries = memory.sampleIndexed(1);
    assert(entries.size() == 1);
    TrainingSample updated = entries[0].sample;
    updated.priority = 42.0;
    memory.update(entries[0].index, updated);
    const std::vector<MemoryEntry> updated_entries = memory.sampleIndexed(2);
    bool found_updated_priority = false;
    for (const MemoryEntry& entry : updated_entries) {
        if (entry.sample.priority == 42.0) {
            found_updated_priority = true;
        }
    }
    assert(found_updated_priority);
}

void testPrioritizedMemoryBiasCorrection() {
    // beta = 0 desactive completement la correction : tous les poids
    // restent a 1.0, quelles que soient les priorites.
    {
        PrioritizedMemory memory(3, 0.6, 1234u, 0.0);
        memory.add({{1.0}, {1.0}, 0.1});
        memory.add({{2.0}, {2.0}, 5.0});
        memory.add({{3.0}, {3.0}, 20.0});
        assertClose(memory.beta(), 0.0);
        const std::vector<MemoryEntry> entries = memory.sampleIndexed(3);
        assert(entries.size() == 3);
        for (const MemoryEntry& entry : entries) {
            assertClose(entry.importance_weight, 1.0);
        }
    }

    // beta > 0 sous-pondere l'echantillon a haute priorite (sur-represente
    // par le tirage) et donne a l'echantillon a faible priorite (rare) le
    // poids maximal 1.0, apres normalisation par le maximum du batch.
    {
        PrioritizedMemory memory(2, 1.0, 1234u, 1.0);
        memory.add({{1.0}, {1.0}, 1.0});
        memory.add({{2.0}, {2.0}, 9.0});
        assertClose(memory.beta(), 1.0);
        const std::vector<MemoryEntry> entries = memory.sampleIndexed(2);
        assert(entries.size() == 2);

        double low_priority_weight = -1.0;
        double high_priority_weight = -1.0;
        for (const MemoryEntry& entry : entries) {
            assert(entry.importance_weight > 0.0 && entry.importance_weight <= 1.0 + 1e-12);
            if (entry.sample.input[0] == 1.0) {
                low_priority_weight = entry.importance_weight;
            } else {
                high_priority_weight = entry.importance_weight;
            }
        }
        assert(low_priority_weight > 0.0 && high_priority_weight > 0.0);
        assertClose(low_priority_weight, 1.0);
        assert(std::abs(high_priority_weight - 1.0 / 9.0) < 1e-9);
        assert(low_priority_weight > high_priority_weight);
    }
}

void testLearningEngineTrainBatchWeighting() {
    const auto run = [](double beta) {
        NeuralNetwork network;
        network.algorithm = "none";
        network.addLayer(1, 1);
        network.layers()[0].weights()[0][0] = 0.0;
        network.layers()[0].bias()[0] = 0.0;
        MSELoss loss;
        SGDOptimizer optimizer(0.1);
        LearningEngine engine(network, loss, optimizer);

        // Deux echantillons aux gradients opposes (target=10 vs target=-10)
        // et aux priorites opposees (0.1 vs 0.9) ; capacite == batch_size
        // == 2 donc les deux sont toujours selectionnes, quel que soit le
        // tirage aleatoire.
        PrioritizedMemory memory(2, 1.0, 1234u, beta);
        memory.add({{1.0}, {10.0}, 0.1});
        memory.add({{1.0}, {-10.0}, 0.9});
        engine.trainFromMemory(memory, 2);
        return network.layers()[0].weights()[0][0];
    };

    // Sans correction (beta=0), les deux echantillons pesent pareil et
    // leurs gradients opposes s'annulent (quasiment exactement).
    assert(std::abs(run(0.0)) < 1e-9);

    // Avec correction (beta=1), l'echantillon rare et sous-pondere par le
    // tirage (target=10, faible priorite) domine desormais la mise a jour :
    // le poids doit nettement bouger vers le positif.
    assert(run(1.0) > 0.5);
}

void testNoveltyMemory() {
    NoveltyMemory memory(2, 0.25);
    memory.add({{0.0, 0.0}, {0.0}});
    memory.add({{0.1, 0.1}, {0.0}});
    assert(memory.size() == 1);

    memory.add({{1.0, 0.0}, {1.0}});
    assert(memory.size() == 2);
    memory.add({{2.0, 0.0}, {2.0}});
    assert(memory.size() == 2);

    const std::vector<TrainingSample> samples = memory.sample(2);
    assert(samples[0].input[0] == 2.0 || samples[1].input[0] == 2.0);

    bool threw = false;
    try {
        memory.add({{1.0}, {1.0}});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testHybridMemory() {
    HybridMemoryRatios ratios;
    HybridMemory memory(8, ratios, 1.0, 1234u);
    for (double value = 0.0; value < 20.0; value += 1.0) {
        memory.add({{value}, {value}, 0.0});
    }

    assert(memory.capacity() == 8);
    assert(memory.size() <= memory.capacity());
    const std::vector<size_t> sizes = memory.partitionSizes();
    size_t partition_total = 0;
    for (size_t size : sizes) {
        partition_total += size;
    }
    assert(partition_total == memory.size());
    assert(sizes.size() == 4);

    const std::vector<TrainingSample> batch = memory.sample(3);
    assert(batch.size() == 3);
}

void testHybridMemoryTrueRecency() {
    // recent_capacity=2, historical_capacity=2, error/novelty desactives.
    HybridMemoryRatios ratios{0.5, 0.0, 0.0, 0.5};
    HybridMemory memory(4, ratios, 0.0, 1234u);

    // Aucune alternance a l'admission : les deux premieres observations
    // generiques remplissent toutes les deux Recent (l'ancienne logique par
    // parite `seen_samples % 2` en aurait envoye une sur deux ailleurs).
    memory.add({{1.0}, {1.0}, 0.0});
    memory.advanceAges();  // seul {1.0} vieillit d'un cran ici.
    memory.add({{2.0}, {2.0}, 0.0});

    std::vector<size_t> sizes = memory.partitionSizes();
    assert(sizes[0] == 2);  // Recent plein
    assert(sizes[3] == 0);  // Historical encore vide

    // Recent est plein : la 3e observation generique evince le membre le
    // plus ancien par age reel ({1.0}, age=1), pas {2.0} (age=0) ni un
    // choix arbitraire de position dans le vecteur.
    memory.add({{3.0}, {3.0}, 0.0});

    sizes = memory.partitionSizes();
    assert(sizes[0] == 2);  // Recent toujours plein (une entree, une sortie)
    assert(sizes[3] == 1);  // Historical a recu le membre evince, pas perdu

    bool found_oldest_still_stored = false;
    for (const TrainingSample& sample : memory.sample(4)) {
        if (sample.input[0] == 1.0) {
            found_oldest_still_stored = true;
        }
    }
    assert(found_oldest_still_stored);
    assert(memory.size() == 3);
}

void testImportanceScorer() {
    ImportanceScorer error_scorer;
    assert(std::abs(error_scorer.score({0.5}) - 0.5) < 1e-12);

    ImportanceWeights weights;
    weights.error = 0.0;
    weights.novelty = 1.0;
    ImportanceScorer novelty_scorer(weights);
    assert(std::abs(novelty_scorer.score({0.0, 0.8}) - 0.8) < 1e-12);

    bool threw = false;
    try {
        ImportanceWeights invalid;
        invalid.error = -1.0;
        ImportanceScorer invalid_scorer(invalid);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testTrainingSchedulers() {
    EverySampleScheduler every_sample;
    assert(every_sample.shouldTrain(0.0));

    EveryNScheduler every_two(2);
    assert(!every_two.shouldTrain(1.0));
    assert(every_two.shouldTrain(1.0));
    assert(!every_two.shouldTrain(1.0));

    OnHighErrorScheduler high_error(0.5);
    assert(!high_error.shouldTrain(0.49));
    assert(high_error.shouldTrain(0.5));
}

void testStreamingNormalization() {
    StreamingNormalizer normalizer(2);
    normalizer.update({1.0, 10.0});
    normalizer.update({3.0, 14.0});
    normalizer.update({5.0, 18.0});

    assert(normalizer.count() == 3);
    assert(std::abs(normalizer.mean()[0] - 3.0) < 1e-12);
    assert(std::abs(normalizer.mean()[1] - 14.0) < 1e-12);
    assert(std::abs(normalizer.variance()[0] - (8.0 / 3.0)) < 1e-12);
    assert(normalizer.minimum()[0] == 1.0);
    assert(normalizer.maximum()[1] == 18.0);

    const std::vector<double> normalized = normalizer.normalize({3.0, 14.0});
    assert(std::abs(normalized[0]) < 1e-12);
    assert(std::abs(normalized[1]) < 1e-12);

    bool threw = false;
    try {
        normalizer.update({1.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testNormalizationSerialization() {
    const std::string path = "/tmp/gloomy_normalizer.bin";
    StreamingNormalizer original(2);
    original.update({1.0, 10.0});
    original.update({3.0, 14.0});
    original.update({5.0, 18.0});
    const std::vector<double> expected = original.normalize({4.0, 16.0});
    NormalizationSerialization::save(path, original);

    StreamingNormalizer restored(2);
    NormalizationSerialization::load(path, restored);
    const std::vector<double> actual = restored.normalize({4.0, 16.0});
    assert(restored.count() == original.count());
    assert(std::abs(actual[0] - expected[0]) < 1e-12);
    assert(std::abs(actual[1] - expected[1]) < 1e-12);
    std::remove(path.c_str());
}

void testInt16Quantization() {
    const std::vector<double> values = {-10.0, -1.5, 0.0, 2.5, 10.0};
    const QuantizationParameters parameters = Int16Quantizer::calibrate(values);
    const QuantizedVector quantized = Int16Quantizer::quantize(values, parameters);
    const std::vector<double> restored = Int16Quantizer::dequantize(quantized);

    assert(quantized.values.size() == values.size());
    assert(parameters.scale > 0.0);
    for (size_t index = 0; index < values.size(); ++index) {
        assert(std::abs(restored[index] - values[index]) <= parameters.scale);
    }

    const QuantizationParameters constant_parameters =
        Int16Quantizer::calibrate({3.0, 3.0});
    const QuantizedVector constant = Int16Quantizer::quantize({3.0}, constant_parameters);
    assert(std::abs(Int16Quantizer::dequantize(constant)[0] - 3.0) <= constant_parameters.scale);

    bool threw = false;
    try {
        Int16Quantizer::calibrate({});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testInt8Quantization() {
    const std::vector<double> values = {-10.0, -1.5, 0.0, 2.5, 10.0};
    const QuantizationParameters parameters = Int8Quantizer::calibrate(values);
    const Int8QuantizedVector quantized = Int8Quantizer::quantize(values, parameters);
    const std::vector<double> restored = Int8Quantizer::dequantize(quantized);

    assert(quantized.values.size() == values.size());
    assert(parameters.scale > 0.0);
    for (size_t index = 0; index < values.size(); ++index) {
        assert(std::abs(restored[index] - values[index]) <= parameters.scale);
    }
}

void testQuantizedInt8FIFOMemory() {
    const std::vector<TrainingSample> calibration_samples = {
        {{{0.0, 1.0}}, {{0.0}}},
        {{{10.0, 11.0}}, {{10.0}}}
    };
    QuantizedInt8FIFOMemory memory(
        2,
        Int8TrainingSampleQuantizer::calibrate(calibration_samples)
    );
    memory.add(calibration_samples[0]);
    memory.add(calibration_samples[1]);
    QuantizedFIFOMemory int16_memory(
        2,
        TrainingSampleQuantizer::calibrate(calibration_samples)
    );
    int16_memory.add(calibration_samples[0]);
    int16_memory.add(calibration_samples[1]);
    assert(memory.size() == 2);
    assert(memory.memoryUsedBytes() == 2 * memory.bytesPerSample());
    assert(memory.bytesPerSample() < int16_memory.bytesPerSample());
    assert(!memory.sample(1).empty());
}

void testTrainingSampleQuantization() {
    const std::vector<TrainingSample> samples = {
        {{{-1.0, 0.0}}, {{0.0}}, 0.7},
        {{{1.0, 2.0}}, {{1.0}}, 0.3}
    };
    const TrainingSampleQuantizer quantizer = TrainingSampleQuantizer::calibrate(samples);
    const QuantizedTrainingSample encoded = quantizer.encode(samples[0]);
    const TrainingSample decoded = quantizer.decode(encoded);

    assert(decoded.input.size() == samples[0].input.size());
    assert(decoded.target.size() == samples[0].target.size());
    assert(std::abs(decoded.input[0] - samples[0].input[0]) <= quantizer.inputParameters().scale);
    assert(std::abs(decoded.target[0] - samples[0].target[0]) <= quantizer.targetParameters().scale);
    assert(decoded.priority == samples[0].priority);
}

void testQuantizedFIFOMemory() {
    const std::vector<TrainingSample> calibration_samples = {
        {{{0.0, 1.0}}, {{0.0}}},
        {{{10.0, 11.0}}, {{10.0}}}
    };
    QuantizedFIFOMemory memory(
        2,
        TrainingSampleQuantizer::calibrate(calibration_samples)
    );
    memory.add(calibration_samples[0]);
    memory.add(calibration_samples[1]);

    assert(memory.size() == 2);
    assert(memory.bytesPerSample() > 0);
    assert(memory.memoryUsedBytes() == 2 * memory.bytesPerSample());
    const std::vector<TrainingSample> restored = memory.sample(1);
    assert(std::abs(restored[0].input[0] - 0.0) <= 1.0);
}

void testTrainingSampleSerialization() {
    const std::string path = "/tmp/gloomy_training_samples.bin";
    const std::vector<TrainingSample> samples = {
        {{{1.0, 2.0}}, {{3.0}}, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 4, 2}
    };
    TrainingSampleSerialization::save(path, samples);
    const std::vector<TrainingSample> restored = TrainingSampleSerialization::load(path);
    assert(restored.size() == 1);
    assert(restored[0].input == samples[0].input);
    assert(restored[0].target == samples[0].target);
    assert(restored[0].priority == samples[0].priority);
    assert(restored[0].age == samples[0].age);
    assert(restored[0].usage_count == samples[0].usage_count);
    std::remove(path.c_str());
}

void testNetworkSerialization() {
    const std::string path = "/tmp/gloomy_network.bin";
    NeuralNetwork original;
    original.algorithm = "tanh";
    original.post_algorithm = "none";
    original.addLayer(2, 2);
    original.addLayer(2, 1);
    original.layers()[0].weights()[0][0] = 0.2;
    original.layers()[0].weights()[0][1] = -0.3;
    original.layers()[0].weights()[1][0] = 0.4;
    original.layers()[0].weights()[1][1] = 0.5;
    original.layers()[0].bias()[0] = 0.1;
    original.layers()[0].bias()[1] = -0.2;
    original.layers()[1].weights()[0][0] = 0.6;
    original.layers()[1].weights()[1][0] = -0.7;
    original.layers()[1].bias()[0] = 0.3;

    const std::vector<double> input = {0.5, -0.25};
    const std::vector<double> expected = original.forward(input);
    NetworkSerialization::save(path, original);

    NeuralNetwork restored;
    NetworkSerialization::load(path, restored);
    const std::vector<double> actual = restored.forward(input);
    assert(actual.size() == expected.size());
    assert(std::abs(actual[0] - expected[0]) < 1e-12);
    assert(restored.algorithm == original.algorithm);
    assert(restored.layers().size() == original.layers().size());

    std::fstream corrupt(path, std::ios::in | std::ios::out | std::ios::binary);
    corrupt.seekp(16);
    char byte = 0;
    corrupt.read(&byte, sizeof(byte));
    corrupt.seekp(16);
    byte ^= 1;
    corrupt.write(&byte, sizeof(byte));
    corrupt.close();
    bool checksum_failed = false;
    try {
        NetworkSerialization::load(path, restored);
    } catch (const std::runtime_error&) {
        checksum_failed = true;
    }
    assert(checksum_failed);
    std::remove(path.c_str());
}

void testMetrics() {
    const RegressionMetrics metrics = Metrics::regression({1.0, 3.0}, {0.0, 1.0});
    assert(std::abs(metrics.mae - 1.5) < 1e-12);
    assert(std::abs(metrics.rmse - std::sqrt(2.5)) < 1e-12);
    assert(std::abs(Metrics::forgetting(0.9, 0.7) - 0.2) < 1e-12);

    bool threw = false;
    try {
        Metrics::regression({1.0}, {});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testBenchmarkCsv() {
    const std::string path = "/tmp/gloomy_benchmark.csv";
    BenchmarkResult result;
    result.memory_strategy = "hybrid,8kb";
    result.precision = "int16";
    result.optimizer = "sgd";
    result.mae = 0.25;
    result.memory_used_bytes = 8192;
    result.samples_stored = 32;
    BenchmarkCsv::write(path, {result});

    std::ifstream stream(path);
    std::string content(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
    );
    assert(content.find("memory_strategy,precision,optimizer") == 0);
    assert(content.find("\"hybrid,8kb\",int16,sgd") != std::string::npos);
    std::remove(path.c_str());
}

void testMomentumOptimizer() {
    DenseLayer layer(1, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.0;
    layer.bias()[0] = 0.0;
    MSELoss loss;
    const std::vector<double> input = {1.0};
    const std::vector<double> target = {2.0};

    layer.forward(input);
    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));
    std::vector<DenseLayer> layers;
    layers.push_back(layer);
    MomentumOptimizer optimizer(0.1, 0.9);
    optimizer.update(layers);
    const double first_weight = layers[0].weights()[0][0];

    layers[0].forward(input);
    layers[0].zeroGradients();
    layers[0].backward(loss.gradient(layers[0].forward(input), target));
    optimizer.update(layers);
    assert(std::abs(layers[0].weights()[0][0]) > std::abs(first_weight));
    assert(optimizer.stateBytes() == sizeof(double) * 2);
}

void testAdamOptimizer() {
    DenseLayer layer(1, 1);
    layer.set_algorithm("none");
    layer.weights()[0][0] = 0.0;
    layer.bias()[0] = 0.0;
    MSELoss loss;
    const std::vector<double> input = {1.0};
    const std::vector<double> target = {2.0};

    layer.forward(input);
    layer.zeroGradients();
    layer.backward(loss.gradient(layer.forward(input), target));
    std::vector<DenseLayer> layers;
    layers.push_back(layer);
    AdamOptimizer optimizer(0.05);
    optimizer.update(layers);
    assert(layers[0].weights()[0][0] > 0.0);
    assert(optimizer.stateBytes() == sizeof(double) * 4);
}

void testOptimizerSerialization() {
    const std::string path = "/tmp/gloomy_optimizer.bin";

    {
        SGDOptimizer original(0.05);
        OptimizerSerialization::save(path, original);
        std::vector<DenseLayer> layers;
        std::unique_ptr<Optimizer> restored = OptimizerSerialization::load(path, layers);
        auto* sgd = dynamic_cast<SGDOptimizer*>(restored.get());
        assert(sgd != nullptr);
        assertClose(sgd->learningRate(), 0.05);
    }

    {
        DenseLayer layer(1, 1);
        layer.set_algorithm("none");
        layer.weights()[0][0] = 0.0;
        layer.bias()[0] = 0.0;
        MSELoss loss;
        const std::vector<double> input = {1.0};
        const std::vector<double> target = {2.0};

        std::vector<DenseLayer> layers;
        layers.push_back(layer);
        MomentumOptimizer original(0.1, 0.9);
        layers[0].forward(input);
        layers[0].zeroGradients();
        layers[0].backward(loss.gradient(layers[0].forward(input), target));
        original.update(layers);

        OptimizerSerialization::save(path, original);
        std::unique_ptr<Optimizer> restored = OptimizerSerialization::load(path, layers);
        auto* momentum = dynamic_cast<MomentumOptimizer*>(restored.get());
        assert(momentum != nullptr);
        assertClose(momentum->learningRate(), 0.1);
        assertClose(momentum->momentum(), 0.9);
        assert(momentum->stateBytes() == original.stateBytes());

        layers[0].forward(input);
        layers[0].zeroGradients();
        layers[0].backward(loss.gradient(layers[0].forward(input), target));
        std::vector<DenseLayer> layers_copy = layers;
        original.update(layers);
        momentum->update(layers_copy);
        assertClose(layers[0].weights()[0][0], layers_copy[0].weights()[0][0]);

        std::vector<DenseLayer> mismatched;
        mismatched.push_back(DenseLayer(2, 1));
        bool shape_rejected = false;
        try {
            OptimizerSerialization::load(path, mismatched);
        } catch (const std::runtime_error&) {
            shape_rejected = true;
        }
        assert(shape_rejected);
    }

    {
        DenseLayer layer(1, 1);
        layer.set_algorithm("none");
        layer.weights()[0][0] = 0.0;
        layer.bias()[0] = 0.0;
        MSELoss loss;
        const std::vector<double> input = {1.0};
        const std::vector<double> target = {2.0};

        std::vector<DenseLayer> layers;
        layers.push_back(layer);
        AdamOptimizer original(0.05);
        layers[0].forward(input);
        layers[0].zeroGradients();
        layers[0].backward(loss.gradient(layers[0].forward(input), target));
        original.update(layers);

        OptimizerSerialization::save(path, original);
        std::unique_ptr<Optimizer> restored = OptimizerSerialization::load(path, layers);
        auto* adam = dynamic_cast<AdamOptimizer*>(restored.get());
        assert(adam != nullptr);
        assertClose(adam->learningRate(), 0.05);
        assert(adam->stateBytes() == original.stateBytes());

        layers[0].forward(input);
        layers[0].zeroGradients();
        layers[0].backward(loss.gradient(layers[0].forward(input), target));
        std::vector<DenseLayer> layers_copy = layers;
        original.update(layers);
        adam->update(layers_copy);
        assertClose(layers[0].weights()[0][0], layers_copy[0].weights()[0][0]);
    }

    {
        SGDOptimizer original(0.2);
        OptimizerSerialization::save(path, original);
        std::fstream corrupt(path, std::ios::in | std::ios::out | std::ios::binary);
        corrupt.seekp(16);
        char byte = 0;
        corrupt.read(&byte, sizeof(byte));
        corrupt.seekp(16);
        byte ^= 1;
        corrupt.write(&byte, sizeof(byte));
        corrupt.close();
        std::vector<DenseLayer> layers;
        bool checksum_failed = false;
        try {
            OptimizerSerialization::load(path, layers);
        } catch (const std::runtime_error&) {
            checksum_failed = true;
        }
        assert(checksum_failed);
    }

    std::remove(path.c_str());
}

void testLearningMemorySerialization() {
    const std::string path = "/tmp/gloomy_memory.bin";

    // FIFO round-trip: capacity, samples and metadata survive.
    {
        FIFOMemory original(3);
        original.add({{1.0}, {2.0}, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 7, 2});
        original.add({{2.0}, {3.0}, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 1, 5});
        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        assert(dynamic_cast<FIFOMemory*>(restored.get()) != nullptr);
        assert(restored->capacity() == 3);
        assert(restored->size() == 2);
        const std::vector<MemoryEntry> entries = restored->sampleIndexed(2);
        assert(entries.size() == 2);
        assertClose(entries[0].sample.input[0], 1.0);
        assert(entries[0].sample.age == 7);
        assert(entries[1].sample.usage_count == 6);
    }

    // Reservoir round-trip: capacity, samples and RNG state (future draws
    // must stay reproducible, not just the seed).
    {
        ReservoirMemory original(3, 42u);
        for (int index = 0; index < 5; ++index) {
            original.add({{static_cast<double>(index)}, {static_cast<double>(index)}});
        }
        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        assert(dynamic_cast<ReservoirMemory*>(restored.get()) != nullptr);
        assert(restored->capacity() == 3);
        assert(restored->size() == original.size());

        const std::vector<TrainingSample> expected = original.sample(3);
        const std::vector<TrainingSample> actual = restored->sample(3);
        assert(expected.size() == actual.size());
        for (size_t index = 0; index < expected.size(); ++index) {
            assertClose(expected[index].input[0], actual[index].input[0]);
            assert(expected[index].usage_count == actual[index].usage_count);
        }
    }

    // Prioritized round-trip: alpha and RNG state must keep draws reproducible.
    {
        PrioritizedMemory original(3, 0.7, 11u);
        original.add({{0.0}, {0.0}, 0.2});
        original.add({{1.0}, {1.0}, 0.9});
        original.add({{2.0}, {2.0}, 0.5});
        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        auto* prioritized = dynamic_cast<PrioritizedMemory*>(restored.get());
        assert(prioritized != nullptr);
        assertClose(prioritized->alpha(), 0.7);

        const std::vector<TrainingSample> expected = original.sample(3);
        const std::vector<TrainingSample> actual = restored->sample(3);
        assert(expected.size() == actual.size());
        for (size_t index = 0; index < expected.size(); ++index) {
            assertClose(expected[index].input[0], actual[index].input[0]);
        }
    }

    // Novelty round-trip.
    {
        NoveltyMemory original(3, 0.5);
        original.add({{0.0, 0.0}, {0.0}});
        original.add({{5.0, 5.0}, {1.0}});
        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        auto* novelty = dynamic_cast<NoveltyMemory*>(restored.get());
        assert(novelty != nullptr);
        assertClose(novelty->noveltyThreshold(), 0.5);
        assert(restored->size() == 2);
    }

    // Hybrid round-trip: ratios, threshold, partitions, seen_samples and RNG state.
    {
        HybridMemory original(8);
        for (int index = 0; index < 6; ++index) {
            TrainingSample sample;
            sample.input = {static_cast<double>(index)};
            sample.target = {static_cast<double>(index)};
            sample.priority = (index % 2 == 0) ? 0.9 : 0.1;
            original.add(sample);
        }
        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        auto* hybrid = dynamic_cast<HybridMemory*>(restored.get());
        assert(hybrid != nullptr);
        assert(hybrid->partitionSizes() == original.partitionSizes());

        const std::vector<TrainingSample> expected = original.sample(6);
        const std::vector<TrainingSample> actual = hybrid->sample(6);
        assert(expected.size() == actual.size());
        for (size_t index = 0; index < expected.size(); ++index) {
            assertClose(expected[index].input[0], actual[index].input[0]);
        }
    }

    // Corruption must be rejected.
    {
        FIFOMemory original(2);
        original.add({{1.0}, {1.0}});
        LearningMemorySerialization::save(path, original);
        std::fstream corrupt(path, std::ios::in | std::ios::out | std::ios::binary);
        corrupt.seekp(16);
        char byte = 0;
        corrupt.read(&byte, sizeof(byte));
        corrupt.seekp(16);
        byte ^= 1;
        corrupt.write(&byte, sizeof(byte));
        corrupt.close();
        bool checksum_failed = false;
        try {
            LearningMemorySerialization::load(path);
        } catch (const std::runtime_error&) {
            checksum_failed = true;
        }
        assert(checksum_failed);
    }

    std::remove(path.c_str());
}

void testGloomyConfigDefaults() {
    const GloomyConfig& config = GloomyConfig::defaults();

    // Architecture: must match the CLI defaults documented in the README
    // and previously hardcoded in src/main.cpp.
    assert(config.runtime == "inference");
    assert(config.activation == "none");
    assert(config.post_activation == "none");
    assert(config.hidden_layers == 2);
    assert(config.neurons == 2);
    assert(config.predictions == 1);

    // Loss: matches HuberLoss's own constructor default.
    assert(config.loss == "mse");
    assertClose(config.huber_delta, 1.0);

    // Optimizer: momentum, beta1, beta2 and epsilon match the concrete
    // optimizers' own constructor defaults.
    assert(config.optimizer == "sgd");
    assertClose(config.momentum, 0.9);
    assertClose(config.beta1, 0.9);
    assertClose(config.beta2, 0.999);
    assertClose(config.epsilon, 1e-8);

    // Learning memory: ratios match HybridMemoryRatios's own defaults,
    // prioritized_alpha matches PrioritizedMemory's own default, and seed
    // matches the shared default used by Reservoir, Prioritized and Hybrid
    // memories (itself std::mt19937's own default seed).
    assert(config.memory_strategy == "fifo");
    assertClose(config.recent_ratio, 0.25);
    assertClose(config.error_ratio, 0.25);
    assertClose(config.novelty_ratio, 0.25);
    assertClose(config.historical_ratio, 0.25);
    assertClose(config.novelty_threshold, 0.0);
    assertClose(config.prioritized_alpha, 0.6);
    assertClose(config.prioritized_beta, 0.4);
    assert(config.seed == 5489u);

    assert(config.precision == "float64");
    assert(config.train_every == 1);

    assert(config.metrics_path == "benchmark_results.csv");

    // Calling defaults() twice must return the exact same values.
    assert(&GloomyConfig::defaults() == &config);
}

void testGloomyConfigFile() {
    const std::string path = "/tmp/gloomy_test.config";

    // Recognized keys override the defaults; comments and blank lines are
    // ignored; whitespace around keys and values is trimmed.
    {
        std::ofstream file(path, std::ios::trunc);
        file << "# comment\n";
        file << "\n";
        file << "  activation = tanh  \n";
        file << "hidden_layers=3\n";
        file << "learning_rate=0.001\n";
        file << "memory_strategy=hybrid\n";
        file << "memory_capacity=64\n";
        file << "seed=99\n";
        file.close();

        const GloomyConfig config = GloomyConfigFile::load(path);
        assert(config.activation == "tanh");
        assert(config.hidden_layers == 3);
        assertClose(config.learning_rate, 0.001);
        assert(config.memory_strategy == "hybrid");
        assert(config.memory_capacity == 64);
        assert(config.seed == 99u);
        // Untouched keys keep the base value (defaults here).
        assert(config.neurons == GloomyConfig::defaults().neurons);
        assert(config.post_activation == GloomyConfig::defaults().post_activation);
    }

    // A base other than the defaults can be layered on top of.
    {
        GloomyConfig base = GloomyConfig::defaults();
        base.neurons = 42;
        const GloomyConfig config = GloomyConfigFile::load(path, base);
        assert(config.neurons == 42);
        assert(config.activation == "tanh");
    }

    // Unknown key is rejected.
    {
        std::ofstream file(path, std::ios::trunc);
        file << "not_a_real_key=1\n";
        file.close();
        bool threw = false;
        try {
            GloomyConfigFile::load(path);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Malformed line (no '=') is rejected.
    {
        std::ofstream file(path, std::ios::trunc);
        file << "this line has no separator\n";
        file.close();
        bool threw = false;
        try {
            GloomyConfigFile::load(path);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Invalid numeric value is rejected.
    {
        std::ofstream file(path, std::ios::trunc);
        file << "hidden_layers=not_a_number\n";
        file.close();
        bool threw = false;
        try {
            GloomyConfigFile::load(path);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Missing file is rejected.
    {
        bool threw = false;
        try {
            GloomyConfigFile::load("/tmp/gloomy_missing_config_file_for_test.config");
        } catch (const std::runtime_error&) {
            threw = true;
        }
        assert(threw);
    }

    std::remove(path.c_str());
}

void testOnlineLearningRuntime() {
    const std::vector<double> sequence = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // Basic success with the defaults: one step per consecutive pair, a
    // finite average loss, and a non-empty memory afterwards.
    {
        const OnlineLearningResult result = runOnlineLearning(GloomyConfig::defaults(), sequence);
        assert(result.steps.size() == sequence.size() - 1);
        for (size_t index = 0; index < result.steps.size(); ++index) {
            assertClose(result.steps[index].observation, sequence[index]);
            assertClose(result.steps[index].target, sequence[index + 1]);
            assert(std::isfinite(result.steps[index].prediction_before_update));
            assert(std::isfinite(result.steps[index].loss_before_update));
        }
        assert(std::isfinite(result.average_loss));
        assert(result.memory_size > 0);
        assert(result.memory_size <= GloomyConfig::defaults().memory_capacity);
    }

    // Each loss/optimizer/memory strategy combination must run without
    // throwing and produce a finite average loss.
    {
        const std::vector<std::string> losses = {"mse", "mae", "huber"};
        const std::vector<std::string> optimizers = {"sgd", "momentum", "adam"};
        const std::vector<std::string> strategies = {"fifo", "reservoir", "prioritized", "novelty", "hybrid"};
        for (const std::string& loss_name : losses) {
            for (const std::string& optimizer_name : optimizers) {
                for (const std::string& strategy : strategies) {
                    GloomyConfig config = GloomyConfig::defaults();
                    config.loss = loss_name;
                    config.optimizer = optimizer_name;
                    config.memory_strategy = strategy;
                    config.memory_capacity = 8;
                    const OnlineLearningResult result = runOnlineLearning(config, sequence);
                    assert(result.steps.size() == sequence.size() - 1);
                    assert(std::isfinite(result.average_loss));
                }
            }
        }
    }

    // A sequence with fewer than two values is rejected.
    {
        bool threw = false;
        try {
            runOnlineLearning(GloomyConfig::defaults(), {1.0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Unknown loss, optimizer and memory strategy are each rejected.
    {
        GloomyConfig config = GloomyConfig::defaults();
        config.loss = "bogus";
        bool threw = false;
        try {
            runOnlineLearning(config, sequence);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        GloomyConfig config = GloomyConfig::defaults();
        config.optimizer = "bogus";
        bool threw = false;
        try {
            runOnlineLearning(config, sequence);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        GloomyConfig config = GloomyConfig::defaults();
        config.memory_strategy = "bogus";
        bool threw = false;
        try {
            runOnlineLearning(config, sequence);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

}

int main() {
    testMSECompute();
    testMSEGradient();
    testAdditionalLosses();
    testInvalidSizes();
    testEmptyInputs();
    testDenseLayerGradient();
    testDenseLayerSeededInitialization();
    testGradientCheckingAllActivations();
    testSoftmaxGradientCheck();
    testActivationStabilityWithLargeValues();
    testNonFiniteValuesRejected();
    testSGDUpdateReducesLoss();
    testLearningEngineBatchTraining();
    testFIFOMemoryCapacity();
    testMemoryMetadata();
    testLearningEngineMemoryReplay();
    testReservoirMemory();
    testPrioritizedMemory();
    testPrioritizedMemoryBiasCorrection();
    testLearningEngineTrainBatchWeighting();
    testNoveltyMemory();
    testHybridMemory();
    testHybridMemoryTrueRecency();
    testImportanceScorer();
    testTrainingSchedulers();
    testStreamingNormalization();
    testNormalizationSerialization();
    testInt16Quantization();
    testInt8Quantization();
    testQuantizedInt8FIFOMemory();
    testTrainingSampleQuantization();
    testQuantizedFIFOMemory();
    testTrainingSampleSerialization();
    testNetworkSerialization();
    testMetrics();
    testBenchmarkCsv();
    testMomentumOptimizer();
    testAdamOptimizer();
    testOptimizerSerialization();
    testLearningMemorySerialization();
    testGloomyConfigDefaults();
    testGloomyConfigFile();
    testOnlineLearningRuntime();
    return 0;
}
