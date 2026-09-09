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
#include "DeltaQuantization.h"
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
#include "CompressedAdamOptimizer.h"
#include "OptimizerSerialization.h"
#include "LearningMemorySerialization.h"
#include "ModelSerialization.h"
#include "NetworkQuantization.h"
#include "QuantizedNetworkSerialization.h"
#include "GloomyConfig.h"
#include "GloomyConfigFile.h"
#include "OnlineLearningRuntime.h"
#include "ConceptDriftDetector.h"
#include "PageHinkleyDetector.h"
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

void testCrossEntropyLoss() {
    // compute()/gradient() directs, sur une distribution one-hot simple.
    {
        CrossEntropyLoss loss;
        const std::vector<double> prediction = {0.7, 0.2, 0.1};
        const std::vector<double> target = {1.0, 0.0, 0.0};
        assertClose(loss.compute(prediction, target), -std::log(0.7));

        const std::vector<double> gradient = loss.gradient(prediction, target);
        assert(gradient.size() == 3);
        assertClose(gradient[0], -1.0 / 0.7);
        assertClose(gradient[1], 0.0);
        assertClose(gradient[2], 0.0);
    }

    // Une prediction negative (pas une probabilite valide — ex. des logits
    // bruts passes par erreur sans post_algorithm=softmax) est rejetee avec
    // un message explicite, plutot que de produire silencieusement
    // log(negatif) = NaN.
    {
        CrossEntropyLoss loss;
        bool threw = false;
        try {
            loss.compute({-0.1, 1.1}, {1.0, 0.0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Verification par difference finie sur un reseau complet a sortie
    // softmax a 3 classes : confirme que le gradient de CrossEntropyLoss se
    // compose correctement avec le jacobien softmax deja implemente dans
    // DenseLayer::backward (meme principe que testSoftmaxGradientCheck,
    // avec une perte de classification cette fois plutot que MSE).
    {
        DenseLayer::seedWeightInitialization(13u);
        NeuralNetwork network;
        network.algorithm = "none";
        network.post_algorithm = "softmax";
        network.addLayer(3, 4);
        network.addLayer(4, 3);

        const std::vector<double> input = {0.5, -0.2, 0.8};
        const std::vector<double> target = {0.0, 1.0, 0.0};
        CrossEntropyLoss loss;
        checkNetworkGradient(network, loss, input, target);
    }
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

void testPrioritizedMemoryBetaAnnealing() {
    // beta_annealing_rate = 0.0 (defaut) : beta reste strictement fixe,
    // meme apres de nombreux replays — comportement inchange par rapport
    // a avant ce champ.
    {
        PrioritizedMemory memory(3, 0.6, 1234u, 0.4);
        memory.add({{1.0}, {1.0}, 0.5});
        memory.add({{2.0}, {2.0}, 0.5});
        memory.add({{3.0}, {3.0}, 0.5});
        assertClose(memory.betaAnnealingRate(), 0.0);
        for (int index = 0; index < 10; ++index) {
            memory.sampleIndexed(2);
        }
        assertClose(memory.beta(), 0.4);
    }

    // beta_annealing_rate > 0 : beta augmente de ce montant apres chaque
    // sampleIndexed(), jusqu'a plafonner a 1.0 (pratique courante :
    // 0.4 -> 1.0, voir docs/memory.md).
    {
        PrioritizedMemory memory(3, 0.6, 1234u, 0.4, 0.1);
        memory.add({{1.0}, {1.0}, 0.5});
        memory.add({{2.0}, {2.0}, 0.5});
        memory.add({{3.0}, {3.0}, 0.5});
        assertClose(memory.beta(), 0.4);

        memory.sampleIndexed(2);
        assertClose(memory.beta(), 0.5);
        memory.sampleIndexed(2);
        assertClose(memory.beta(), 0.6);

        for (int index = 0; index < 20; ++index) {
            memory.sampleIndexed(2);
        }
        assertClose(memory.beta(), 1.0);
    }

    // Une valeur negative est rejetee.
    {
        bool threw = false;
        try {
            PrioritizedMemory memory(2, 0.6, 1234u, 0.4, -0.1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

void testPrioritizedMemoryExplorationEpsilon() {
    // exploration_epsilon = 0.0 (defaut) ne change rien : un echantillon a
    // priorite quasi nulle n'est (quasiment) jamais choisi si un autre a
    // une forte priorite.
    {
        PrioritizedMemory memory(2, 1.0, 1234u, 0.4, 0.0, 0.0);
        assertClose(memory.explorationEpsilon(), 0.0);
        memory.add({{1.0}, {1.0}, 1e-9});
        memory.add({{2.0}, {2.0}, 1000.0});

        int low_priority_selected = 0;
        for (int trial = 0; trial < 200; ++trial) {
            const std::vector<TrainingSample> drawn = memory.sample(1);
            if (drawn[0].input[0] == 1.0) ++low_priority_selected;
        }
        assert(low_priority_selected < 5);
    }

    // exploration_epsilon proche de 1 rend le tirage quasi uniforme : le
    // meme echantillon a priorite quasi nulle est desormais choisi une
    // fraction significative du temps.
    {
        PrioritizedMemory memory(2, 1.0, 1234u, 0.4, 0.0, 0.9);
        assertClose(memory.explorationEpsilon(), 0.9);
        memory.add({{1.0}, {1.0}, 1e-9});
        memory.add({{2.0}, {2.0}, 1000.0});

        int low_priority_selected = 0;
        for (int trial = 0; trial < 200; ++trial) {
            const std::vector<TrainingSample> drawn = memory.sample(1);
            if (drawn[0].input[0] == 1.0) ++low_priority_selected;
        }
        assert(low_priority_selected > 50);
    }

    // Une valeur hors de [0, 1] est rejetee.
    {
        bool threw = false;
        try {
            PrioritizedMemory memory(2, 0.6, 1234u, 0.4, 0.0, 1.5);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
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

void testImportanceScoreComponents() {
    // learn() (sans replay immediat, via un scheduler qui ne se declenche
    // jamais dans ce test) : recency et rarity sont desormais reellement
    // calculees pour l'echantillon tout juste ajoute (age=0, usage_count=0)
    // au lieu de rester a 0.0 comme avant ce changement ; novelty/diversity
    // ne le sont pas a ce stade (pas de contexte de batch disponible — voir
    // docs/memory.md, « Score d'importance »). Sous les poids par defaut
    // (error=1.0, le reste a 0.0), la priorite reste exactement l'erreur
    // seule : comportement historique inchange.
    {
        NeuralNetwork network;
        network.algorithm = "none";
        network.addLayer(1, 1);
        network.layers()[0].weights()[0][0] = 0.0;
        network.layers()[0].bias()[0] = 0.0;
        MSELoss loss;
        SGDOptimizer optimizer(0.1);
        LearningEngine engine(network, loss, optimizer);
        FIFOMemory memory(4);
        EveryNScheduler never(1000);

        engine.learn(memory, {{1.0}, {1.0}}, 1, never);
        const std::vector<TrainingSample> stored = memory.sample(1);
        assertClose(stored[0].recency, 1.0);
        assertClose(stored[0].rarity, 1.0);
        assertClose(stored[0].novelty, 0.0);
        assertClose(stored[0].diversity, 0.0);
        assertClose(stored[0].priority, 1.0);
    }

    // trainFromMemory() : novelty/diversity sont calculees par rapport aux
    // AUTRES echantillons du meme batch de replay. Avec un batch de deux
    // echantillons distincts, chacun n'a qu'un seul "autre" a comparer :
    // novelty et diversity coincident alors exactement.
    {
        NeuralNetwork network;
        network.algorithm = "none";
        network.addLayer(1, 1);
        network.layers()[0].weights()[0][0] = 0.0;
        network.layers()[0].bias()[0] = 0.0;
        MSELoss loss;
        SGDOptimizer optimizer(0.1);
        LearningEngine engine(network, loss, optimizer);
        FIFOMemory memory(2);
        memory.add({{1.0}, {2.0}});
        memory.add({{10.0}, {20.0}});

        engine.trainFromMemory(memory, 2);
        const std::vector<TrainingSample> stored = memory.sample(2);
        for (const TrainingSample& sample : stored) {
            assert(sample.novelty > 0.0);
            assertClose(sample.novelty, sample.diversity);
        }
    }

    // Un batch d'un seul echantillon (memoire a un seul element) : novelty
    // maximale (1.0) et diversite nulle par convention, faute d'un autre
    // echantillon auquel se comparer.
    {
        NeuralNetwork network;
        network.algorithm = "none";
        network.addLayer(1, 1);
        network.layers()[0].weights()[0][0] = 0.0;
        network.layers()[0].bias()[0] = 0.0;
        MSELoss loss;
        SGDOptimizer optimizer(0.1);
        LearningEngine engine(network, loss, optimizer);
        FIFOMemory memory(2);
        memory.add({{1.0}, {2.0}});

        engine.trainFromMemory(memory, 1);
        const std::vector<TrainingSample> stored = memory.sample(1);
        assertClose(stored[0].novelty, 1.0);
        assertClose(stored[0].diversity, 0.0);
    }

    // Poids non par defaut : donner tout le poids a recency (aucun a
    // error) fait dependre la priorite de l'age plutot que de l'erreur.
    {
        NeuralNetwork network;
        network.algorithm = "none";
        network.addLayer(1, 1);
        network.layers()[0].weights()[0][0] = 0.0;
        network.layers()[0].bias()[0] = 0.0;
        MSELoss loss;
        SGDOptimizer optimizer(0.1);
        ImportanceWeights weights;
        weights.error = 0.0;
        weights.recency = 1.0;
        LearningEngine engine(network, loss, optimizer, weights);
        FIFOMemory memory(3);
        memory.add({{1.0}, {1.0}});
        memory.add({{2.0}, {2.0}});
        memory.advanceAges();
        memory.advanceAges();
        memory.advanceAges();

        // trainFromMemory() avance l'age une fois de plus en interne :
        // age=4 au moment du calcul, recency = 1/(1+4) = 0.2.
        engine.trainFromMemory(memory, 2);
        const std::vector<TrainingSample> stored = memory.sample(2);
        for (const TrainingSample& sample : stored) {
            assertClose(sample.priority, sample.recency);
            assertClose(sample.priority, 0.2);
        }
    }
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

    // Bug corrige : une plage de valeurs qui ne contient pas 0 (ex. un
    // capteur toujours positif, loin de 0) ecrasait auparavant toutes les
    // valeurs vers le meme code quantifie (zero_point sature silencieusement
    // hors de [int16_min, int16_max]), une perte totale d'information sans
    // aucune exception. Desormais la plage de calibration est etendue pour
    // toujours inclure 0, ce qui elimine la saturation (voir
    // Quantization.cpp). Verifie ici avec des valeurs realistes de type
    // capteur, loin de zero et de faible amplitude relative.
    {
        const std::vector<double> offset_values = {17.0, 20.0, 23.0, 26.0};
        const QuantizationParameters offset_parameters = Int16Quantizer::calibrate(offset_values);
        const QuantizedVector offset_quantized = Int16Quantizer::quantize(offset_values, offset_parameters);
        const std::vector<double> offset_restored = Int16Quantizer::dequantize(offset_quantized);

        // Les 4 valeurs distinctes doivent rester distinguables apres
        // dequantification (pas toutes ecrasees vers la meme valeur).
        assert(offset_restored[0] != offset_restored[3]);
        for (size_t index = 0; index < offset_values.size(); ++index) {
            assert(std::abs(offset_restored[index] - offset_values[index]) <= offset_parameters.scale);
        }
    }
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

    // Meme regression que testInt16Quantization ci-dessus, pour le codec
    // int8 (voir Int8Quantization.cpp).
    {
        const std::vector<double> offset_values = {17.0, 20.0, 23.0, 26.0};
        const QuantizationParameters offset_parameters = Int8Quantizer::calibrate(offset_values);
        const Int8QuantizedVector offset_quantized = Int8Quantizer::quantize(offset_values, offset_parameters);
        const std::vector<double> offset_restored = Int8Quantizer::dequantize(offset_quantized);

        assert(offset_restored[0] != offset_restored[3]);
        for (size_t index = 0; index < offset_values.size(); ++index) {
            assert(std::abs(offset_restored[index] - offset_values[index]) <= offset_parameters.scale);
        }
    }
}

namespace {
double maxAbsError(const std::vector<double>& expected, const std::vector<double>& actual) {
    double worst = 0.0;
    for (size_t index = 0; index < expected.size(); ++index) {
        worst = std::max(worst, std::abs(expected[index] - actual[index]));
    }
    return worst;
}
}

void testDeltaQuantization() {
    // Serie lisse (derive lente, offset de type capteur) : les deltas
    // consecutifs ont une plage bien plus etroite que les valeurs brutes,
    // donc un pas de quantification (scale) plus fin pour le meme nombre de
    // bits — la compression differentielle doit donc reduire nettement
    // l'erreur de reconstruction par rapport a une quantification directe
    // des valeurs brutes. Valeurs mesurees reellement (voir
    // docs/quantization.md, « Compression differentielle »).
    {
        std::vector<double> values;
        for (int index = 0; index < 100; ++index) {
            values.push_back(20.0 + std::sin(index / 15.0) * 3.0 + index * 0.01);
        }
        const DeltaQuantizedSeries encoded = DeltaQuantizer::encode(values);
        const std::vector<double> delta_restored = DeltaQuantizer::decode(encoded);
        assert(delta_restored.size() == values.size());

        const QuantizationParameters direct_parameters = Int16Quantizer::calibrate(values);
        const QuantizedVector direct_quantized = Int16Quantizer::quantize(values, direct_parameters);
        const std::vector<double> direct_restored = Int16Quantizer::dequantize(direct_quantized);

        const double delta_error = maxAbsError(values, delta_restored);
        const double direct_error = maxAbsError(values, direct_restored);
        assert(delta_error < direct_error);
        assert(delta_error < 0.001);
    }

    // Rampe parfaitement lineaire (deltas exactement constants) : cas ideal
    // pour la compression differentielle, l'erreur de reconstruction doit
    // rester quasiment nulle alors que la quantification directe, sur une
    // plage de valeurs bien plus large (offset 1000), perd nettement plus
    // de precision.
    {
        std::vector<double> values;
        for (int index = 0; index < 100; ++index) {
            values.push_back(1000.0 + index * 0.5);
        }
        const DeltaQuantizedSeries encoded = DeltaQuantizer::encode(values);
        const std::vector<double> delta_restored = DeltaQuantizer::decode(encoded);

        const QuantizationParameters direct_parameters = Int16Quantizer::calibrate(values);
        const QuantizedVector direct_quantized = Int16Quantizer::quantize(values, direct_parameters);
        const std::vector<double> direct_restored = Int16Quantizer::dequantize(direct_quantized);

        const double delta_error = maxAbsError(values, delta_restored);
        const double direct_error = maxAbsError(values, direct_restored);
        assert(delta_error < 1e-6);
        assert(delta_error < direct_error);
    }

    // Serie bruitee/erratique : les deltas n'ont pas une plage plus etroite
    // que les valeurs elles-memes (les sauts sont aussi grands que la
    // serie), donc la compression differentielle n'apporte ici aucun
    // avantage et peut meme faire legerement moins bien qu'une
    // quantification directe (erreur cumulative le long de la
    // reconstruction) — un compromis reel, pas systematiquement gagnant,
    // documente explicitement (voir docs/quantization.md).
    {
        const std::vector<double> values = {10.0, 15.0, 8.0, 20.0, 3.0, 18.0, 6.0, 22.0, 1.0, 25.0};
        const DeltaQuantizedSeries encoded = DeltaQuantizer::encode(values);
        const std::vector<double> delta_restored = DeltaQuantizer::decode(encoded);

        const QuantizationParameters direct_parameters = Int16Quantizer::calibrate(values);
        const QuantizedVector direct_quantized = Int16Quantizer::quantize(values, direct_parameters);
        const std::vector<double> direct_restored = Int16Quantizer::dequantize(direct_quantized);

        const double delta_error = maxAbsError(values, delta_restored);
        const double direct_error = maxAbsError(values, direct_restored);
        assert(delta_error > direct_error);
    }

    // Serie a une seule valeur : aucun delta a encoder, decode() retrouve
    // exactement la valeur d'origine.
    {
        const DeltaQuantizedSeries encoded = DeltaQuantizer::encode({42.0});
        assert(encoded.deltas.values.empty());
        const std::vector<double> restored = DeltaQuantizer::decode(encoded);
        assert(restored.size() == 1);
        assertClose(restored[0], 42.0);
    }

    // Serie vide rejetee.
    {
        bool threw = false;
        try {
            DeltaQuantizer::encode({});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // quantizedBytes() : first_value (double) + un int16 par delta + les
    // parametres de calibration.
    {
        const DeltaQuantizedSeries encoded = DeltaQuantizer::encode({1.0, 2.0, 3.0});
        assert(encoded.deltas.values.size() == 2);
        const size_t expected_bytes =
            sizeof(double) + 2 * sizeof(std::int16_t) + sizeof(QuantizationParameters);
        assert(DeltaQuantizer::quantizedBytes(encoded) == expected_bytes);
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

void testMemoryRejectsNonFiniteOrEmptySamples() {
    // Before this change, FIFO/Reservoir/Hybrid never validated input/target
    // at all; Prioritized only validated `priority`; Novelty only validated
    // non-emptiness and dimension consistency. All seven strategies must
    // now reject an empty or non-finite sample the same way, via the shared
    // validateTrainingSampleVectors (voir docs/roadmap.md, « Priorité
    // haute »).
    const double nan_value = std::numeric_limits<double>::quiet_NaN();
    const double inf_value = std::numeric_limits<double>::infinity();

    {
        FIFOMemory memory(2);
        bool threw = false;
        try {
            memory.add({{}, {1.0}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        threw = false;
        try {
            memory.add({{nan_value}, {1.0}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        threw = false;
        try {
            memory.add({{1.0}, {inf_value}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
        assert(memory.size() == 0);
    }
    {
        ReservoirMemory memory(2, 1234u);
        bool threw = false;
        try {
            memory.add({{nan_value}, {1.0}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        PrioritizedMemory memory(2, 0.6, 1234u);
        bool threw = false;
        try {
            memory.add({{nan_value}, {1.0}, 0.5});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        NoveltyMemory memory(2, 0.5);
        bool threw = false;
        try {
            memory.add({{1.0}, {inf_value}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        HybridMemory memory(4);
        bool threw = false;
        try {
            memory.add({{nan_value}, {1.0}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        const std::vector<TrainingSample> calibration_samples = {
            {{0.0}, {0.0}},
            {{10.0}, {10.0}}
        };
        QuantizedFIFOMemory memory(2, TrainingSampleQuantizer::calibrate(calibration_samples));
        bool threw = false;
        try {
            memory.add({{nan_value}, {1.0}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        const std::vector<TrainingSample> calibration_samples = {
            {{0.0}, {0.0}},
            {{10.0}, {10.0}}
        };
        QuantizedInt8FIFOMemory memory(2, Int8TrainingSampleQuantizer::calibrate(calibration_samples));
        bool threw = false;
        try {
            memory.add({{1.0}, {inf_value}});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
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

void testModelSerialization() {
    const std::string path = "/tmp/gloomy_model.bin";

    NeuralNetwork network;
    network.algorithm = "tanh";
    network.post_algorithm = "none";
    network.addLayer(2, 2);
    network.addLayer(2, 1);
    network.layers()[0].weights()[0][0] = 0.2;
    network.layers()[0].weights()[0][1] = -0.3;
    network.layers()[0].weights()[1][0] = 0.4;
    network.layers()[0].weights()[1][1] = 0.5;
    network.layers()[0].bias()[0] = 0.1;
    network.layers()[0].bias()[1] = -0.2;
    network.layers()[1].weights()[0][0] = 0.6;
    network.layers()[1].weights()[1][0] = -0.7;
    network.layers()[1].bias()[0] = 0.3;

    StreamingNormalizer normalizer(2);
    normalizer.update({1.0, 2.0});
    normalizer.update({3.0, 4.0});

    MomentumOptimizer optimizer(0.01, 0.9);
    ReservoirMemory memory(4, 7u);
    memory.add({{1.0, 2.0}, {3.0}, 0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 4, 1});

    const std::vector<double> model_input = {0.5, -0.25};
    network.forward(model_input);
    network.zeroGradients();
    network.backward({1.0});
    optimizer.update(network.layers());

    ModelSerialization::save(path, network, normalizer, optimizer, memory);

    ModelSerialization::LoadedModel loaded = ModelSerialization::load(path);
    assert(loaded.normalizer != nullptr);
    assert(loaded.optimizer != nullptr);
    assert(loaded.memory != nullptr);

    const std::vector<double> input = {0.5, -0.25};
    const std::vector<double> expected = network.forward(input);
    const std::vector<double> actual = loaded.network.forward(input);
    assert(actual.size() == expected.size());
    assert(std::abs(actual[0] - expected[0]) < 1e-12);
    assert(loaded.network.algorithm == network.algorithm);
    assert(loaded.normalizer->count() == normalizer.count());
    assert(loaded.normalizer->dimensions() == normalizer.dimensions());

    auto* momentum = dynamic_cast<MomentumOptimizer*>(loaded.optimizer.get());
    assert(momentum != nullptr);
    assertClose(momentum->learningRate(), 0.01);
    assertClose(momentum->momentum(), 0.9);

    auto* reservoir = dynamic_cast<ReservoirMemory*>(loaded.memory.get());
    assert(reservoir != nullptr);
    assert(reservoir->capacity() == memory.capacity());
    assert(reservoir->size() == memory.size());

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
        ModelSerialization::load(path);
    } catch (const std::runtime_error&) {
        checksum_failed = true;
    }
    assert(checksum_failed);

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

void testNetworkQuantization() {
    // Reseau reellement entraine (pas des poids arbitraires) : c'est le cas
    // d'usage vise, quantifier un modele apres apprentissage.
    DenseLayer::seedWeightInitialization(4242u);
    std::vector<TrainingSample> training;
    training.reserve(80);
    for (size_t index = 0; index < 80; ++index) {
        const double x = static_cast<double>(index) / 10.0;
        training.push_back({{x}, {2.0 * x + 1.0}});
    }

    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 8);
    network.addLayer(8, 1);
    MSELoss loss;
    SGDOptimizer optimizer(0.01);
    LearningEngine engine(network, loss, optimizer);
    engine.train(training, 80, 8);

    const QuantizedNetwork quantized = NetworkQuantization::quantize(network);
    assert(quantized.layers.size() == network.layers().size());
    NeuralNetwork dequantized = NetworkQuantization::dequantize(quantized);
    assert(dequantized.algorithm == network.algorithm);
    assert(dequantized.layers().size() == network.layers().size());

    // Impact sur la precision : mesure sur plusieurs points avant d'ecrire
    // ce test (voir docs/quantization.md) — max_abs_diff ~0.049,
    // max_rel_diff ~0.23% sur cette tache. Marges larges ci-dessous.
    double max_abs_diff = 0.0;
    for (double x = 0.0; x <= 10.0; x += 0.5) {
        const double original = network.forward({x})[0];
        const double approx = dequantized.forward({x})[0];
        max_abs_diff = std::max(max_abs_diff, std::abs(original - approx));
    }
    assert(max_abs_diff < 0.2);

    // Occupe moins de memoire que les poids/biais flottants d'origine,
    // meme avec le surcout de calibration par couche (voir
    // docs/quantization.md pour le detail : le ratio depend fortement de la
    // taille du reseau, negligeable ici sur un si petit reseau).
    size_t float_bytes = 0;
    for (const DenseLayer& layer : network.layers()) {
        for (const auto& row : layer.weights()) {
            float_bytes += row.size() * sizeof(double);
        }
        float_bytes += layer.bias().size() * sizeof(double);
    }
    const size_t quantized_bytes = NetworkQuantization::quantizedBytes(quantized);
    assert(quantized_bytes < float_bytes);

    // Dimensions incoherentes explicitement rejetees plutot que silencieusement
    // mal interpretees.
    QuantizedNetwork broken = quantized;
    broken.layers[0].input_size = 999;
    bool threw = false;
    try {
        NetworkQuantization::dequantize(broken);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

void testQuantizedNetworkSerialization() {
    NeuralNetwork network;
    network.algorithm = "tanh";
    network.post_algorithm = "none";
    network.addLayer(2, 2);
    network.addLayer(2, 1);
    network.layers()[0].weights()[0][0] = 0.2;
    network.layers()[0].weights()[0][1] = -0.3;
    network.layers()[0].weights()[1][0] = 0.4;
    network.layers()[0].weights()[1][1] = 0.5;
    network.layers()[0].bias()[0] = 0.1;
    network.layers()[0].bias()[1] = -0.2;
    network.layers()[1].weights()[0][0] = 0.6;
    network.layers()[1].weights()[1][0] = -0.7;
    network.layers()[1].bias()[0] = 0.3;

    const QuantizedNetwork quantized = NetworkQuantization::quantize(network);
    const std::string path = "/tmp/gloomy_quantized_network.bin";
    QuantizedNetworkSerialization::save(path, quantized);

    const QuantizedNetwork restored = QuantizedNetworkSerialization::load(path);
    assert(restored.algorithm == quantized.algorithm);
    assert(restored.post_algorithm == quantized.post_algorithm);
    assert(restored.layers.size() == quantized.layers.size());
    for (size_t index = 0; index < quantized.layers.size(); ++index) {
        assert(restored.layers[index].input_size == quantized.layers[index].input_size);
        assert(restored.layers[index].output_size == quantized.layers[index].output_size);
        assert(restored.layers[index].weights.values == quantized.layers[index].weights.values);
        assert(restored.layers[index].bias.values == quantized.layers[index].bias.values);
        assertClose(restored.layers[index].weights.parameters.scale, quantized.layers[index].weights.parameters.scale);
    }

    // La reconstruction flottante depuis le fichier rejoue doit produire la
    // meme prediction que la reconstruction depuis l'objet en memoire.
    const std::vector<double> input = {0.5, -0.25};
    const NeuralNetwork from_memory = NetworkQuantization::dequantize(quantized);
    const NeuralNetwork from_disk = NetworkQuantization::dequantize(restored);
    NeuralNetwork mutable_from_memory = from_memory;
    NeuralNetwork mutable_from_disk = from_disk;
    assertClose(mutable_from_memory.forward(input)[0], mutable_from_disk.forward(input)[0]);

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
        QuantizedNetworkSerialization::load(path);
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
    result.loss_function = "huber";
    result.mae = 0.25;
    result.memory_used_bytes = 8192;
    result.samples_stored = 32;
    result.memory_capacity = 64;
    result.mae_ratio_to_full_dataset = 2.5;
    result.seed = 1234u;
    result.mae_mean = 0.5;
    result.mae_stddev = 0.25;
    result.mae_ci95_margin = 0.75;
    result.approximate_macs = 4096;
    result.samples_per_second = 1000;
    result.updates_per_second = 125;
    BenchmarkCsv::write(path, {result});

    std::ifstream stream(path);
    std::string content(
        (std::istreambuf_iterator<char>(stream)),
        std::istreambuf_iterator<char>()
    );
    assert(content.find("memory_strategy,precision,optimizer,loss_function") == 0);
    assert(content.find("memory_capacity,mae_ratio_to_full_dataset") != std::string::npos);
    assert(content.find("seed,mae_mean,mae_stddev,mae_ci95_margin") != std::string::npos);
    assert(content.find("samples_per_second,updates_per_second") != std::string::npos);
    assert(content.find("\"hybrid,8kb\",int16,sgd,huber") != std::string::npos);
    assert(content.find(",64,2.5") != std::string::npos);
    assert(content.find(",1234,0.5,0.25,0.75,4096,1000,125") != std::string::npos);
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

void testCompressedAdamOptimizer() {
    // Meme mise a jour de base qu'AdamOptimizer : le poids doit bouger dans
    // la meme direction.
    {
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
        CompressedAdamOptimizer optimizer(0.05);
        optimizer.update(layers);
        assert(layers[0].weights()[0][0] > 0.0);
    }

    // Convergence comparable a AdamOptimizer sur une tache reelle (y=2x+1,
    // reseau 1-8-1, 80 epochs), et etat compresse reellement plus petit
    // (mesure : 228 octets contre 400 pour un Adam natif sur cette
    // architecture — un gain reel mais loin du 4x theorique, le cout fixe
    // de calibration par vecteur dominant sur un reseau aussi petit, comme
    // pour NetworkQuantization — voir docs/optimizers.md).
    {
        const auto makeSamples = []() {
            std::vector<TrainingSample> samples;
            for (int index = 0; index < 80; ++index) {
                const double x = index / 10.0;
                samples.push_back({{x}, {2.0 * x + 1.0}});
            }
            return samples;
        };

        DenseLayer::seedWeightInitialization(4242u);
        NeuralNetwork adam_network;
        adam_network.algorithm = "none";
        adam_network.addLayer(1, 8);
        adam_network.addLayer(8, 1);
        MSELoss adam_loss;
        AdamOptimizer adam_optimizer(0.01);
        LearningEngine adam_engine(adam_network, adam_loss, adam_optimizer);
        const double adam_final_loss = adam_engine.train(makeSamples(), 80, 8);

        DenseLayer::seedWeightInitialization(4242u);
        NeuralNetwork compressed_network;
        compressed_network.algorithm = "none";
        compressed_network.addLayer(1, 8);
        compressed_network.addLayer(8, 1);
        MSELoss compressed_loss;
        CompressedAdamOptimizer compressed_optimizer(0.01);
        LearningEngine compressed_engine(compressed_network, compressed_loss, compressed_optimizer);
        const double compressed_final_loss = compressed_engine.train(makeSamples(), 80, 8);

        assert(std::isfinite(adam_final_loss));
        assert(std::isfinite(compressed_final_loss));
        assert(adam_final_loss < 0.001);
        // La quantification des moments introduit une petite perte de
        // precision : la perte finale compressee reste proche de celle
        // d'Adam natif, sans etre forcement identique.
        assert(compressed_final_loss < 0.01);
        assert(compressed_optimizer.stateBytes() < adam_optimizer.stateBytes());
    }

    // Validation des hyperparametres, comme AdamOptimizer.
    {
        bool threw = false;
        try {
            CompressedAdamOptimizer optimizer(-0.1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

void testOptimizerRejectsNonFiniteGradients() {
    // DenseLayer::backward already rejects a non-finite *incoming* gradient
    // (see testNonFiniteValuesRejected), so it cannot be used here to get a
    // non-finite weight gradient into an optimizer. A legitimate way this
    // can still happen is numeric overflow during backward's own
    // multiplication/accumulation (input * local_gradient) when both
    // operands are finite but extreme — this is exactly what
    // Optimizer::update's own validation (voir docs/roadmap.md, « Priorité
    // haute ») must catch, since backward() never re-checks its own output.
    auto makeOverflowingLayer = []() {
        DenseLayer layer(1, 1);
        layer.set_algorithm("none");
        layer.weights()[0][0] = 0.0;
        layer.bias()[0] = 0.0;
        layer.forward({1e200});
        layer.zeroGradients();
        layer.backward({1e200});
        return layer;
    };
    assert(!std::isfinite(makeOverflowingLayer().weightGradients()[0][0]));

    {
        std::vector<DenseLayer> layers;
        layers.push_back(makeOverflowingLayer());
        bool threw = false;
        try {
            SGDOptimizer(0.1).update(layers);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        std::vector<DenseLayer> layers;
        layers.push_back(makeOverflowingLayer());
        bool threw = false;
        try {
            MomentumOptimizer(0.1, 0.9).update(layers);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        std::vector<DenseLayer> layers;
        layers.push_back(makeOverflowingLayer());
        bool threw = false;
        try {
            AdamOptimizer(0.05).update(layers);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
    {
        std::vector<DenseLayer> layers;
        layers.push_back(makeOverflowingLayer());
        bool threw = false;
        try {
            CompressedAdamOptimizer(0.05).update(layers);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // A non-finite or non-positive gradient_scale is rejected too, even
    // with otherwise-finite gradients.
    {
        DenseLayer layer(1, 1);
        layer.set_algorithm("none");
        layer.weights()[0][0] = 0.0;
        layer.bias()[0] = 0.0;
        layer.forward({1.0});
        layer.zeroGradients();
        layer.backward({1.0});
        std::vector<DenseLayer> layers;
        layers.push_back(layer);

        SGDOptimizer optimizer(0.1);
        bool threw = false;
        try {
            optimizer.update(layers, std::numeric_limits<double>::quiet_NaN());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
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
        assertClose(prioritized->betaAnnealingRate(), 0.0);
        assertClose(prioritized->explorationEpsilon(), 0.0);

        const std::vector<TrainingSample> expected = original.sample(3);
        const std::vector<TrainingSample> actual = restored->sample(3);
        assert(expected.size() == actual.size());
        for (size_t index = 0; index < expected.size(); ++index) {
            assertClose(expected[index].input[0], actual[index].input[0]);
        }
    }

    // Prioritized round-trip with beta annealing and exploration epsilon
    // set: both must survive the round-trip, and annealing already applied
    // before saving (a non-default beta) must be preserved exactly.
    {
        PrioritizedMemory original(3, 0.6, 22u, 0.4, 0.1, 0.2);
        original.add({{0.0}, {0.0}, 0.2});
        original.add({{1.0}, {1.0}, 0.9});
        original.sampleIndexed(2);
        assertClose(original.beta(), 0.5);

        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        auto* prioritized = dynamic_cast<PrioritizedMemory*>(restored.get());
        assert(prioritized != nullptr);
        assertClose(prioritized->beta(), 0.5);
        assertClose(prioritized->betaAnnealingRate(), 0.1);
        assertClose(prioritized->explorationEpsilon(), 0.2);
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

    // QuantizedFIFOMemory (int16) round-trip: capacity, calibration
    // parameters (scale/zero_point, shared by all samples) and quantized
    // samples must survive, within the codec's own quantization error.
    {
        const std::vector<TrainingSample> calibration_samples = {
            {{0.0, 1.0}, {0.0}},
            {{10.0, 11.0}, {10.0}}
        };
        QuantizedFIFOMemory original(3, TrainingSampleQuantizer::calibrate(calibration_samples));
        original.add(calibration_samples[0]);
        original.add(calibration_samples[1]);
        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        assert(dynamic_cast<QuantizedFIFOMemory*>(restored.get()) != nullptr);
        assert(restored->capacity() == 3);
        assert(restored->size() == 2);

        const std::vector<TrainingSample> expected = original.sample(2);
        const std::vector<TrainingSample> actual = restored->sample(2);
        assert(expected.size() == actual.size());
        for (size_t index = 0; index < expected.size(); ++index) {
            assertClose(expected[index].input[0], actual[index].input[0]);
            assertClose(expected[index].target[0], actual[index].target[0]);
        }
    }

    // QuantizedInt8FIFOMemory round-trip: same as above, int8 codec.
    {
        const std::vector<TrainingSample> calibration_samples = {
            {{0.0, 1.0}, {0.0}},
            {{10.0, 11.0}, {10.0}}
        };
        QuantizedInt8FIFOMemory original(3, Int8TrainingSampleQuantizer::calibrate(calibration_samples));
        original.add(calibration_samples[0]);
        original.add(calibration_samples[1]);
        LearningMemorySerialization::save(path, original);
        std::unique_ptr<LearningMemory> restored = LearningMemorySerialization::load(path);
        assert(dynamic_cast<QuantizedInt8FIFOMemory*>(restored.get()) != nullptr);
        assert(restored->capacity() == 3);
        assert(restored->size() == 2);

        const std::vector<TrainingSample> expected = original.sample(2);
        const std::vector<TrainingSample> actual = restored->sample(2);
        assert(expected.size() == actual.size());
        for (size_t index = 0; index < expected.size(); ++index) {
            assertClose(expected[index].input[0], actual[index].input[0]);
            assertClose(expected[index].target[0], actual[index].target[0]);
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
    // window_size = 1 reproduces the historical scalar-only runtime input
    // exactly (voir docs/roadmap.md, « Priorité haute »).
    assert(config.window_size == 1);

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
    // 0.0 desactive l'annealing et l'exploration : comportement inchange
    // par rapport a avant ces deux champs (voir docs/memory.md).
    assertClose(config.prioritized_beta_annealing_rate, 0.0);
    assertClose(config.prioritized_exploration_epsilon, 0.0);
    assert(config.seed == 5489u);

    // Score d'importance : le defaut (error seule) reproduit exactement le
    // comportement historique de LearningEngine.
    assertClose(config.importance_weight_error, 1.0);
    assertClose(config.importance_weight_novelty, 0.0);
    assertClose(config.importance_weight_rarity, 0.0);
    assertClose(config.importance_weight_recency, 0.0);
    assertClose(config.importance_weight_diversity, 0.0);

    assert(config.precision == "float64");
    assert(config.train_every == 1);

    // Persistence paths are opt-in: empty by default so that a CLI run
    // never silently writes files (or overwrites make benchmark's own
    // benchmark_results.csv) unless the user explicitly asks for it.
    assert(config.model_path.empty());
    assert(config.optimizer_path.empty());
    assert(config.memory_path.empty());
    assert(config.metrics_path.empty());

    // Concept drift detection is opt-in: disabled by default, no behavior
    // change for the online runtime unless explicitly enabled.
    assert(config.concept_drift_detection == false);
    assert(config.concept_drift_recent_window == 10);
    assert(config.concept_drift_minimum_history == 20);
    assertClose(config.concept_drift_std_devs, 3.0);

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
        file << "window_size=5\n";
        file << "prioritized_beta_annealing_rate=0.05\n";
        file << "prioritized_exploration_epsilon=0.1\n";
        file << "concept_drift_detection=true\n";
        file << "concept_drift_recent_window=7\n";
        file << "concept_drift_minimum_history=15\n";
        file << "concept_drift_std_devs=2.5\n";
        file << "importance_weight_novelty=0.2\n";
        file << "importance_weight_recency=0.3\n";
        file.close();

        const GloomyConfig config = GloomyConfigFile::load(path);
        assert(config.activation == "tanh");
        assert(config.hidden_layers == 3);
        assertClose(config.learning_rate, 0.001);
        assert(config.memory_strategy == "hybrid");
        assert(config.memory_capacity == 64);
        assert(config.seed == 99u);
        assert(config.window_size == 5);
        assertClose(config.prioritized_beta_annealing_rate, 0.05);
        assertClose(config.prioritized_exploration_epsilon, 0.1);
        assert(config.concept_drift_detection == true);
        assert(config.concept_drift_recent_window == 7);
        assert(config.concept_drift_minimum_history == 15);
        assertClose(config.concept_drift_std_devs, 2.5);
        assertClose(config.importance_weight_novelty, 0.2);
        assertClose(config.importance_weight_recency, 0.3);
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

    // Invalid boolean value is rejected (only true/false/1/0 accepted).
    {
        std::ofstream file(path, std::ios::trunc);
        file << "concept_drift_detection=maybe\n";
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

void testTrainingRuntime() {
    const std::vector<double> sequence = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    GloomyConfig config = GloomyConfig::defaults();
    config.runtime = "training";
    config.optimizer = "sgd";
    config.loss = "mse";
    config.memory_strategy = "fifo";
    config.batch_size = 4;
    config.epochs = 2;

    const TrainingResult result = runTraining(config, sequence);
    assert(result.network.layers().size() > 0);
    assert(result.normalizer != nullptr);
    assert(result.optimizer != nullptr);
    assert(std::isfinite(result.average_loss));

    bool threw = false;
    try {
        runTraining(config, {1.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    // window_size > 1: the network's input dimension must follow, and a
    // sequence with at most window_size values must be rejected (voir
    // docs/roadmap.md, « Priorité haute »).
    {
        GloomyConfig windowed = config;
        windowed.window_size = 3;
        const TrainingResult windowed_result = runTraining(windowed, sequence);
        assert(windowed_result.network.layers().front().weights().size() == 3);
        assert(std::isfinite(windowed_result.average_loss));

        bool window_rejected = false;
        try {
            runTraining(windowed, {1.0, 2.0, 3.0});
        } catch (const std::invalid_argument&) {
            window_rejected = true;
        }
        assert(window_rejected);
    }

    // window_size = 0 is rejected outright.
    {
        GloomyConfig zero_window = config;
        zero_window.window_size = 0;
        bool window_zero_rejected = false;
        try {
            runTraining(zero_window, sequence);
        } catch (const std::invalid_argument&) {
            window_zero_rejected = true;
        }
        assert(window_zero_rejected);
    }
}

void testTrainingRuntimeResumesFromSavedModel() {
    const std::string path = "/tmp/gloomy_training_loaded_model.bin";
    std::remove(path.c_str());

    const std::vector<double> sequence = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    GloomyConfig config = GloomyConfig::defaults();
    config.runtime = "training";
    config.batch_size = 4;
    config.epochs = 2;
    config.model_path = path;

    const TrainingResult baseline = runTraining(config, sequence);
    ModelSerialization::save(
        path,
        baseline.network,
        *baseline.normalizer,
        *baseline.optimizer,
        *baseline.memory
    );

    // Resuming must reuse the saved network/normalizer/optimizer/memory
    // rather than rebuilding a fresh network from config.
    const TrainingResult resumed = runTraining(config, sequence, path);
    assert(resumed.network.layers().size() > 0);
    assert(resumed.normalizer != nullptr);
    assert(resumed.optimizer != nullptr);
    assert(resumed.memory != nullptr);
    assert(std::isfinite(resumed.average_loss));

    // A window_size mismatch against the resumed model is rejected rather
    // than silently reinterpreting the input dimension.
    GloomyConfig mismatched = config;
    mismatched.window_size = 2;
    bool threw = false;
    try {
        runTraining(mismatched, sequence, path);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::remove(path.c_str());
}

void testTrainingRuntimePersistencePaths() {
    const std::string prefix = "/tmp/gloomy_training_runtime_artifacts";
    const std::string model_path = prefix + ".gloomy";
    const std::string optimizer_path = prefix + ".optimizer.gloomy";
    const std::string memory_path = prefix + ".memory.gloomy";
    const std::string metrics_path = prefix + ".metrics.csv";

    std::remove(model_path.c_str());
    std::remove(optimizer_path.c_str());
    std::remove(memory_path.c_str());
    std::remove(metrics_path.c_str());

    GloomyConfig config = GloomyConfig::defaults();
    config.runtime = "training";
    config.model_path = model_path;
    config.optimizer_path = optimizer_path;
    config.memory_path = memory_path;
    config.metrics_path = metrics_path;
    config.batch_size = 4;
    config.epochs = 2;

    const TrainingResult result = runTraining(config, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    saveTrainingArtifacts(config, result);

    assert(std::ifstream(model_path, std::ios::binary).good());
    assert(std::ifstream(optimizer_path, std::ios::binary).good());
    assert(std::ifstream(memory_path, std::ios::binary).good());

    std::ifstream metrics(metrics_path);
    std::string metrics_content(
        (std::istreambuf_iterator<char>(metrics)),
        std::istreambuf_iterator<char>()
    );
    assert(metrics_content.find("average_loss=") != std::string::npos);

    std::remove(model_path.c_str());
    std::remove(optimizer_path.c_str());
    std::remove(memory_path.c_str());
    std::remove(metrics_path.c_str());
}

void testOnlineLearningRuntimePersistencePaths() {
    const std::string prefix = "/tmp/gloomy_online_runtime_artifacts";
    const std::string model_path = prefix + ".gloomy";
    const std::string optimizer_path = prefix + ".optimizer.gloomy";
    const std::string memory_path = prefix + ".memory.gloomy";
    const std::string metrics_path = prefix + ".metrics.csv";

    std::remove(model_path.c_str());
    std::remove(optimizer_path.c_str());
    std::remove(memory_path.c_str());
    std::remove(metrics_path.c_str());

    GloomyConfig config = GloomyConfig::defaults();
    config.model_path = model_path;
    config.optimizer_path = optimizer_path;
    config.memory_path = memory_path;
    config.metrics_path = metrics_path;
    config.memory_capacity = 8;

    const OnlineLearningResult result = runOnlineLearning(config, {1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    saveOnlineArtifacts(config, result);

    assert(std::ifstream(model_path, std::ios::binary).good());
    assert(std::ifstream(optimizer_path, std::ios::binary).good());
    assert(std::ifstream(memory_path, std::ios::binary).good());

    std::ifstream metrics(metrics_path);
    std::string metrics_content(
        (std::istreambuf_iterator<char>(metrics)),
        std::istreambuf_iterator<char>()
    );
    assert(metrics_content.find("average_loss=") != std::string::npos);
    assert(metrics_content.find("memory_size=") != std::string::npos);

    std::remove(model_path.c_str());
    std::remove(optimizer_path.c_str());
    std::remove(memory_path.c_str());
    std::remove(metrics_path.c_str());
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

    // The online learning result should expose the fully trained state for the
    // CLI to persist as a unified GLOOMY_MODEL, and a saved model should be
    // loadable back into the online runtime.
    {
        const std::string path = "/tmp/gloomy_online_loaded_model.bin";
        std::remove(path.c_str());

        GloomyConfig config = GloomyConfig::defaults();
        config.memory_capacity = 8;
        config.model_path = path;

        const OnlineLearningResult baseline = runOnlineLearning(config, sequence);
        ModelSerialization::save(
            path,
            baseline.network,
            *baseline.normalizer,
            *baseline.optimizer,
            *baseline.memory
        );

        const OnlineLearningResult resumed = runOnlineLearning(config, sequence, path);
        assert(resumed.network.layers().size() > 0);
        assert(resumed.normalizer != nullptr);
        assert(resumed.optimizer != nullptr);
        assert(resumed.memory != nullptr);
        assert(resumed.memory_size == resumed.memory->size());

        std::remove(path.c_str());
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

void testOnlineLearningRuntimeWindowSize() {
    const std::vector<double> sequence = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    // window_size = 3: the network's input dimension must follow, the
    // number of steps must shrink to one per full window, and the recorded
    // scalar observation/target keep their existing meaning (the newest raw
    // value entering the window / the following value) — voir
    // docs/roadmap.md, « Priorité haute ».
    {
        GloomyConfig config = GloomyConfig::defaults();
        config.window_size = 3;
        config.memory_capacity = 8;
        const OnlineLearningResult result = runOnlineLearning(config, sequence);
        assert(result.steps.size() == sequence.size() - config.window_size);
        assert(result.network.layers().front().weights().size() == 3);
        for (size_t index = 0; index < result.steps.size(); ++index) {
            assertClose(result.steps[index].observation, sequence[index + config.window_size - 1]);
            assertClose(result.steps[index].target, sequence[index + config.window_size]);
        }
        assert(std::isfinite(result.average_loss));
    }

    // A sequence with at most window_size values cannot produce a single
    // step and must be rejected.
    {
        GloomyConfig config = GloomyConfig::defaults();
        config.window_size = 4;
        bool threw = false;
        try {
            runOnlineLearning(config, {1.0, 2.0, 3.0, 4.0});
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // window_size = 0 is rejected outright.
    {
        GloomyConfig config = GloomyConfig::defaults();
        config.window_size = 0;
        bool threw = false;
        try {
            runOnlineLearning(config, sequence);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Resuming a saved model with a different window_size than the one it
    // was trained with is rejected rather than silently mismatched.
    {
        const std::string path = "/tmp/gloomy_online_window_mismatch.bin";
        std::remove(path.c_str());

        GloomyConfig config = GloomyConfig::defaults();
        config.window_size = 2;
        config.memory_capacity = 8;
        config.model_path = path;
        const OnlineLearningResult baseline = runOnlineLearning(config, sequence);
        ModelSerialization::save(
            path,
            baseline.network,
            *baseline.normalizer,
            *baseline.optimizer,
            *baseline.memory
        );

        GloomyConfig mismatched = config;
        mismatched.window_size = 3;
        bool threw = false;
        try {
            runOnlineLearning(mismatched, sequence, path);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        std::remove(path.c_str());
    }
}

void testOnlineLearningRuntimeConceptDriftDetection() {
    std::vector<double> sequence;
    const int n_a = 60;
    for (int i = 0; i < n_a; ++i) sequence.push_back(2.0 * i + 1.0);
    const int n_b = 30;
    for (int j = 0; j < n_b; ++j) sequence.push_back(-3.0 * j + 1000.0);

    // Desactivee par defaut : ne signale jamais de derive, quelle que soit
    // la sequence (mecanisme opt-in, aucun changement de comportement tant
    // qu'il n'est pas active explicitement).
    {
        const OnlineLearningResult result = runOnlineLearning(GloomyConfig::defaults(), sequence);
        for (const OnlineLearningStep& step : result.steps) {
            assert(!step.drift_detected);
        }
    }

    // Activee : un changement de regime net (pente 2 -> pente -3, ordonnee
    // 1 -> 1000) apres un regime A assez long pour converger doit etre
    // signale peu apres la transition, et le signal doit s'eteindre a
    // nouveau une fois le reseau readapte — calibre sur une execution reelle
    // (voir docs/memory.md, « Detection de concept drift »).
    {
        GloomyConfig config = GloomyConfig::defaults();
        config.concept_drift_detection = true;
        config.concept_drift_recent_window = 5;
        config.concept_drift_minimum_history = 20;
        config.concept_drift_std_devs = 3.0;
        config.memory_capacity = 16;

        const OnlineLearningResult result = runOnlineLearning(config, sequence);
        assert(result.steps.size() == sequence.size() - 1);

        int first_drift_step = -1;
        int drift_count = 0;
        for (size_t index = 0; index < result.steps.size(); ++index) {
            assert(std::isfinite(result.steps[index].loss_before_update));
            if (result.steps[index].drift_detected) {
                if (first_drift_step == -1) first_drift_step = static_cast<int>(index);
                ++drift_count;
            }
        }

        assert(drift_count > 0);
        // Le pas n_a - 1 est le premier dont la cible appartient au regime
        // B ; une marge est laissee pour le temps necessaire a la fenetre
        // recente de se remplir de valeurs du nouveau regime.
        assert(first_drift_step >= n_a - 1);
        assert(first_drift_step < n_a + 20);

        // Le reseau finit par se readapter : la derive n'est pas signalee
        // indefiniment (auto-correction naturelle, voir ConceptDriftDetector.h).
        assert(!result.steps.back().drift_detected);
    }
}

void testOnlineLearningRuntimeLongSequenceStability() {
    // Sequence plus longue que les autres tests du runtime online : exerce
    // sur davantage d'iterations la reutilisation des buffers
    // (raw_observation/raw_target/observation/target/sample) introduite
    // dans OnlineLearningRuntime.cpp pour reduire les allocations de la
    // boucle (voir docs/roadmap.md, « Priorité moyenne : compression et
    // embarqué »). Un bug de reutilisation (buffer pas redimensionne, valeur
    // d'une iteration qui fuite dans la suivante) se manifesterait plus
    // probablement sur un grand nombre d'iterations que sur les sequences
    // courtes deja testees ailleurs.
    std::vector<double> sequence(500);
    for (size_t index = 0; index < sequence.size(); ++index) {
        sequence[index] = std::sin(static_cast<double>(index) / 10.0) * 5.0 + 10.0;
    }

    GloomyConfig config = GloomyConfig::defaults();
    config.memory_capacity = 32;
    const OnlineLearningResult result = runOnlineLearning(config, sequence);

    assert(result.steps.size() == sequence.size() - 1);
    for (size_t index = 0; index < result.steps.size(); ++index) {
        assertClose(result.steps[index].observation, sequence[index]);
        assertClose(result.steps[index].target, sequence[index + 1]);
        assert(std::isfinite(result.steps[index].prediction_before_update));
        assert(std::isfinite(result.steps[index].loss_before_update));
    }
    assert(std::isfinite(result.average_loss));
    assert(result.memory_size > 0);
    assert(result.memory_size <= config.memory_capacity);
}

void testConceptDriftDetector() {
    // Regime stable : une fois la ligne de base et la fenetre recente
    // suffisamment remplies, une erreur parfaitement constante ne doit
    // jamais etre signalee comme une derive (recent_mean == baseline_mean,
    // jamais strictement superieur).
    {
        ConceptDriftDetector detector(/*recent_window=*/5, /*minimum_history=*/10, /*num_std_devs=*/3.0);
        bool any_drift = false;
        for (int index = 0; index < 50; ++index) {
            any_drift = any_drift || detector.update(0.01);
        }
        assert(!any_drift);
        assertClose(detector.recentMean(), 0.01);
        assert(detector.baselineCount() == 50);
    }

    // Saut net : apres un long regime stable a faible erreur, une serie de
    // valeurs nettement plus elevees doit etre signalee comme une derive
    // une fois que la fenetre recente en est dominee, alors que la ligne de
    // base (qui integre l'historique complet) n'a pas encore rattrape ce
    // niveau.
    {
        ConceptDriftDetector detector(/*recent_window=*/5, /*minimum_history=*/10, /*num_std_devs=*/3.0);
        for (int index = 0; index < 50; ++index) {
            detector.update(0.01);
        }
        assert(!detector.driftDetected());

        bool drift_seen = false;
        for (int index = 0; index < 10; ++index) {
            if (detector.update(5.0)) drift_seen = true;
        }
        assert(drift_seen);
        assert(detector.recentMean() > 4.0);
        // La ligne de base integre 50 valeurs basses et au plus 10 hautes :
        // elle reste tres en-dessous de la moyenne recente.
        assert(detector.baselineMean() < 2.0);
    }

    // Avant que la fenetre recente ou l'historique minimal ne soient
    // atteints, aucune derive n'est jamais signalee, meme avec des valeurs
    // tres dispersees (pas de faux positif au demarrage).
    {
        ConceptDriftDetector detector(/*recent_window=*/5, /*minimum_history=*/10, /*num_std_devs=*/3.0);
        assert(!detector.update(0.0));
        assert(!detector.update(100.0));
        assert(!detector.update(0.0));
    }

    // reset() efface completement l'etat.
    {
        ConceptDriftDetector detector(3, 3, 3.0);
        detector.update(1.0);
        detector.update(1.0);
        detector.update(1.0);
        assert(detector.baselineCount() == 3);
        detector.reset();
        assert(detector.baselineCount() == 0);
        assertClose(detector.baselineMean(), 0.0);
        assert(!detector.driftDetected());
    }

    // Constructeur : fenetre nulle ou seuil non fini/negatif rejetes.
    {
        bool threw = false;
        try {
            ConceptDriftDetector detector(0, 10, 3.0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        threw = false;
        try {
            ConceptDriftDetector detector(5, 10, -1.0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Une erreur non finie est rejetee.
    {
        ConceptDriftDetector detector(3, 3, 3.0);
        bool threw = false;
        try {
            detector.update(std::numeric_limits<double>::quiet_NaN());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

void testPageHinkleyDetector() {
    // Test sequentiel de detection de rupture (Page-Hinkley), une methode
    // distincte de ConceptDriftDetector (accumulation d'un ecart tolere
    // plutot que comparaison moyenne recente / ligne de base) — voir
    // docs/memory.md, « Détection de dérive plus avancée ».

    // Regime stable : un bruit de faible amplitude autour de la meme
    // moyenne ne doit jamais declencher de derive, avec les parametres par
    // defaut (calibres sur ce cas precisement).
    {
        PageHinkleyDetector detector;
        bool any_drift = false;
        for (int index = 0; index < 200; ++index) {
            const double value = 0.05 + (index % 2 == 0 ? 0.001 : -0.001);
            any_drift = any_drift || detector.update(value);
        }
        assert(!any_drift);
        assert(detector.count() == 200);
    }

    // Saut net apres un long regime stable a faible erreur : signale une
    // derive quelques pas seulement apres la transition (calibre sur une
    // execution reelle : detection au premier pas du nouveau regime).
    {
        PageHinkleyDetector detector;
        for (int index = 0; index < 50; ++index) {
            detector.update(0.01);
        }
        assert(!detector.driftDetected());

        bool drift_seen = false;
        int drift_step = -1;
        for (int index = 0; index < 20; ++index) {
            if (detector.update(5.0)) {
                drift_seen = true;
                if (drift_step == -1) drift_step = index;
            }
        }
        assert(drift_seen);
        assert(drift_step <= 4);
    }

    // reset() efface completement l'etat.
    {
        PageHinkleyDetector detector;
        detector.update(1.0);
        detector.update(1.0);
        assert(detector.count() == 2);
        detector.reset();
        assert(detector.count() == 0);
        assert(!detector.driftDetected());
    }

    // Constructeur : delta negatif ou lambda non fini/non positif rejetes.
    {
        bool threw = false;
        try {
            PageHinkleyDetector detector(-0.1, 10.0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);

        threw = false;
        try {
            PageHinkleyDetector detector(0.05, 0.0);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }

    // Une valeur non finie est rejetee.
    {
        PageHinkleyDetector detector;
        bool threw = false;
        try {
            detector.update(std::numeric_limits<double>::quiet_NaN());
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        assert(threw);
    }
}

// Regimes synthetiques pour les tests de catastrophic forgetting et de
// concept drift ci-dessous : memes fonctions que l'experience de
// BenchmarkRunner.cpp (regime A lineaire croissant, regime B lineaire
// decroissant), reprises ici pour que ces phenomenes soient verifies par
// des assertions numeriques dans la suite de tests, pas seulement observes
// dans le CSV du benchmark.
std::vector<TrainingSample> makeForgettingRegimeA(size_t count, size_t start) {
    std::vector<TrainingSample> samples;
    samples.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const double x = static_cast<double>(index + start) / 10.0;
        samples.push_back({{x}, {2.0 * x + 1.0}});
    }
    return samples;
}

std::vector<TrainingSample> makeForgettingRegimeB(size_t count, size_t start) {
    std::vector<TrainingSample> samples;
    samples.reserve(count);
    for (size_t index = 0; index < count; ++index) {
        const double x = static_cast<double>(index + start) / 10.0;
        samples.push_back({{x}, {-2.0 * x + 20.0}});
    }
    return samples;
}

double regimeLoss(NeuralNetwork& network, const LossFunction& loss, const std::vector<TrainingSample>& regime) {
    double total = 0.0;
    for (const TrainingSample& sample : regime) {
        total += loss.compute(network.forward(sample.input), sample.target);
    }
    return total / static_cast<double>(regime.size());
}

void testCatastrophicForgettingWithoutReplay() {
    DenseLayer::seedWeightInitialization(4242u);
    const std::vector<TrainingSample> regime_a = makeForgettingRegimeA(40, 0);
    const std::vector<TrainingSample> regime_b = makeForgettingRegimeB(40, 40);

    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 8);
    network.addLayer(8, 1);
    MSELoss loss;
    SGDOptimizer optimizer(0.01);
    LearningEngine engine(network, loss, optimizer);

    engine.train(regime_a, 80, 8);
    const double loss_a_before = regimeLoss(network, loss, regime_a);
    assert(loss_a_before < 0.01); // le reseau a correctement appris le regime A

    engine.train(regime_b, 80, 8);
    const double loss_a_after = regimeLoss(network, loss, regime_a);

    // Sans rejouer d'anciens exemples, apprendre B degrade fortement la
    // performance sur A : c'est l'oubli catastrophique (verifie sur
    // plusieurs seeds avant d'ecrire ce test : perte apres B toujours entre
    // ~18 et ~19 ici, jamais proche de loss_a_before).
    assert(loss_a_after > 10.0);
    assert(loss_a_after > loss_a_before * 100.0);
}

void testCatastrophicForgettingMitigatedByReplay() {
    // Meme seed et meme procedure que testCatastrophicForgettingWithoutReplay,
    // pour une comparaison directe : seul l'ajout d'un replay FIFO de A
    // pendant l'apprentissage de B change.
    DenseLayer::seedWeightInitialization(4242u);
    const std::vector<TrainingSample> regime_a = makeForgettingRegimeA(40, 0);
    const std::vector<TrainingSample> regime_b = makeForgettingRegimeB(40, 40);

    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 8);
    network.addLayer(8, 1);
    MSELoss loss;
    SGDOptimizer optimizer(0.01);
    LearningEngine engine(network, loss, optimizer);
    FIFOMemory memory(regime_a.size());

    engine.train(regime_a, 80, 8);
    for (const TrainingSample& sample : regime_a) {
        memory.add(sample);
    }

    engine.train(regime_b, 80, 8);
    const double loss_a_after_b = regimeLoss(network, loss, regime_a);
    assert(loss_a_after_b > 10.0); // memes conditions que sans replay : l'oubli a bien eu lieu

    for (size_t update = 0; update < 20; ++update) {
        engine.trainFromMemory(memory, 8);
    }
    const double loss_a_after_replay = regimeLoss(network, loss, regime_a);

    // Le replay reduit nettement la degradation (observe : facteur ~4 sur
    // plusieurs seeds), sans l'annuler completement.
    assert(loss_a_after_replay > 1.0);
    assert(loss_a_after_replay < loss_a_after_b / 2.0);
}

void testConceptDriftReturnToPreviousRegime() {
    // A -> B -> A : le reseau doit pouvoir se re-adapter au regime A apres
    // l'avoir revu, la preuve qu'il n'est pas durablement endommage par le
    // passage par B (adaptation a un drift de concept, pas seulement
    // l'oubli lui-meme, deja couvert par les deux tests precedents).
    DenseLayer::seedWeightInitialization(4242u);
    const std::vector<TrainingSample> regime_a = makeForgettingRegimeA(40, 0);
    const std::vector<TrainingSample> regime_b = makeForgettingRegimeB(40, 40);

    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 8);
    network.addLayer(8, 1);
    MSELoss loss;
    SGDOptimizer optimizer(0.01);
    LearningEngine engine(network, loss, optimizer);

    engine.train(regime_a, 80, 8);
    const double loss_a_first = regimeLoss(network, loss, regime_a);
    assert(loss_a_first < 0.01);

    engine.train(regime_b, 80, 8);
    const double loss_a_after_b = regimeLoss(network, loss, regime_a);
    assert(loss_a_after_b > 10.0); // le drift vers B a bien degrade A

    engine.train(regime_a, 80, 8);
    const double loss_a_second = regimeLoss(network, loss, regime_a);

    // Le reseau retrouve une performance sur A comparable a la premiere
    // fois : il s'adapte au retour du regime, il n'est pas bloque par le
    // passage par B.
    assert(loss_a_second < 0.01);
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
    testCrossEntropyLoss();
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
    testPrioritizedMemoryBetaAnnealing();
    testPrioritizedMemoryExplorationEpsilon();
    testLearningEngineTrainBatchWeighting();
    testImportanceScoreComponents();
    testNoveltyMemory();
    testHybridMemory();
    testHybridMemoryTrueRecency();
    testImportanceScorer();
    testTrainingSchedulers();
    testStreamingNormalization();
    testNormalizationSerialization();
    testInt16Quantization();
    testInt8Quantization();
    testDeltaQuantization();
    testQuantizedInt8FIFOMemory();
    testTrainingSampleQuantization();
    testQuantizedFIFOMemory();
    testMemoryRejectsNonFiniteOrEmptySamples();
    testTrainingSampleSerialization();
    testModelSerialization();
    testNetworkSerialization();
    testNetworkQuantization();
    testQuantizedNetworkSerialization();
    testMetrics();
    testBenchmarkCsv();
    testMomentumOptimizer();
    testAdamOptimizer();
    testCompressedAdamOptimizer();
    testOptimizerRejectsNonFiniteGradients();
    testOptimizerSerialization();
    testLearningMemorySerialization();
    testGloomyConfigDefaults();
    testGloomyConfigFile();
    testTrainingRuntime();
    testTrainingRuntimeResumesFromSavedModel();
    testTrainingRuntimePersistencePaths();
    testOnlineLearningRuntimePersistencePaths();
    testOnlineLearningRuntime();
    testOnlineLearningRuntimeWindowSize();
    testOnlineLearningRuntimeConceptDriftDetection();
    testOnlineLearningRuntimeLongSequenceStability();
    testConceptDriftDetector();
    testPageHinkleyDetector();
    testCatastrophicForgettingWithoutReplay();
    testCatastrophicForgettingMitigatedByReplay();
    testConceptDriftReturnToPreviousRegime();
    return 0;
}
