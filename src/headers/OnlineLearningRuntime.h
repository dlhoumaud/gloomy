#ifndef ONLINE_LEARNING_RUNTIME_H
#define ONLINE_LEARNING_RUNTIME_H

#include "GloomyConfig.h"
#include "NeuralNetwork.h"
#include "Normalization.h"
#include "Optimizer.h"
#include "LearningMemory.h"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// Une observation traitée par le runtime online, avant la mise à jour des
// poids qu'elle a déclenchée (ou non, selon le scheduler).
struct OnlineLearningStep {
    double observation = 0.0;
    double target = 0.0;
    double prediction_before_update = 0.0;
    double loss_before_update = 0.0;
};

struct OnlineLearningResult {
    std::vector<OnlineLearningStep> steps;
    double average_loss = 0.0;
    std::size_t memory_size = 0;
    NeuralNetwork network;
    std::unique_ptr<StreamingNormalizer> normalizer;
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<LearningMemory> memory;
};

struct TrainingResult {
    double average_loss = 0.0;
    NeuralNetwork network;
    std::unique_ptr<StreamingNormalizer> normalizer;
    std::unique_ptr<Optimizer> optimizer;
    std::unique_ptr<LearningMemory> memory;
};

// Exécute le ONLINE_LEARNING_RUNTIME décrit dans docs/roadmap.md : chaque
// valeur consécutive de `sequence` devient une observation scalaire
// (sequence[i]) et sa cible (sequence[i + 1]), et la boucle
//   observation -> normalisation -> prédiction -> cible
//   -> erreur -> mémoire -> scheduler -> replay -> mise à jour
// (LearningEngine::learn) est répétée pour chaque paire consécutive.
//
// Le réseau, la perte, l'optimiseur, la mémoire d'apprentissage et le
// scheduler sont construits à partir de `config`. `sequence` doit contenir
// au moins deux valeurs.
OnlineLearningResult runOnlineLearning(
    const GloomyConfig& config,
    const std::vector<double>& sequence
);

OnlineLearningResult runOnlineLearning(
    const GloomyConfig& config,
    const std::vector<double>& sequence,
    const std::string& model_path
);

TrainingResult runTraining(
    const GloomyConfig& config,
    const std::vector<double>& sequence
);

#endif // ONLINE_LEARNING_RUNTIME_H
