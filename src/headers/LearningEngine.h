#ifndef LEARNING_ENGINE_H
#define LEARNING_ENGINE_H

#include "LossFunction.h"
#include "LearningMemory.h"
#include "NeuralNetwork.h"
#include "Optimizer.h"
#include "TrainingSample.h"
#include "TrainingScheduler.h"
#include <vector>

class LearningEngine {
public:
    LearningEngine(
        NeuralNetwork& network,
        const LossFunction& loss,
        Optimizer& optimizer
    );

    double trainBatch(const std::vector<TrainingSample>& batch);
    double trainFromMemory(LearningMemory& memory, size_t batch_size);
    double learn(
        LearningMemory& memory,
        const TrainingSample& sample,
        size_t batch_size
    );
    double learn(
        LearningMemory& memory,
        const TrainingSample& sample,
        size_t batch_size,
        TrainingScheduler& scheduler
    );
    double train(
        const std::vector<TrainingSample>& samples,
        size_t epochs,
        size_t batch_size
    );

private:
    // Comme trainBatch, mais met a l'echelle la contribution au gradient de
    // chaque echantillon par son poids (correction du biais
    // d'echantillonnage, voir MemoryEntry::importance_weight et
    // docs/memory.md). Un poids de 1.0 partout reproduit exactement
    // trainBatch ; trainBatch delegue d'ailleurs a cette methode.
    double trainWeightedBatch(
        const std::vector<TrainingSample>& batch,
        const std::vector<double>& sample_weights
    );

    NeuralNetwork& network;
    const LossFunction& loss;
    Optimizer& optimizer;
};

#endif // LEARNING_ENGINE_H
