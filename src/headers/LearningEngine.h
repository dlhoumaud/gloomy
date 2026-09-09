#ifndef LEARNING_ENGINE_H
#define LEARNING_ENGINE_H

#include "ImportanceScorer.h"
#include "LossFunction.h"
#include "LearningMemory.h"
#include "NeuralNetwork.h"
#include "Optimizer.h"
#include "TrainingSample.h"
#include "TrainingScheduler.h"
#include <vector>

class LearningEngine {
public:
    // importance_weights (defaut : error=1.0, tout le reste a 0.0, soit
    // exactement le comportement historique) pondere les cinq composantes
    // du score d'importance calculees par learn()/trainFromMemory() — voir
    // docs/memory.md, « Score d'importance ».
    LearningEngine(
        NeuralNetwork& network,
        const LossFunction& loss,
        Optimizer& optimizer,
        ImportanceWeights importance_weights = {}
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
    ImportanceScorer scorer;
};

#endif // LEARNING_ENGINE_H
