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
    size_t allocatedBytes() const;

private:
    double trainBatchRange(
        const std::vector<TrainingSample>& samples,
        size_t offset,
        size_t count
    );

    NeuralNetwork& network;
    const LossFunction& loss;
    Optimizer& optimizer;
    std::vector<double> gradient_buffer;
    std::vector<MemoryEntry> replay_entries;
};

#endif // LEARNING_ENGINE_H
