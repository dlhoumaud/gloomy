#ifndef MODEL_SERIALIZATION_H
#define MODEL_SERIALIZATION_H

#include "NeuralNetwork.h"
#include "Normalization.h"
#include "Optimizer.h"
#include "LearningMemory.h"
#include <memory>
#include <string>

class ModelSerialization {
public:
    struct LoadedModel {
        NeuralNetwork network;
        std::unique_ptr<StreamingNormalizer> normalizer;
        std::unique_ptr<Optimizer> optimizer;
        std::unique_ptr<LearningMemory> memory;
    };

    static void save(
        const std::string& path,
        const NeuralNetwork& network,
        const StreamingNormalizer& normalizer,
        const Optimizer& optimizer,
        const LearningMemory& memory
    );

    static LoadedModel load(const std::string& path);
};

#endif // MODEL_SERIALIZATION_H
