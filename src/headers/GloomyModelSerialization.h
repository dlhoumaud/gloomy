#ifndef GLOOMY_MODEL_SERIALIZATION_H
#define GLOOMY_MODEL_SERIALIZATION_H

#include "NeuralNetwork.h"
#include "Normalization.h"
#include "Optimizer.h"
#include "LearningMemory.h"
#include <memory>
#include <string>

class GloomyModelSerialization {
public:
    static void saveInference(
        const std::string& path,
        const NeuralNetwork& network,
        const StreamingNormalizer& normalizer
    );
    static void loadInference(
        const std::string& path,
        NeuralNetwork& network,
        StreamingNormalizer& normalizer
    );
    static bool isInferenceArtifact(const std::string& path);
    static void save(
        const std::string& path,
        const NeuralNetwork& network,
        const StreamingNormalizer& normalizer,
        const Optimizer& optimizer,
        const LearningMemory* memory = nullptr
    );
    static void load(
        const std::string& path,
        NeuralNetwork& network,
        StreamingNormalizer& normalizer,
        Optimizer& optimizer,
        std::unique_ptr<LearningMemory>* memory = nullptr
    );
};

#endif
