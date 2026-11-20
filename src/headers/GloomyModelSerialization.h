#ifndef GLOOMY_MODEL_SERIALIZATION_H
#define GLOOMY_MODEL_SERIALIZATION_H

#include "NeuralNetwork.h"
#include "Normalization.h"
#include "Optimizer.h"
#include <string>

class GloomyModelSerialization {
public:
    static void save(
        const std::string& path,
        const NeuralNetwork& network,
        const StreamingNormalizer& normalizer,
        const Optimizer& optimizer
    );
    static void load(
        const std::string& path,
        NeuralNetwork& network,
        StreamingNormalizer& normalizer,
        Optimizer& optimizer
    );
};

#endif
