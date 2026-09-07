#ifndef TRAINING_SAMPLE_QUANTIZATION_H
#define TRAINING_SAMPLE_QUANTIZATION_H

#include "Quantization.h"
#include "TrainingSample.h"
#include <vector>

struct QuantizedTrainingSample {
    QuantizedVector input;
    QuantizedVector target;
    double priority;
    double error;
    double novelty;
    double rarity;
    double recency;
    double diversity;
    std::size_t age;
    std::size_t usage_count;
};

class TrainingSampleQuantizer {
public:
    TrainingSampleQuantizer(
        QuantizationParameters input_parameters,
        QuantizationParameters target_parameters
    );

    static TrainingSampleQuantizer calibrate(
        const std::vector<TrainingSample>& samples
    );

    QuantizedTrainingSample encode(const TrainingSample& sample) const;
    TrainingSample decode(const QuantizedTrainingSample& sample) const;

    const QuantizationParameters& inputParameters() const;
    const QuantizationParameters& targetParameters() const;

private:
    QuantizationParameters input_parameters;
    QuantizationParameters target_parameters;
};

#endif // TRAINING_SAMPLE_QUANTIZATION_H
