#ifndef INT8_TRAINING_SAMPLE_QUANTIZATION_H
#define INT8_TRAINING_SAMPLE_QUANTIZATION_H

#include "Int8Quantization.h"
#include "TrainingSample.h"
#include <vector>

struct Int8QuantizedTrainingSample {
    Int8QuantizedVector input;
    Int8QuantizedVector target;
    double priority;
    double error;
    double novelty;
    double rarity;
    double recency;
    double diversity;
    std::size_t age;
    std::size_t usage_count;
};

class Int8TrainingSampleQuantizer {
public:
    Int8TrainingSampleQuantizer(
        QuantizationParameters input_parameters,
        QuantizationParameters target_parameters
    );

    static Int8TrainingSampleQuantizer calibrate(
        const std::vector<TrainingSample>& samples
    );

    Int8QuantizedTrainingSample encode(const TrainingSample& sample) const;
    TrainingSample decode(const Int8QuantizedTrainingSample& sample) const;

    const QuantizationParameters& inputParameters() const;
    const QuantizationParameters& targetParameters() const;

private:
    QuantizationParameters input_parameters;
    QuantizationParameters target_parameters;
};

#endif // INT8_TRAINING_SAMPLE_QUANTIZATION_H
