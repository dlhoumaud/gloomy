#include "headers/TrainingSampleQuantization.h"
#include <stdexcept>

TrainingSampleQuantizer::TrainingSampleQuantizer(
    QuantizationParameters input_parameters,
    QuantizationParameters target_parameters
)
    : input_parameters(input_parameters), target_parameters(target_parameters) {}

TrainingSampleQuantizer TrainingSampleQuantizer::calibrate(
    const std::vector<TrainingSample>& samples
) {
    if (samples.empty()) {
        throw std::invalid_argument("Cannot calibrate an empty sample set");
    }

    std::vector<double> inputs;
    std::vector<double> targets;
    for (const TrainingSample& sample : samples) {
        inputs.insert(inputs.end(), sample.input.begin(), sample.input.end());
        targets.insert(targets.end(), sample.target.begin(), sample.target.end());
    }
    if (inputs.empty() || targets.empty()) {
        throw std::invalid_argument("Training samples must contain input and target values");
    }

    return TrainingSampleQuantizer(
        Int16Quantizer::calibrate(inputs),
        Int16Quantizer::calibrate(targets)
    );
}

QuantizedTrainingSample TrainingSampleQuantizer::encode(
    const TrainingSample& sample
) const {
    return {
        Int16Quantizer::quantize(sample.input, input_parameters),
        Int16Quantizer::quantize(sample.target, target_parameters),
        sample.priority,
        sample.error,
        sample.novelty,
        sample.rarity,
        sample.recency,
        sample.diversity,
        sample.age,
        sample.usage_count
    };
}

TrainingSample TrainingSampleQuantizer::decode(
    const QuantizedTrainingSample& sample
) const {
    TrainingSample result;
    result.input = Int16Quantizer::dequantize(sample.input);
    result.target = Int16Quantizer::dequantize(sample.target);
    result.priority = sample.priority;
    result.error = sample.error;
    result.novelty = sample.novelty;
    result.rarity = sample.rarity;
    result.recency = sample.recency;
    result.diversity = sample.diversity;
    result.age = sample.age;
    result.usage_count = sample.usage_count;
    return result;
}

const QuantizationParameters& TrainingSampleQuantizer::inputParameters() const {
    return input_parameters;
}

const QuantizationParameters& TrainingSampleQuantizer::targetParameters() const {
    return target_parameters;
}
