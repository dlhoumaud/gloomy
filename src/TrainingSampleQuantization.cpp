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
    QuantizedTrainingSample result;
    encodeInto(sample, result);
    return result;
}

void TrainingSampleQuantizer::encodeInto(const TrainingSample& sample, QuantizedTrainingSample& output) const {
    Int16Quantizer::quantizeInto(sample.input, input_parameters, output.input);
    Int16Quantizer::quantizeInto(sample.target, target_parameters, output.target);
    output.priority = sample.priority;
    output.error = sample.error;
    output.novelty = sample.novelty;
    output.rarity = sample.rarity;
    output.recency = sample.recency;
    output.diversity = sample.diversity;
    output.age = sample.age;
    output.usage_count = sample.usage_count;
}

TrainingSample TrainingSampleQuantizer::decode(
    const QuantizedTrainingSample& sample
) const {
    TrainingSample result;
    decodeInto(sample, result);
    return result;
}

void TrainingSampleQuantizer::decodeInto(const QuantizedTrainingSample& sample, TrainingSample& result) const {
    Int16Quantizer::dequantizeInto(sample.input, result.input);
    Int16Quantizer::dequantizeInto(sample.target, result.target);
    result.priority = sample.priority;
    result.error = sample.error;
    result.novelty = sample.novelty;
    result.rarity = sample.rarity;
    result.recency = sample.recency;
    result.diversity = sample.diversity;
    result.age = sample.age;
    result.usage_count = sample.usage_count;
}

const QuantizationParameters& TrainingSampleQuantizer::inputParameters() const {
    return input_parameters;
}

const QuantizationParameters& TrainingSampleQuantizer::targetParameters() const {
    return target_parameters;
}
