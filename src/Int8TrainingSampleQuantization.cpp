#include "headers/Int8TrainingSampleQuantization.h"
#include <stdexcept>

Int8TrainingSampleQuantizer::Int8TrainingSampleQuantizer(
    QuantizationParameters input_parameters,
    QuantizationParameters target_parameters
)
    : input_parameters(input_parameters), target_parameters(target_parameters) {}

Int8TrainingSampleQuantizer Int8TrainingSampleQuantizer::calibrate(
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
    return Int8TrainingSampleQuantizer(
        Int8Quantizer::calibrate(inputs),
        Int8Quantizer::calibrate(targets)
    );
}

Int8QuantizedTrainingSample Int8TrainingSampleQuantizer::encode(const TrainingSample& sample) const {
    Int8QuantizedTrainingSample result;
    encodeInto(sample, result);
    return result;
}

void Int8TrainingSampleQuantizer::encodeInto(const TrainingSample& sample, Int8QuantizedTrainingSample& output) const {
    Int8Quantizer::quantizeInto(sample.input, input_parameters, output.input);
    Int8Quantizer::quantizeInto(sample.target, target_parameters, output.target);
    output.priority = sample.priority;
    output.error = sample.error;
    output.novelty = sample.novelty;
    output.rarity = sample.rarity;
    output.recency = sample.recency;
    output.diversity = sample.diversity;
    output.age = sample.age;
    output.usage_count = sample.usage_count;
}

TrainingSample Int8TrainingSampleQuantizer::decode(const Int8QuantizedTrainingSample& sample) const {
    TrainingSample result;
    decodeInto(sample, result);
    return result;
}

void Int8TrainingSampleQuantizer::decodeInto(const Int8QuantizedTrainingSample& sample, TrainingSample& result) const {
    Int8Quantizer::dequantizeInto(sample.input, result.input);
    Int8Quantizer::dequantizeInto(sample.target, result.target);
    result.priority = sample.priority;
    result.error = sample.error;
    result.novelty = sample.novelty;
    result.rarity = sample.rarity;
    result.recency = sample.recency;
    result.diversity = sample.diversity;
    result.age = sample.age;
    result.usage_count = sample.usage_count;
}

const QuantizationParameters& Int8TrainingSampleQuantizer::inputParameters() const {
    return input_parameters;
}

const QuantizationParameters& Int8TrainingSampleQuantizer::targetParameters() const {
    return target_parameters;
}
