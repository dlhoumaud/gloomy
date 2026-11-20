#include "headers/Int8TrainingSampleQuantization.h"
#include <algorithm>
#include <cmath>
#include <limits>
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
    double input_minimum = std::numeric_limits<double>::infinity();
    double input_maximum = -std::numeric_limits<double>::infinity();
    double target_minimum = std::numeric_limits<double>::infinity();
    double target_maximum = -std::numeric_limits<double>::infinity();
    for (const TrainingSample& sample : samples) {
        for (double value : sample.input) {
            if (!std::isfinite(value)) throw std::invalid_argument("Quantization values must be finite");
            input_minimum = std::min(input_minimum, value);
            input_maximum = std::max(input_maximum, value);
        }
        for (double value : sample.target) {
            if (!std::isfinite(value)) throw std::invalid_argument("Quantization values must be finite");
            target_minimum = std::min(target_minimum, value);
            target_maximum = std::max(target_maximum, value);
        }
    }
    if (!std::isfinite(input_minimum) || !std::isfinite(target_minimum)) {
        throw std::invalid_argument("Training samples must contain input and target values");
    }
    return Int8TrainingSampleQuantizer(
        Int8Quantizer::calibrateRange(input_minimum, input_maximum),
        Int8Quantizer::calibrateRange(target_minimum, target_maximum)
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
