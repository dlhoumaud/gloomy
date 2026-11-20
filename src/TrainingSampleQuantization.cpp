#include "headers/TrainingSampleQuantization.h"
#include <algorithm>
#include <cmath>
#include <limits>
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

    return TrainingSampleQuantizer(
        Int16Quantizer::calibrateRange(input_minimum, input_maximum),
        Int16Quantizer::calibrateRange(target_minimum, target_maximum)
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
