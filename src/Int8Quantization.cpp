#include "headers/Int8Quantization.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
constexpr double minimum_scale = 1e-12;
constexpr double int8_min = -128.0;
constexpr double int8_max = 127.0;

void validateParameters(const QuantizationParameters& parameters) {
    if (!std::isfinite(parameters.scale) || parameters.scale <= 0.0) {
        throw std::invalid_argument("Quantization scale must be finite and positive");
    }
}
}

QuantizationParameters Int8Quantizer::calibrate(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot calibrate an empty vector");
    }
    const auto bounds = std::minmax_element(values.begin(), values.end());
    if (!std::isfinite(*bounds.first) || !std::isfinite(*bounds.second)) {
        throw std::invalid_argument("Quantization values must be finite");
    }
    const double range = *bounds.second - *bounds.first;
    if (range == 0.0) {
        return {std::max(std::abs(*bounds.first) / int8_max, minimum_scale), 0};
    }
    const double scale = std::max(range / (int8_max - int8_min), minimum_scale);
    const double zero_point = std::clamp(
        std::round(int8_min - *bounds.first / scale), int8_min, int8_max
    );
    return {scale, static_cast<std::int16_t>(zero_point)};
}

Int8QuantizedVector Int8Quantizer::quantize(
    const std::vector<double>& values,
    const QuantizationParameters& parameters
) {
    Int8QuantizedVector result;
    quantizeInto(values, parameters, result);
    return result;
}

void Int8Quantizer::quantizeInto(const std::vector<double>& values,
    const QuantizationParameters& parameters, Int8QuantizedVector& result) {
    validateParameters(parameters);
    result.parameters = parameters;
    result.values.resize(values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Quantization values must be finite");
        }
        const double quantized = std::clamp(
            std::round(value / parameters.scale + parameters.zero_point),
            int8_min,
            int8_max
        );
        result.values[index] = static_cast<std::int8_t>(quantized);
    }
}

std::vector<double> Int8Quantizer::dequantize(const Int8QuantizedVector& quantized) {
    std::vector<double> result;
    dequantizeInto(quantized, result);
    return result;
}

void Int8Quantizer::dequantizeInto(const Int8QuantizedVector& quantized, std::vector<double>& output) {
    validateParameters(quantized.parameters);
    output.resize(quantized.values.size());
    for (size_t index = 0; index < quantized.values.size(); ++index) {
        output[index] = (static_cast<double>(quantized.values[index]) - quantized.parameters.zero_point) *
            quantized.parameters.scale;
    }
}
