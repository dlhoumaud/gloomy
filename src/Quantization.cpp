#include "headers/Quantization.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {
constexpr double minimum_scale = 1e-12;
constexpr double int16_min = -32768.0;
constexpr double int16_max = 32767.0;

void validateParameters(const QuantizationParameters& parameters) {
    if (!std::isfinite(parameters.scale) || parameters.scale <= 0.0) {
        throw std::invalid_argument("Quantization scale must be finite and positive");
    }
}
}

QuantizationParameters Int16Quantizer::calibrate(const std::vector<double>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot calibrate an empty vector");
    }

    const auto bounds = std::minmax_element(values.begin(), values.end());
    if (!std::isfinite(*bounds.first) || !std::isfinite(*bounds.second)) {
        throw std::invalid_argument("Quantization values must be finite");
    }

    const double range = *bounds.second - *bounds.first;
    if (range == 0.0) {
        const double scale = std::max(
            std::abs(*bounds.first) / int16_max,
            minimum_scale
        );
        return {scale, 0};
    }
    const double scale = std::max(range / (int16_max - int16_min), minimum_scale);
    const double zero_point_value = int16_min - (*bounds.first / scale);
    const double clamped_zero_point = std::clamp(
        std::round(zero_point_value),
        int16_min,
        int16_max
    );
    return {scale, static_cast<std::int16_t>(clamped_zero_point)};
}

QuantizedVector Int16Quantizer::quantize(
    const std::vector<double>& values,
    const QuantizationParameters& parameters
) {
    QuantizedVector result;
    quantizeInto(values, parameters, result);
    return result;
}

void Int16Quantizer::quantizeInto(const std::vector<double>& values,
    const QuantizationParameters& parameters, QuantizedVector& result) {
    validateParameters(parameters);
    result.parameters = parameters;
    result.values.resize(values.size());

    for (size_t index = 0; index < values.size(); ++index) {
        const double value = values[index];
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Quantization values must be finite");
        }
        const double raw = std::round(value / parameters.scale + parameters.zero_point);
        const double clamped = std::clamp(raw, int16_min, int16_max);
        result.values[index] = static_cast<std::int16_t>(clamped);
    }
}

std::vector<double> Int16Quantizer::dequantize(const QuantizedVector& quantized) {
    std::vector<double> result;
    dequantizeInto(quantized, result);
    return result;
}

void Int16Quantizer::dequantizeInto(const QuantizedVector& quantized, std::vector<double>& output) {
    validateParameters(quantized.parameters);
    output.resize(quantized.values.size());
    for (size_t index = 0; index < quantized.values.size(); ++index) {
        output[index] =
            (static_cast<double>(quantized.values[index]) - quantized.parameters.zero_point) *
            quantized.parameters.scale;
    }
}
