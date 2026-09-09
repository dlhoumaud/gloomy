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

    // Meme correction que Int16Quantizer::calibrate (voir Quantization.cpp
    // pour l'explication complete du bug corrige) : la plage de calibration
    // est etendue pour toujours inclure 0, ce qui garantit que le
    // zero_point calcule ci-dessous reste dans [int8_min, int8_max] sans
    // jamais avoir besoin d'etre sature (sans quoi une plage ne contenant
    // pas 0 ecrasait toutes les valeurs vers le meme code quantifie).
    const double effective_min = std::min(*bounds.first, 0.0);
    const double effective_max = std::max(*bounds.second, 0.0);

    const double range = effective_max - effective_min;
    const double scale = std::max(range / (int8_max - int8_min), minimum_scale);
    const double zero_point = std::clamp(
        std::round(int8_min - effective_min / scale), int8_min, int8_max
    );
    return {scale, static_cast<std::int16_t>(zero_point)};
}

Int8QuantizedVector Int8Quantizer::quantize(
    const std::vector<double>& values,
    const QuantizationParameters& parameters
) {
    validateParameters(parameters);
    Int8QuantizedVector result{{}, parameters};
    result.values.reserve(values.size());
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Quantization values must be finite");
        }
        const double quantized = std::clamp(
            std::round(value / parameters.scale + parameters.zero_point),
            int8_min,
            int8_max
        );
        result.values.push_back(static_cast<std::int8_t>(quantized));
    }
    return result;
}

std::vector<double> Int8Quantizer::dequantize(const Int8QuantizedVector& quantized) {
    validateParameters(quantized.parameters);
    std::vector<double> result;
    result.reserve(quantized.values.size());
    for (std::int8_t value : quantized.values) {
        result.push_back(
            (static_cast<double>(value) - quantized.parameters.zero_point) *
            quantized.parameters.scale
        );
    }
    return result;
}