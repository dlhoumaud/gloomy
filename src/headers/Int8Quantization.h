#ifndef INT8_QUANTIZATION_H
#define INT8_QUANTIZATION_H

#include "Quantization.h"
#include <cstdint>
#include <vector>

struct Int8QuantizedVector {
    std::vector<std::int8_t> values;
    QuantizationParameters parameters;
};

class Int8Quantizer {
public:
    static QuantizationParameters calibrate(const std::vector<double>& values);
    static Int8QuantizedVector quantize(
        const std::vector<double>& values,
        const QuantizationParameters& parameters
    );
    static void quantizeInto(const std::vector<double>& values,
        const QuantizationParameters& parameters, Int8QuantizedVector& output);
    static std::vector<double> dequantize(const Int8QuantizedVector& quantized);
    static void dequantizeInto(const Int8QuantizedVector& quantized, std::vector<double>& output);
};

#endif // INT8_QUANTIZATION_H
