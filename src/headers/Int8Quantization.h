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
    static std::vector<double> dequantize(const Int8QuantizedVector& quantized);
};

#endif // INT8_QUANTIZATION_H
