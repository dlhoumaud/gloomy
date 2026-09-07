#ifndef QUANTIZATION_H
#define QUANTIZATION_H

#include <cstdint>
#include <vector>

struct QuantizationParameters {
    double scale;
    std::int16_t zero_point;
};

struct QuantizedVector {
    std::vector<std::int16_t> values;
    QuantizationParameters parameters;
};

class Int16Quantizer {
public:
    static QuantizationParameters calibrate(const std::vector<double>& values);
    static QuantizedVector quantize(
        const std::vector<double>& values,
        const QuantizationParameters& parameters
    );
    static std::vector<double> dequantize(const QuantizedVector& quantized);
};

#endif // QUANTIZATION_H
