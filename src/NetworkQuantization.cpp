#include "headers/NetworkQuantization.h"
#include <stdexcept>
#include <utility>

QuantizedNetwork NetworkQuantization::quantize(const NeuralNetwork& network) {
    QuantizedNetwork result;
    result.algorithm = network.algorithm;
    result.post_algorithm = network.post_algorithm;
    result.layers.reserve(network.layers().size());

    for (const DenseLayer& layer : network.layers()) {
        const auto& weight_rows = layer.weights();
        const auto& biases = layer.bias();
        if (weight_rows.empty() || biases.empty()) {
            throw std::invalid_argument("Cannot quantize an empty layer");
        }

        QuantizedDenseLayer quantized_layer;
        quantized_layer.input_size = weight_rows.size();
        quantized_layer.output_size = biases.size();

        std::vector<double> flat_weights;
        flat_weights.reserve(quantized_layer.input_size * quantized_layer.output_size);
        for (const auto& row : weight_rows) {
            flat_weights.insert(flat_weights.end(), row.begin(), row.end());
        }

        const QuantizationParameters weight_parameters = Int8Quantizer::calibrate(flat_weights);
        quantized_layer.weights = Int8Quantizer::quantize(flat_weights, weight_parameters);

        const QuantizationParameters bias_parameters = Int8Quantizer::calibrate(biases);
        quantized_layer.bias = Int8Quantizer::quantize(biases, bias_parameters);

        result.layers.push_back(std::move(quantized_layer));
    }
    return result;
}

NeuralNetwork NetworkQuantization::dequantize(const QuantizedNetwork& quantized) {
    NeuralNetwork network;
    network.algorithm = quantized.algorithm;
    network.post_algorithm = quantized.post_algorithm;

    for (const QuantizedDenseLayer& layer : quantized.layers) {
        if (layer.input_size == 0 || layer.output_size == 0) {
            throw std::invalid_argument("Quantized layer dimensions must be positive");
        }
        if (layer.weights.values.size() != layer.input_size * layer.output_size) {
            throw std::invalid_argument("Quantized layer weight count does not match its dimensions");
        }
        if (layer.bias.values.size() != layer.output_size) {
            throw std::invalid_argument("Quantized layer bias count does not match its dimensions");
        }

        network.addLayer(static_cast<int>(layer.input_size), static_cast<int>(layer.output_size));
        DenseLayer& dense_layer = network.layers().back();

        const std::vector<double> flat_weights = Int8Quantizer::dequantize(layer.weights);
        auto& weight_rows = dense_layer.weights();
        for (std::size_t input_index = 0; input_index < layer.input_size; ++input_index) {
            for (std::size_t output_index = 0; output_index < layer.output_size; ++output_index) {
                weight_rows[input_index][output_index] =
                    flat_weights[input_index * layer.output_size + output_index];
            }
        }

        dense_layer.bias() = Int8Quantizer::dequantize(layer.bias);
    }
    return network;
}

std::size_t NetworkQuantization::quantizedBytes(const QuantizedNetwork& quantized) {
    std::size_t bytes = 0;
    for (const QuantizedDenseLayer& layer : quantized.layers) {
        bytes += layer.weights.values.size() * sizeof(std::int8_t) + sizeof(QuantizationParameters);
        bytes += layer.bias.values.size() * sizeof(std::int8_t) + sizeof(QuantizationParameters);
    }
    return bytes;
}
