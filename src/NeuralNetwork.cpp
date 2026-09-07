/**
 * @ Author: GloomShade
 * @ Create Time: 2025-01-03 08:20:13
 * @ Modified by: GloomShade
 * @ Modified time: 2025-01-03 12:02:53
 * @ Description:
 */

#include "headers/NeuralNetwork.h"
#include <stdexcept>

void NeuralNetwork::addLayer(int input_size, int output_size) {
    layers.emplace_back(input_size, output_size);
}

double NeuralNetwork::predict(std::vector<double> &sequence) {
    if (layers.empty()) {
        throw std::runtime_error("The network has no layers");
    }

    std::vector<double> inputs = sequence;

    for (size_t index = 0; index < layers.size(); ++index) {
        auto &layer = layers[index];
        layer.set_algorithm(algorithm);
        // Softmax is a normalization of the final output, not a hidden-layer activation.
        layer.set_post_algorithm(index + 1 == layers.size() ? post_algorithm : "none");
        inputs = layer.forward(inputs);
    }

    if (inputs.empty()) {
        throw std::runtime_error("The network produced no output");
    }

    return inputs[0];
}

void NeuralNetwork::clear() {
    layers.clear();
}

