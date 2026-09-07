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
    layers_data.emplace_back(input_size, output_size);
}

std::vector<double> NeuralNetwork::forward(const std::vector<double>& inputs) {
    if (layers_data.empty()) {
        throw std::runtime_error("The network has no layers");
    }

    std::vector<double> current_inputs = inputs;
    for (size_t index = 0; index < layers_data.size(); ++index) {
        auto &layer = layers_data[index];
        layer.set_algorithm(algorithm);
        // Softmax is a normalization of the final output, not a hidden-layer activation.
        layer.set_post_algorithm(index + 1 == layers_data.size() ? post_algorithm : "none");
        current_inputs = layer.forward(current_inputs);
    }

    if (current_inputs.empty()) {
        throw std::runtime_error("The network produced no output");
    }

    return current_inputs;
}

std::vector<double> NeuralNetwork::backward(const std::vector<double>& gradient_output) {
    if (layers_data.empty()) {
        throw std::runtime_error("The network has no layers");
    }

    std::vector<double> gradient = gradient_output;
    for (auto layer = layers_data.rbegin(); layer != layers_data.rend(); ++layer) {
        gradient = layer->backward(gradient);
    }
    return gradient;
}

void NeuralNetwork::zeroGradients() {
    for (auto& layer : layers_data) {
        layer.zeroGradients();
    }
}

std::vector<DenseLayer>& NeuralNetwork::layers() {
    return layers_data;
}

const std::vector<DenseLayer>& NeuralNetwork::layers() const {
    return layers_data;
}

double NeuralNetwork::predict(const std::vector<double>& sequence) {
    return forward(sequence)[0];
}

void NeuralNetwork::clear() {
    layers_data.clear();
}

