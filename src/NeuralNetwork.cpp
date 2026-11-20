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
    return forwardRef(inputs);
}

const std::vector<double>& NeuralNetwork::forwardRef(const std::vector<double>& inputs) {
    return forwardImpl(inputs, true);
}

const std::vector<double>& NeuralNetwork::forwardInferenceRef(const std::vector<double>& inputs) {
    return forwardImpl(inputs, false);
}

const std::vector<double>& NeuralNetwork::forwardImpl(const std::vector<double>& inputs, bool cache_inputs) {
    if (layers_data.empty()) {
        throw std::runtime_error("The network has no layers");
    }

    const std::vector<double>* current_inputs = &inputs;
    for (size_t index = 0; index < layers_data.size(); ++index) {
        auto &layer = layers_data[index];
        layer.set_algorithm(algorithm);
        // Softmax is a normalization of the final output, not a hidden-layer activation.
        layer.set_post_algorithm(index + 1 == layers_data.size() ? post_algorithm : "none");
        current_inputs = cache_inputs
            ? &layer.forwardRef(*current_inputs)
            : &layer.forwardInferenceRef(*current_inputs);
    }

    if (current_inputs->empty()) {
        throw std::runtime_error("The network produced no output");
    }

    return *current_inputs;
}

std::vector<double> NeuralNetwork::backward(const std::vector<double>& gradient_output) {
    return backwardRef(gradient_output);
}

const std::vector<double>& NeuralNetwork::backwardRef(const std::vector<double>& gradient_output) {
    if (layers_data.empty()) {
        throw std::runtime_error("The network has no layers");
    }

    const std::vector<double>* gradient = &gradient_output;
    for (auto layer = layers_data.rbegin(); layer != layers_data.rend(); ++layer) {
        gradient = &layer->backwardRef(*gradient);
    }
    return *gradient;
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

std::size_t NeuralNetwork::allocatedBytes() const {
    std::size_t bytes = sizeof(*this) + layers_data.capacity() * sizeof(DenseLayer);
    for (const DenseLayer& layer : layers_data) bytes += layer.allocatedBytes();
    return bytes;
}

