#include "headers/Optimizer.h"
#include <stdexcept>

SGDOptimizer::SGDOptimizer(double learning_rate)
    : learning_rate(learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
}

void SGDOptimizer::update(std::vector<DenseLayer>& layers, double gradient_scale) {
    if (gradient_scale <= 0.0) {
        throw std::invalid_argument("Gradient scale must be positive");
    }

    for (auto& layer : layers) {
        auto& parameters = layer.weights();
        const auto& gradients = layer.weightGradients();
        for (size_t input_index = 0; input_index < parameters.size(); ++input_index) {
            for (size_t output_index = 0; output_index < parameters[input_index].size(); ++output_index) {
                parameters[input_index][output_index] -=
                    learning_rate * gradient_scale * gradients[input_index][output_index];
            }
        }

        auto& biases = layer.bias();
        const auto& bias_gradients = layer.biasGradients();
        for (size_t output_index = 0; output_index < biases.size(); ++output_index) {
            biases[output_index] -= learning_rate * gradient_scale * bias_gradients[output_index];
        }
    }
}

double SGDOptimizer::learningRate() const {
    return learning_rate;
}

void SGDOptimizer::setLearningRate(double learning_rate) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    this->learning_rate = learning_rate;
}
