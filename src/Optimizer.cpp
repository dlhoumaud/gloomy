#include "headers/Optimizer.h"
#include <stdexcept>
#include <cmath>

SGDOptimizer::SGDOptimizer(double learning_rate)
    : learning_rate(learning_rate) {
    if (learning_rate <= 0.0 || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("Learning rate must be positive");
    }
}

void SGDOptimizer::update(std::vector<DenseLayer>& layers, double gradient_scale) {
    if (gradient_scale <= 0.0) {
        throw std::invalid_argument("Gradient scale must be positive");
    }

    for (auto& layer : layers) {
        auto& parameters = layer.weightValues();
        const auto& gradients = layer.weightGradientValues();
        for (size_t index = 0; index < parameters.size(); ++index) {
            parameters[index] -= learning_rate * gradient_scale * gradients[index];
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
    if (learning_rate <= 0.0 || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    this->learning_rate = learning_rate;
}

std::string SGDOptimizer::checkpointType() const { return "sgd"; }

void SGDOptimizer::writeCheckpoint(std::ostream& stream) const {
    stream.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    if (!stream) throw std::runtime_error("Unable to write optimizer checkpoint");
}

void SGDOptimizer::readCheckpoint(std::istream& stream, const std::vector<DenseLayer>&) {
    double restored_rate = 0.0;
    stream.read(reinterpret_cast<char*>(&restored_rate), sizeof(restored_rate));
    if (!stream || restored_rate <= 0.0 || !std::isfinite(restored_rate))
        throw std::runtime_error("Invalid SGD optimizer checkpoint");
    if (stream.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Unexpected optimizer checkpoint data");
    learning_rate = restored_rate;
}

std::size_t SGDOptimizer::allocatedBytes() const {
    return sizeof(*this);
}
