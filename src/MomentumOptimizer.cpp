#include "headers/MomentumOptimizer.h"
#include <stdexcept>
#include <utility>

MomentumOptimizer::MomentumOptimizer(double learning_rate, double momentum)
    : learning_rate(learning_rate), momentum_factor(momentum) {
    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (momentum < 0.0 || momentum >= 1.0) {
        throw std::invalid_argument("Momentum must be in [0, 1)");
    }
}

void MomentumOptimizer::update(
    std::vector<DenseLayer>& layers,
    double gradient_scale
) {
    if (gradient_scale <= 0.0) {
        throw std::invalid_argument("Gradient scale must be positive");
    }
    ensureState(layers);

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        DenseLayer& layer = layers[layer_index];
        LayerState& state = states[layer_index];
        auto& parameters = layer.weights();
        const auto& gradients = layer.weightGradients();
        for (size_t input_index = 0; input_index < parameters.size(); ++input_index) {
            for (size_t output_index = 0; output_index < parameters[input_index].size(); ++output_index) {
                const double gradient = gradient_scale * gradients[input_index][output_index];
                state.weight_velocity[input_index][output_index] =
                    momentum_factor * state.weight_velocity[input_index][output_index] + gradient;
                parameters[input_index][output_index] -=
                    learning_rate * state.weight_velocity[input_index][output_index];
            }
        }

        auto& biases = layer.bias();
        const auto& bias_gradients = layer.biasGradients();
        for (size_t output_index = 0; output_index < biases.size(); ++output_index) {
            const double gradient = gradient_scale * bias_gradients[output_index];
            state.bias_velocity[output_index] =
                momentum_factor * state.bias_velocity[output_index] + gradient;
            biases[output_index] -= learning_rate * state.bias_velocity[output_index];
        }
    }
}

double MomentumOptimizer::learningRate() const {
    return learning_rate;
}

double MomentumOptimizer::momentum() const {
    return momentum_factor;
}

size_t MomentumOptimizer::stateBytes() const {
    size_t count = 0;
    for (const LayerState& state : states) {
        for (const auto& row : state.weight_velocity) {
            count += row.size();
        }
        count += state.bias_velocity.size();
    }
    return count * sizeof(double);
}

void MomentumOptimizer::ensureState(const std::vector<DenseLayer>& layers) {
    if (states.size() == layers.size()) {
        bool shape_matches = true;
        for (size_t index = 0; index < layers.size(); ++index) {
            if (states[index].weight_velocity.size() != layers[index].weights().size() ||
                states[index].bias_velocity.size() != layers[index].bias().size()) {
                shape_matches = false;
                break;
            }
        }
        if (shape_matches) {
            return;
        }
    }

    states.clear();
    states.reserve(layers.size());
    for (const DenseLayer& layer : layers) {
        LayerState state;
        state.weight_velocity.resize(
            layer.weights().size(),
            std::vector<double>(layer.bias().size(), 0.0)
        );
        state.bias_velocity.assign(layer.bias().size(), 0.0);
        states.push_back(std::move(state));
    }
}
