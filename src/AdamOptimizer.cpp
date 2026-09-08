#include "headers/AdamOptimizer.h"
#include <cmath>
#include <stdexcept>
#include <utility>

AdamOptimizer::AdamOptimizer(
    double learning_rate,
    double beta1,
    double beta2,
    double epsilon
)
    : learning_rate(learning_rate),
      first_decay(beta1),
      second_decay(beta2),
      epsilon(epsilon) {
    if (learning_rate <= 0.0 || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("Learning rate must be finite and positive");
    }
    if (beta1 < 0.0 || beta1 >= 1.0 || beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("Adam beta values must be in [0, 1)");
    }
    if (epsilon <= 0.0 || !std::isfinite(epsilon)) {
        throw std::invalid_argument("Adam epsilon must be finite and positive");
    }
}

void AdamOptimizer::update(
    std::vector<DenseLayer>& layers,
    double gradient_scale
) {
    if (gradient_scale <= 0.0 || !std::isfinite(gradient_scale)) {
        throw std::invalid_argument("Gradient scale must be finite and positive");
    }
    validateFiniteGradients(layers);
    ensureState(layers);
    ++update_count;

    const double first_correction = 1.0 - std::pow(first_decay, static_cast<double>(update_count));
    const double second_correction = 1.0 - std::pow(second_decay, static_cast<double>(update_count));

    for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
        DenseLayer& layer = layers[layer_index];
        LayerState& state = states[layer_index];
        auto& parameters = layer.weights();
        const auto& gradients = layer.weightGradients();
        for (size_t input_index = 0; input_index < parameters.size(); ++input_index) {
            for (size_t output_index = 0; output_index < parameters[input_index].size(); ++output_index) {
                const double gradient = gradient_scale * gradients[input_index][output_index];
                double& first = state.first_moment[input_index][output_index];
                double& second = state.second_moment[input_index][output_index];
                first = first_decay * first + (1.0 - first_decay) * gradient;
                second = second_decay * second + (1.0 - second_decay) * gradient * gradient;
                const double corrected_first = first / first_correction;
                const double corrected_second = second / second_correction;
                parameters[input_index][output_index] -=
                    learning_rate * corrected_first / (std::sqrt(corrected_second) + epsilon);
            }
        }

        auto& biases = layer.bias();
        const auto& bias_gradients = layer.biasGradients();
        for (size_t output_index = 0; output_index < biases.size(); ++output_index) {
            const double gradient = gradient_scale * bias_gradients[output_index];
            double& first = state.bias_first_moment[output_index];
            double& second = state.bias_second_moment[output_index];
            first = first_decay * first + (1.0 - first_decay) * gradient;
            second = second_decay * second + (1.0 - second_decay) * gradient * gradient;
            const double corrected_first = first / first_correction;
            const double corrected_second = second / second_correction;
            biases[output_index] -=
                learning_rate * corrected_first / (std::sqrt(corrected_second) + epsilon);
        }
    }
}

double AdamOptimizer::learningRate() const {
    return learning_rate;
}

double AdamOptimizer::beta1() const {
    return first_decay;
}

double AdamOptimizer::beta2() const {
    return second_decay;
}

size_t AdamOptimizer::stateBytes() const {
    size_t count = 0;
    for (const LayerState& state : states) {
        for (const auto& row : state.first_moment) {
            count += row.size();
        }
        for (const auto& row : state.second_moment) {
            count += row.size();
        }
        count += state.bias_first_moment.size() + state.bias_second_moment.size();
    }
    return count * sizeof(double);
}

void AdamOptimizer::ensureState(const std::vector<DenseLayer>& layers) {
    if (states.size() == layers.size()) {
        bool shape_matches = true;
        for (size_t index = 0; index < layers.size(); ++index) {
            if (states[index].first_moment.size() != layers[index].weights().size() ||
                states[index].bias_first_moment.size() != layers[index].bias().size()) {
                shape_matches = false;
                break;
            }
        }
        if (shape_matches) {
            return;
        }
    }

    states.clear();
    update_count = 0;
    states.reserve(layers.size());
    for (const DenseLayer& layer : layers) {
        LayerState state;
        const size_t input_count = layer.weights().size();
        const size_t output_count = layer.bias().size();
        state.first_moment.resize(input_count, std::vector<double>(output_count, 0.0));
        state.second_moment.resize(input_count, std::vector<double>(output_count, 0.0));
        state.bias_first_moment.assign(output_count, 0.0);
        state.bias_second_moment.assign(output_count, 0.0);
        states.push_back(std::move(state));
    }
}
