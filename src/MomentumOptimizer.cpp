#include "headers/MomentumOptimizer.h"
#include <stdexcept>
#include <utility>
#include <cmath>
#include <cstdint>

MomentumOptimizer::MomentumOptimizer(double learning_rate, double momentum)
    : learning_rate(learning_rate), momentum_factor(momentum) {
    if (learning_rate <= 0.0 || !std::isfinite(learning_rate)) {
        throw std::invalid_argument("Learning rate must be finite and positive");
    }
    if (!std::isfinite(momentum) || momentum < 0.0 || momentum >= 1.0) {
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
        auto& parameters = layer.weightValues();
        const auto& gradients = layer.weightGradientValues();
        const size_t output_count = layer.bias().size();
        for (size_t input_index = 0; input_index < layer.weights().size(); ++input_index) {
            for (size_t output_index = 0; output_index < output_count; ++output_index) {
                const size_t weight_index = input_index * output_count + output_index;
                const double gradient = gradient_scale * gradients[weight_index];
                state.weight_velocity[input_index][output_index] =
                    momentum_factor * state.weight_velocity[input_index][output_index] + gradient;
                parameters[weight_index] -=
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

std::string MomentumOptimizer::checkpointType() const { return "momentum"; }

void MomentumOptimizer::writeCheckpoint(std::ostream& stream) const {
    auto write = [&stream](const void* data, size_t size) {
        stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!stream) throw std::runtime_error("Unable to write optimizer checkpoint");
    };
    write(&learning_rate, sizeof(learning_rate));
    write(&momentum_factor, sizeof(momentum_factor));
    const std::uint64_t count = states.size();
    write(&count, sizeof(count));
    for (const LayerState& state : states) {
        const std::uint64_t inputs = state.weight_velocity.size();
        const std::uint64_t outputs = state.bias_velocity.size();
        write(&inputs, sizeof(inputs)); write(&outputs, sizeof(outputs));
        for (const auto& row : state.weight_velocity) write(row.data(), row.size() * sizeof(double));
        write(state.bias_velocity.data(), state.bias_velocity.size() * sizeof(double));
    }
}

void MomentumOptimizer::readCheckpoint(std::istream& stream, const std::vector<DenseLayer>& layers) {
    auto read = [&stream](void* data, size_t size) {
        stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
        if (!stream) throw std::runtime_error("Invalid or truncated optimizer checkpoint");
    };
    double rate = 0.0, factor = 0.0; std::uint64_t count = 0;
    read(&rate, sizeof(rate)); read(&factor, sizeof(factor)); read(&count, sizeof(count));
    if (!std::isfinite(rate) || rate <= 0.0 || !std::isfinite(factor) || factor < 0.0 || factor >= 1.0 || count != layers.size())
        throw std::runtime_error("Optimizer checkpoint configuration or layer count mismatch");
    std::vector<LayerState> restored; restored.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        std::uint64_t inputs = 0, outputs = 0; read(&inputs, sizeof(inputs)); read(&outputs, sizeof(outputs));
        if (inputs != layers[i].weights().size() || outputs != layers[i].bias().size())
            throw std::runtime_error("Optimizer checkpoint layer shape mismatch");
        LayerState state; state.weight_velocity.resize(inputs, std::vector<double>(outputs)); state.bias_velocity.resize(outputs);
        for (auto& row : state.weight_velocity) read(row.data(), row.size() * sizeof(double));
        read(state.bias_velocity.data(), state.bias_velocity.size() * sizeof(double));
        for (const auto& row : state.weight_velocity) for (double v : row) if (!std::isfinite(v)) throw std::runtime_error("Non-finite optimizer state");
        for (double v : state.bias_velocity) if (!std::isfinite(v)) throw std::runtime_error("Non-finite optimizer state");
        restored.push_back(std::move(state));
    }
    if (stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected optimizer checkpoint data");
    learning_rate = rate; momentum_factor = factor; states = std::move(restored);
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
