#include "headers/AdamOptimizer.h"
#include <cmath>
#include <stdexcept>
#include <utility>
#include <cstdint>

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
    if (gradient_scale <= 0.0) {
        throw std::invalid_argument("Gradient scale must be positive");
    }
    ensureState(layers);
    ++update_count;

    const double first_correction = 1.0 - std::pow(first_decay, static_cast<double>(update_count));
    const double second_correction = 1.0 - std::pow(second_decay, static_cast<double>(update_count));

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
                double& first = state.first_moment[input_index][output_index];
                double& second = state.second_moment[input_index][output_index];
                first = first_decay * first + (1.0 - first_decay) * gradient;
                second = second_decay * second + (1.0 - second_decay) * gradient * gradient;
                const double corrected_first = first / first_correction;
                const double corrected_second = second / second_correction;
                parameters[weight_index] -=
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

std::string AdamOptimizer::checkpointType() const { return "adam"; }

void AdamOptimizer::writeCheckpoint(std::ostream& stream) const {
    auto write = [&stream](const void* data, size_t size) {
        stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
        if (!stream) throw std::runtime_error("Unable to write optimizer checkpoint");
    };
    write(&learning_rate, sizeof(learning_rate)); write(&first_decay, sizeof(first_decay));
    write(&second_decay, sizeof(second_decay)); write(&epsilon, sizeof(epsilon));
    const std::uint64_t updates = update_count, count = states.size();
    write(&updates, sizeof(updates)); write(&count, sizeof(count));
    for (const LayerState& state : states) {
        const std::uint64_t inputs = state.first_moment.size(), outputs = state.bias_first_moment.size();
        write(&inputs, sizeof(inputs)); write(&outputs, sizeof(outputs));
        for (const auto& row : state.first_moment) write(row.data(), row.size() * sizeof(double));
        for (const auto& row : state.second_moment) write(row.data(), row.size() * sizeof(double));
        write(state.bias_first_moment.data(), outputs * sizeof(double));
        write(state.bias_second_moment.data(), outputs * sizeof(double));
    }
}

void AdamOptimizer::readCheckpoint(std::istream& stream, const std::vector<DenseLayer>& layers) {
    auto read = [&stream](void* data, size_t size) {
        stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
        if (!stream) throw std::runtime_error("Invalid or truncated optimizer checkpoint");
    };
    double rate = 0.0, beta_1 = 0.0, beta_2 = 0.0, eps = 0.0;
    std::uint64_t updates = 0, count = 0;
    read(&rate, sizeof(rate)); read(&beta_1, sizeof(beta_1)); read(&beta_2, sizeof(beta_2)); read(&eps, sizeof(eps));
    read(&updates, sizeof(updates)); read(&count, sizeof(count));
    if (!std::isfinite(rate) || rate <= 0.0 || !std::isfinite(beta_1) || beta_1 < 0.0 || beta_1 >= 1.0 ||
        !std::isfinite(beta_2) || beta_2 < 0.0 || beta_2 >= 1.0 || !std::isfinite(eps) || eps <= 0.0 || count != layers.size())
        throw std::runtime_error("Optimizer checkpoint configuration or layer count mismatch");
    std::vector<LayerState> restored; restored.reserve(layers.size());
    for (size_t i = 0; i < layers.size(); ++i) {
        std::uint64_t inputs = 0, outputs = 0; read(&inputs, sizeof(inputs)); read(&outputs, sizeof(outputs));
        if (inputs != layers[i].weights().size() || outputs != layers[i].bias().size())
            throw std::runtime_error("Optimizer checkpoint layer shape mismatch");
        LayerState state;
        state.first_moment.resize(inputs, std::vector<double>(outputs)); state.second_moment.resize(inputs, std::vector<double>(outputs));
        state.bias_first_moment.resize(outputs); state.bias_second_moment.resize(outputs);
        for (auto& row : state.first_moment) read(row.data(), row.size() * sizeof(double));
        for (auto& row : state.second_moment) read(row.data(), row.size() * sizeof(double));
        read(state.bias_first_moment.data(), outputs * sizeof(double)); read(state.bias_second_moment.data(), outputs * sizeof(double));
        auto finite = [](const std::vector<double>& values) { for (double value : values) if (!std::isfinite(value)) return false; return true; };
        for (const auto& row : state.first_moment) if (!finite(row)) throw std::runtime_error("Non-finite optimizer state");
        for (const auto& row : state.second_moment) if (!finite(row)) throw std::runtime_error("Non-finite optimizer state");
        if (!finite(state.bias_first_moment) || !finite(state.bias_second_moment)) throw std::runtime_error("Non-finite optimizer state");
        restored.push_back(std::move(state));
    }
    if (stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected optimizer checkpoint data");
    learning_rate = rate; first_decay = beta_1; second_decay = beta_2; epsilon = eps;
    update_count = static_cast<size_t>(updates); states = std::move(restored);
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
