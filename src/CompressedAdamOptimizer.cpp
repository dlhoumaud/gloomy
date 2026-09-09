#include "headers/CompressedAdamOptimizer.h"
#include <cmath>
#include <stdexcept>

namespace {
QuantizedVector quantizeZeros(std::size_t count) {
    const std::vector<double> zeros(count, 0.0);
    const QuantizationParameters parameters = Int16Quantizer::calibrate(zeros);
    return Int16Quantizer::quantize(zeros, parameters);
}
}

CompressedAdamOptimizer::CompressedAdamOptimizer(
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

void CompressedAdamOptimizer::update(
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

        // Dequantifie les moments courants (poids) une seule fois pour ce
        // pas de mise a jour : le calcul Adam lui-meme se fait en double,
        // seul le stockage entre deux appels a update() est compresse.
        std::vector<double> weight_first_moment = Int16Quantizer::dequantize(state.weight_first_moment);
        std::vector<double> weight_second_moment = Int16Quantizer::dequantize(state.weight_second_moment);

        for (size_t input_index = 0; input_index < parameters.size(); ++input_index) {
            for (size_t output_index = 0; output_index < parameters[input_index].size(); ++output_index) {
                const size_t flat_index = input_index * state.output_size + output_index;
                const double gradient = gradient_scale * gradients[input_index][output_index];

                double& first = weight_first_moment[flat_index];
                double& second = weight_second_moment[flat_index];
                first = first_decay * first + (1.0 - first_decay) * gradient;
                second = second_decay * second + (1.0 - second_decay) * gradient * gradient;
                const double corrected_first = first / first_correction;
                const double corrected_second = second / second_correction;
                parameters[input_index][output_index] -=
                    learning_rate * corrected_first / (std::sqrt(corrected_second) + epsilon);
            }
        }

        // Requantifie (recalibre) les moments mis a jour avant de les
        // restocker : la plage des moments evolue au fil de l'entrainement,
        // une recalibration a chaque pas reste donc necessaire (contrairement
        // a QuantizedFIFOMemory, calibree une fois pour toutes a la
        // construction sur un jeu d'echantillons fixe).
        const QuantizationParameters weight_first_parameters = Int16Quantizer::calibrate(weight_first_moment);
        const QuantizationParameters weight_second_parameters = Int16Quantizer::calibrate(weight_second_moment);
        state.weight_first_moment = Int16Quantizer::quantize(weight_first_moment, weight_first_parameters);
        state.weight_second_moment = Int16Quantizer::quantize(weight_second_moment, weight_second_parameters);

        auto& biases = layer.bias();
        const auto& bias_gradients = layer.biasGradients();
        std::vector<double> bias_first_moment = Int16Quantizer::dequantize(state.bias_first_moment);
        std::vector<double> bias_second_moment = Int16Quantizer::dequantize(state.bias_second_moment);

        for (size_t output_index = 0; output_index < biases.size(); ++output_index) {
            const double gradient = gradient_scale * bias_gradients[output_index];
            double& first = bias_first_moment[output_index];
            double& second = bias_second_moment[output_index];
            first = first_decay * first + (1.0 - first_decay) * gradient;
            second = second_decay * second + (1.0 - second_decay) * gradient * gradient;
            const double corrected_first = first / first_correction;
            const double corrected_second = second / second_correction;
            biases[output_index] -=
                learning_rate * corrected_first / (std::sqrt(corrected_second) + epsilon);
        }

        const QuantizationParameters bias_first_parameters = Int16Quantizer::calibrate(bias_first_moment);
        const QuantizationParameters bias_second_parameters = Int16Quantizer::calibrate(bias_second_moment);
        state.bias_first_moment = Int16Quantizer::quantize(bias_first_moment, bias_first_parameters);
        state.bias_second_moment = Int16Quantizer::quantize(bias_second_moment, bias_second_parameters);
    }
}

double CompressedAdamOptimizer::learningRate() const {
    return learning_rate;
}

double CompressedAdamOptimizer::beta1() const {
    return first_decay;
}

double CompressedAdamOptimizer::beta2() const {
    return second_decay;
}

size_t CompressedAdamOptimizer::stateBytes() const {
    size_t count = 0;
    for (const LayerState& state : states) {
        count += state.weight_first_moment.values.size() * sizeof(std::int16_t);
        count += state.weight_second_moment.values.size() * sizeof(std::int16_t);
        count += state.bias_first_moment.values.size() * sizeof(std::int16_t);
        count += state.bias_second_moment.values.size() * sizeof(std::int16_t);
        count += 4 * sizeof(QuantizationParameters);
    }
    return count;
}

void CompressedAdamOptimizer::ensureState(const std::vector<DenseLayer>& layers) {
    if (states.size() == layers.size()) {
        bool shape_matches = true;
        for (size_t index = 0; index < layers.size(); ++index) {
            if (states[index].input_size != layers[index].weights().size() ||
                states[index].output_size != layers[index].bias().size()) {
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
        const size_t input_count = layer.weights().size();
        const size_t output_count = layer.bias().size();
        LayerState state;
        state.input_size = input_count;
        state.output_size = output_count;
        state.weight_first_moment = quantizeZeros(input_count * output_count);
        state.weight_second_moment = quantizeZeros(input_count * output_count);
        state.bias_first_moment = quantizeZeros(output_count);
        state.bias_second_moment = quantizeZeros(output_count);
        states.push_back(std::move(state));
    }
}
