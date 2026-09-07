/**
 * @ Author: GloomShade
 * @ Create Time: 2025-01-03 08:20:00
 * @ Modified by: GloomShade
 * @ Modified time: 2025-01-03 13:53:22
 * @ Description:
 */

#include "headers/DenseLayer.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

// Constructeur pour initialiser les poids et les biais
DenseLayer::DenseLayer(int input_size, int output_size) {
    if (input_size <= 0 || output_size <= 0) {
        throw std::invalid_argument("Layer dimensions must be positive");
    }

    weights_data.resize(input_size, std::vector<double>(output_size));
    bias_data.resize(output_size, 0.0);
    weight_gradients.resize(input_size, std::vector<double>(output_size, 0.0));
    bias_gradients.resize(output_size, 0.0);

    // Initialisation aléatoire des poids
    for (auto &row : weights_data) {
        for (auto &weight : row) {
            weight = ((double)rand() / RAND_MAX) - 0.5;  // Plage [-0.5, 0.5]
        }
    }
}

// Définir la fonction d'activation pour cette couche
void DenseLayer::set_algorithm(std::string algo) {
    algorithm = algo;
}

// Définir la fonction d'activation post-activation pour cette couche
void DenseLayer::set_post_algorithm(std::string algo) {
    post_algorithm = algo;
}

// Propagation avant dans la couche
std::vector<double> DenseLayer::forward(const std::vector<double>& inputs) {
    if (inputs.size() != weights_data.size()) {
        throw std::invalid_argument("Input size does not match layer size");
    }

    this->inputs = inputs;
    preActivations.assign(bias_data.size(), 0.0);
    outputs.assign(bias_data.size(), 0.0);

    // Calcul des sorties de la couche
    for (size_t j = 0; j < bias_data.size(); ++j) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            preActivations[j] += inputs[i] * weights_data[i][j];
        }
        preActivations[j] += bias_data[j];
        outputs[j] = preActivations[j];
        // Appliquer la fonction d'activation
        if (this->algorithm == "sigmoid") {
            outputs[j] = sigmoid(outputs[j]);
        } else if (this->algorithm == "sigmoid_derivative") {
            outputs[j] = sigmoid_derivative(outputs[j]);
        } else if (this->algorithm == "relu") {
            outputs[j] = relu(outputs[j]);
        } else if (this->algorithm == "leaky_relu") {
            outputs[j] = leakyRelu(outputs[j]);
        } else if (this->algorithm == "tanh") {
            outputs[j] = tanhActivation(outputs[j]);
        } else if (this->algorithm == "tanh_derivative") {
            outputs[j] = tanhDerivative(outputs[j]);
        }
    }

    if (this->post_algorithm == "softmax") {
        outputs = softmax(outputs);
    }

    has_forward_cache = true;
    return outputs;
}

std::vector<double> DenseLayer::backward(const std::vector<double>& gradient_output) {
    if (!has_forward_cache) {
        throw std::logic_error("Backward pass requires a preceding forward pass");
    }
    if (gradient_output.size() != bias_data.size()) {
        throw std::invalid_argument("Gradient size does not match layer output size");
    }
    if (algorithm == "sigmoid_derivative" || algorithm == "tanh_derivative") {
        throw std::invalid_argument("Derivative functions cannot be used as training activations");
    }

    std::vector<double> gradient_after_activation = gradient_output;
    if (post_algorithm == "softmax") {
        double dot_product = 0.0;
        for (size_t index = 0; index < outputs.size(); ++index) {
            dot_product += gradient_output[index] * outputs[index];
        }
        for (size_t index = 0; index < outputs.size(); ++index) {
            gradient_after_activation[index] = outputs[index] * (gradient_output[index] - dot_product);
        }
    } else if (post_algorithm != "none") {
        throw std::invalid_argument("Unknown post-activation function");
    }

    std::vector<double> gradient_inputs(inputs.size(), 0.0);
    for (size_t output_index = 0; output_index < bias_data.size(); ++output_index) {
        const double local_gradient = gradient_after_activation[output_index] *
            activationDerivative(algorithm, preActivations[output_index]);
        bias_gradients[output_index] += local_gradient;

        for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
            weight_gradients[input_index][output_index] +=
                inputs[input_index] * local_gradient;
            gradient_inputs[input_index] += weights_data[input_index][output_index] * local_gradient;
        }
    }

    return gradient_inputs;
}

void DenseLayer::zeroGradients() {
    for (auto& row : weight_gradients) {
        std::fill(row.begin(), row.end(), 0.0);
    }
    std::fill(bias_gradients.begin(), bias_gradients.end(), 0.0);
}

std::vector<std::vector<double>>& DenseLayer::weights() {
    return weights_data;
}

const std::vector<std::vector<double>>& DenseLayer::weights() const {
    return weights_data;
}

std::vector<double>& DenseLayer::bias() {
    return bias_data;
}

const std::vector<double>& DenseLayer::bias() const {
    return bias_data;
}

const std::vector<std::vector<double>>& DenseLayer::weightGradients() const {
    return weight_gradients;
}

const std::vector<double>& DenseLayer::biasGradients() const {
    return bias_gradients;
}

// Fonction d'activation Sigmoid
double DenseLayer::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));  // Sigmoid : 1 / (1 + exp(-x))
}

// Fonction dérivée de Sigmoid
double DenseLayer::sigmoid_derivative(double x) {
    const double value = sigmoid(x);
    return value * (1.0 - value);  // Derivee de sigmoid : sigmoid(x) * (1 - sigmoid(x))
}

// Fonction d'activation ReLU
double DenseLayer::relu(double x) {
    return (x > 0) ? x : 0.0;  // ReLU : max(0, x)
}

// Fonction d'activation Tanh
double DenseLayer::tanhActivation(double x) {
    return std::tanh(x);  // Tanh : tanh(x)
}

// Dérivée de la fonction d'activation Tanh
double DenseLayer::tanhDerivative(double x) {
    return 1.0 - std::tanh(x) * std::tanh(x);  // Derivée de Tanh : 1 - tanh^2(x)
}

// Fonction d'activation Leaky ReLU
double DenseLayer::leakyRelu(double x, double alpha) {
    return (x > 0) ? x : alpha * x;  // Leaky ReLU : alpha * x si x < 0
}

double DenseLayer::activationDerivative(const std::string& activation, double x) {
    if (activation == "none") {
        return 1.0;
    }
    if (activation == "sigmoid") {
        const double value = sigmoid(x);
        return value * (1.0 - value);
    }
    if (activation == "relu") {
        return x > 0.0 ? 1.0 : 0.0;
    }
    if (activation == "leaky_relu") {
        return x > 0.0 ? 1.0 : 0.01;
    }
    if (activation == "tanh") {
        const double value = std::tanh(x);
        return 1.0 - value * value;
    }
    throw std::invalid_argument("Unknown activation function");
}

// Fonction Softmax
std::vector<double> DenseLayer::softmax(const std::vector<double> &inputs) {
    if (inputs.empty()) {
        return {};
    }

    const double max_input = *std::max_element(inputs.begin(), inputs.end());
    std::vector<double> exp_values(inputs.size());
    double sum = 0.0;

    // Soustraire le maximum evite les debordements pour les grandes valeurs.
    for (size_t i = 0; i < inputs.size(); ++i) {
        exp_values[i] = std::exp(inputs[i] - max_input);
        sum += exp_values[i];
    }

    // Normalisation des exponentielles pour obtenir les probabilités
    for (size_t i = 0; i < exp_values.size(); ++i) {
        exp_values[i] /= sum;
    }

    return exp_values;
}
