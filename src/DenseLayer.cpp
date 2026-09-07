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
    weights.resize(input_size, std::vector<double>(output_size));
    bias.resize(output_size, 0.0);

    // Initialisation aléatoire des poids
    for (auto &row : weights) {
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
std::vector<double> DenseLayer::forward(const std::vector<double> &inputs) {
    if (inputs.size() != weights.size()) {
        throw std::invalid_argument("Input size does not match layer size");
    }

    this->inputs = inputs;  // Stocker les entrées pour le calcul du gradient (pour l'entraînement)
    std::vector<double> outputs(bias.size(), 0.0);  // Initialisation des sorties avec des zéros

    // Calcul des sorties de la couche
    for (size_t j = 0; j < bias.size(); ++j) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            outputs[j] += inputs[i] * weights[i][j];  // Somme pondérée des entrées
        }
        outputs[j] += bias[j];  // Ajouter le biais
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

    // Appliquer la fonction d'activation post (si définie)
    if (this->post_algorithm == "softmax") {
        outputs = softmax(outputs);
    }

    return outputs;
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
