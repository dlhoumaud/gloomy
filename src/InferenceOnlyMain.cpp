// Runtime d'inference minimal : ne depend que du reseau (DenseLayer,
// NeuralNetwork) et de ses (de)serializers, pas du moteur d'apprentissage
// (LossFunction, Optimizer, LearningMemory, GloomyConfig...). Prevu comme
// point de depart pour une cible embarquee — voir docs/quantization.md et
// docs/roadmap.md, « Priorité moyenne : compression et embarqué ». Compare
// son empreinte (voir la cible `make infer` et le Makefile) a celle du CLI
// complet `bin/gloomy`.
#include "headers/NeuralNetwork.h"
#include "headers/NetworkSerialization.h"
#include "headers/NetworkQuantization.h"
#include "headers/QuantizedNetworkSerialization.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<double> parseInputs(const std::string& text) {
    std::vector<double> values;
    std::istringstream stream(text);
    double value = 0.0;
    while (stream >> value) {
        values.push_back(value);
    }
    if (!stream.eof()) {
        throw std::invalid_argument("Input sequence contains a non-numeric value");
    }
    if (values.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    return values;
}

void displayHelp(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <model_path> \"<sequence_values>\" [--quantized]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "  <model_path>      : fichier reseau (GLOOMYNN, NetworkSerialization) ou, avec" << std::endl;
    std::cerr << "                      --quantized, fichier reseau quantifie (GLOOMYQN," << std::endl;
    std::cerr << "                      QuantizedNetworkSerialization)." << std::endl;
    std::cerr << "  <sequence_values> : valeurs d'entree separees par des espaces." << std::endl;
    std::cerr << "  --quantized       : charge un artefact int8 au lieu d'un float64." << std::endl;
}
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        displayHelp(argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string sequence_text = argv[2];
    bool quantized = false;
    for (int index = 3; index < argc; ++index) {
        if (std::string(argv[index]) == "--quantized") {
            quantized = true;
        }
    }

    std::vector<double> sequence;
    try {
        sequence = parseInputs(sequence_text);
    } catch (const std::exception& e) {
        std::cerr << "Input error: " << e.what() << std::endl;
        return 1;
    }

    try {
        NeuralNetwork network;
        if (quantized) {
            const QuantizedNetwork loaded = QuantizedNetworkSerialization::load(model_path);
            network = NetworkQuantization::dequantize(loaded);
        } else {
            NetworkSerialization::load(model_path, network);
        }

        const std::vector<double> output = network.forward(sequence);
        for (double value : output) {
            std::cout << value << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during inference: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
