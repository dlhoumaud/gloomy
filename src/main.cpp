#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <algorithm>
#include <stdexcept>
#include "headers/DenseLayer.h"
#include "headers/NeuralNetwork.h"

// Fonction pour analyser les entrées sous forme de chaîne de caractères et les convertir en vecteur de doubles
std::vector<double> parseInputs(const std::string &str) {
    std::vector<double> inputs;
    std::istringstream iss(str);
    double value;
    while (iss >> value) {
        inputs.push_back(value);
    }
    if (!iss.eof()) {
        throw std::invalid_argument("Input sequence contains a non-numeric value");
    }
    if (inputs.empty()) {
        throw std::invalid_argument("Input sequence cannot be empty");
    }
    return inputs;
}

// Fonction pour afficher l'aide sur les options du programme
void displayHelp(const char *programName) {
    std::cerr << "Usage: " << programName << " <sequence_values> [-c C] [-a A] [-n N] [-l L] [-h]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  <sequence_values>    : Sequence of space-separated values, ex: \"10.5 11.0 12.3\"" << std::endl;
    std::cerr << "  -c C                 : Number of predictions to generate (default 1)" << std::endl;
    std::cerr << "  -l L                 : Number of hidden layers (default 2)" << std::endl;
    std::cerr << "  -n N                 : Number of neurons per hidden layer (default 2)" << std::endl;
    std::cerr << "  -a [none|sigmoid|sigmoid_derivative|relu|leaky_relu|tanh|tanh_derivative]" << std::endl;
    std::cerr << "                       : Activation function (default none)" << std::endl;
    std::cerr << "  -A [none|softmax]    : Post-activation function (default none)" << std::endl;
    std::cerr << "  -h                   : Show this help" << std::endl;
    std::cerr << std::endl;
}

// Fonction pour analyser les arguments de la ligne de commande
void parseArguments(int argc, char *argv[], int &num_predictions, int &num_layers, int &num_neurons, std::string &algorithm, std::string &post_algorithm) {
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-c" && i + 1 < argc) {
            num_predictions = std::stoi(argv[++i]);
        } else if (arg == "-l" && i + 1 < argc) {
            num_layers = std::stoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            num_neurons = std::stoi(argv[++i]);
        } else if (arg == "-a" && i + 1 < argc) {
            algorithm = argv[++i];
        } else if (arg == "-A" && i + 1 < argc) {
            post_algorithm = argv[++i];
        } else if (arg == "-h") {
            displayHelp(argv[0]);
            exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete option: " + arg);
        }
    }

    if (num_predictions < 0 || num_layers < 0 || num_neurons <= 0) {
        throw std::invalid_argument("-c and -l must be non-negative, and -n must be positive");
    }

    const std::vector<std::string> activations = {"none", "sigmoid", "sigmoid_derivative", "relu", "leaky_relu", "tanh", "tanh_derivative"};
    if (std::find(activations.begin(), activations.end(), algorithm) == activations.end()) {
        throw std::invalid_argument("Unknown activation function: " + algorithm);
    }
    if (post_algorithm != "none" && post_algorithm != "softmax") {
        throw std::invalid_argument("Unknown post-activation function: " + post_algorithm);
    }
}

int main(int argc, char *argv[]) {
    std::srand(std::time(0));  // Initialisation du générateur de nombres aléatoires

    // Paramètres par défaut
    int num_predictions = 1;
    int num_layers = 2;
    int num_neurons = 2;
    std::string algorithm = "none";
    std::string post_algorithm = "none";

    // Vérification que les arguments sont fournis correctement
    if (argc < 2) {
        displayHelp(argv[0]);
        return 1;
    }

    // Analyser les arguments
    std::string input_str = argv[1];
    try {
        parseArguments(argc, argv, num_predictions, num_layers, num_neurons, algorithm, post_algorithm);
    } catch (const std::exception &e) {
        std::cerr << "Argument error: " << e.what() << std::endl;
        displayHelp(argv[0]);
        return 1;
    }

    std::vector<double> sequence;
    try {
        sequence = parseInputs(input_str);
    } catch (const std::exception &e) {
        std::cerr << "Input error: " << e.what() << std::endl;
        return 1;
    }

    // Initialisation du réseau neuronal
    NeuralNetwork nn;
    nn.algorithm = algorithm;
    nn.post_algorithm = post_algorithm;

    // -l correspond au nombre de couches cachees.
    if (num_layers == 0) {
        nn.addLayer(sequence.size(), 1);
    } else {
        nn.addLayer(sequence.size(), num_neurons);
        for (int i = 1; i < num_layers; ++i) {
            nn.addLayer(num_neurons, num_neurons);
        }
        nn.addLayer(num_neurons, 1);
    }

    // Generer les predictions demandees
    for (int i = 0; i < num_predictions; ++i) {
        try {
            double prediction = nn.predict(sequence);  // Faire une prediction
            std::cout << prediction << std::endl;

            // Ajouter la prediction a la sequence pour la prochaine prediction
            sequence.push_back(prediction);

            // Reconstruire le reseau avec la nouvelle dimension d'entree
            nn.clear();
            if (num_layers == 0) {
                nn.addLayer(sequence.size(), 1);
            } else {
                nn.addLayer(sequence.size(), num_neurons);
                for (int j = 1; j < num_layers; ++j) {
                    nn.addLayer(num_neurons, num_neurons);
                }
                nn.addLayer(num_neurons, 1);
            }
        } catch (const std::exception &e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}
