#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include "headers/DenseLayer.h"
#include "headers/NeuralNetwork.h"
#include "headers/GloomyConfig.h"
#include "headers/GloomyConfigFile.h"
#include "headers/OnlineLearningRuntime.h"
#include "headers/ModelSerialization.h"

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
    std::cerr << "Usage: " << programName << " <sequence_values> [-c C] [-a A] [-n N] [-l L] [-f PATH] [-h]" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  <sequence_values>    : Sequence of space-separated values, ex: \"10.5 11.0 12.3\"" << std::endl;
    std::cerr << "  -c C                 : Number of predictions to generate (default 1)" << std::endl;
    std::cerr << "  -l L                 : Number of hidden layers (default 2)" << std::endl;
    std::cerr << "  -n N                 : Number of neurons per hidden layer (default 2)" << std::endl;
    std::cerr << "  -a [none|sigmoid|sigmoid_derivative|relu|leaky_relu|tanh|tanh_derivative]" << std::endl;
    std::cerr << "                       : Activation function (default none)" << std::endl;
    std::cerr << "  -A [none|softmax]    : Post-activation function (default none)" << std::endl;
    std::cerr << "  -f, --config PATH    : Load a gloomy.config key=value file (see docs/roadmap.md)." << std::endl;
    std::cerr << "                         Priority is CLI > file > defaults: any -c/-l/-n/-a/-A" << std::endl;
    std::cerr << "                         flag overrides the value loaded from the file. The file's" << std::endl;
    std::cerr << "                         \"runtime\" key selects \"inference\" (default)," << std::endl;
    std::cerr << "                         \"online_learning\", or \"training\"." << std::endl;
    std::cerr << "  -h                   : Show this help" << std::endl;
    std::cerr << std::endl;
}

// Fonction pour analyser les arguments de la ligne de commande. `config` est
// initialisee par l'appelant a GloomyConfig::defaults() ; cette fonction
// applique d'abord un eventuel fichier -f/--config par-dessus (fichier >
// defauts), puis les options explicites de la ligne de commande (seconde
// passe), qui l'emportent toujours (CLI > fichier > defauts).
void parseArguments(int argc, char *argv[], GloomyConfig &config) {
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if ((arg == "-f" || arg == "--config") && i + 1 < argc) {
            config = GloomyConfigFile::load(argv[i + 1], config);
            break;
        }
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-f" || arg == "--config") {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + arg);
            }
            ++i; // Deja pris en compte lors de la premiere passe.
        } else if (arg == "-c" && i + 1 < argc) {
            config.predictions = std::stoi(argv[++i]);
        } else if (arg == "-l" && i + 1 < argc) {
            config.hidden_layers = std::stoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            config.neurons = std::stoi(argv[++i]);
        } else if (arg == "-a" && i + 1 < argc) {
            config.activation = argv[++i];
        } else if (arg == "-A" && i + 1 < argc) {
            config.post_activation = argv[++i];
        } else if (arg == "-h") {
            displayHelp(argv[0]);
            exit(0);
        } else {
            throw std::invalid_argument("Unknown or incomplete option: " + arg);
        }
    }

    if (config.predictions < 0 || config.hidden_layers < 0 || config.neurons <= 0) {
        throw std::invalid_argument("-c and -l must be non-negative, and -n must be positive");
    }

    const std::vector<std::string> activations = {"none", "sigmoid", "sigmoid_derivative", "relu", "leaky_relu", "tanh", "tanh_derivative"};
    if (std::find(activations.begin(), activations.end(), config.activation) == activations.end()) {
        throw std::invalid_argument("Unknown activation function: " + config.activation);
    }
    if (config.post_activation != "none" && config.post_activation != "softmax") {
        throw std::invalid_argument("Unknown post-activation function: " + config.post_activation);
    }
    if (config.runtime != "inference" && config.runtime != "online_learning" && config.runtime != "training") {
        throw std::invalid_argument("Unknown runtime: " + config.runtime);
    }
}

// INFERENCE_RUNTIME : comportement historique du CLI (prediction
// autoregressive avec un reseau reconstruit a chaque etape, voir
// docs/roadmap.md, section « Limites connues »).
int runInference(const GloomyConfig &config, std::vector<double> sequence) {
    NeuralNetwork nn;
    nn.algorithm = config.activation;
    nn.post_algorithm = config.post_activation;

    // -l correspond au nombre de couches cachees.
    auto rebuild = [&]() {
        nn.clear();
        if (config.hidden_layers == 0) {
            nn.addLayer(sequence.size(), 1);
        } else {
            nn.addLayer(sequence.size(), config.neurons);
            for (int i = 1; i < config.hidden_layers; ++i) {
                nn.addLayer(config.neurons, config.neurons);
            }
            nn.addLayer(config.neurons, 1);
        }
    };
    rebuild();

    for (int i = 0; i < config.predictions; ++i) {
        try {
            double prediction = nn.predict(sequence);  // Faire une prediction
            std::cout << prediction << std::endl;

            // Ajouter la prediction a la sequence pour la prochaine prediction
            sequence.push_back(prediction);
            rebuild();  // Reconstruire le reseau avec la nouvelle dimension d'entree
        } catch (const std::exception &e) {
            std::cerr << "Error during prediction: " << e.what() << std::endl;
            return 1;
        }
    }

    return 0;
}

// TRAINING_RUNTIME : entrainement complet sur un jeu de donnees complet,
// avec une seule passe par epoch et un batch configure par config.batch_size.
int runTrainingRuntime(const GloomyConfig &config, const std::vector<double> &sequence) {
    try {
        const TrainingResult result = runTraining(config, sequence, config.model_path);
        saveTrainingArtifacts(config, result);
        std::cerr << "average_loss=" << result.average_loss << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error during training: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

// ONLINE_LEARNING_RUNTIME : boucle observation -> normalisation ->
// prediction -> cible -> erreur -> memoire -> scheduler -> replay -> mise a
// jour (voir src/headers/OnlineLearningRuntime.h et docs/roadmap.md).
int runOnline(const GloomyConfig &config, const std::vector<double> &sequence) {
    try {
        const OnlineLearningResult result = runOnlineLearning(config, sequence, config.model_path);
        saveOnlineArtifacts(config, result);
        for (size_t index = 0; index < result.steps.size(); ++index) {
            const OnlineLearningStep &step = result.steps[index];
            std::cout << index << '\t' << step.observation << '\t' << step.target
                       << '\t' << step.prediction_before_update << '\t'
                       << step.loss_before_update << std::endl;
        }
        std::cerr << "average_loss=" << result.average_loss
                   << " memory_size=" << result.memory_size << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Error during online learning: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

int main(int argc, char *argv[]) {
    // Paramètres par défaut, centralisés dans GloomyConfig (voir
    // src/headers/GloomyConfig.h et docs/roadmap.md, section
    // « Configuration fichier »).
    GloomyConfig config = GloomyConfig::defaults();

    // Vérification que les arguments sont fournis correctement
    if (argc < 2) {
        displayHelp(argv[0]);
        return 1;
    }

    // Analyser les arguments
    std::string input_str = argv[1];
    try {
        parseArguments(argc, argv, config);
    } catch (const std::exception &e) {
        std::cerr << "Argument error: " << e.what() << std::endl;
        displayHelp(argv[0]);
        return 1;
    }

    // Seed l'initialisation des poids une fois la configuration finale
    // resolue (defauts, puis fichier -f/--config), pour que le CLI soit
    // reproductible par defaut (voir docs/roadmap.md, « Priorite moyenne :
    // robustesse mathematique »).
    DenseLayer::seedWeightInitialization(config.seed);

    std::vector<double> sequence;
    try {
        sequence = parseInputs(input_str);
    } catch (const std::exception &e) {
        std::cerr << "Input error: " << e.what() << std::endl;
        return 1;
    }

    if (config.runtime == "online_learning") {
        return runOnline(config, sequence);
    }
    if (config.runtime == "training") {
        return runTrainingRuntime(config, sequence);
    }
    return runInference(config, std::move(sequence));
}
