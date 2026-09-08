#ifndef GLOOMY_CONFIG_H
#define GLOOMY_CONFIG_H

#include <cstddef>
#include <cstdint>
#include <string>

// Rassemble en un seul endroit les valeurs par défaut du runtime Gloomy,
// aujourd'hui dispersées entre le CLI (src/main.cpp), les valeurs par défaut
// des constructeurs (Optimizer, LearningMemory) et le benchmark. C'est une
// préparation directe au fichier `gloomy.config` prévu par la feuille de
// route (voir docs/roadmap.md, section « Configuration fichier ») : ce
// fichier pourra surcharger ces défauts, eux-mêmes surchargeables par les
// options de la ligne de commande (priorité CLI > fichier > défauts).
//
// Chaque champ reprend, quand il existe, le défaut déjà utilisé par le
// composant correspondant (voir le commentaire associé) ; les autres
// établissent ici, pour la première fois, un défaut central documenté.
struct GloomyConfig {
    // Mode de fonctionnement. Le CLI supporte aujourd'hui
    // "inference", "online_learning" et "training".
    std::string runtime = "inference";

    // Architecture du réseau : reprend les défauts actuels de src/main.cpp.
    std::string activation = "none";
    std::string post_activation = "none";
    int hidden_layers = 2;
    int neurons = 2;
    int predictions = 1;

    // Fonction de perte : "mse" comme choix le plus simple ; huber_delta
    // reprend le défaut de HuberLoss (voir src/headers/LossFunction.h).
    std::string loss = "mse";
    double huber_delta = 1.0;

    // Optimiseur : momentum, beta1, beta2 et epsilon reprennent les défauts
    // de MomentumOptimizer et AdamOptimizer (voir src/headers/
    // MomentumOptimizer.h et AdamOptimizer.h). Ni SGD ni Momentum ni Adam
    // n'ont de learning rate par défaut dans leur constructeur ; 0.01 est
    // établi ici comme valeur centrale, cohérente avec celle déjà utilisée
    // pour SGD et Adam dans BenchmarkRunner.
    std::string optimizer = "sgd";
    double learning_rate = 0.01;
    double momentum = 0.9;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    // Mémoire d'apprentissage : "fifo" comme stratégie la plus simple.
    // recent/error/novelty/historical_ratio reprennent le défaut de
    // HybridMemoryRatios ; prioritized_alpha reprend celui de
    // PrioritizedMemory ; novelty_threshold reprend celui de HybridMemory
    // (NoveltyMemory seule n'a pas de défaut et exige un seuil explicite).
    // seed reprend le défaut partagé par ReservoirMemory, PrioritizedMemory
    // et HybridMemory, qui est aussi le seed par défaut de std::mt19937.
    std::string memory_strategy = "fifo";
    std::size_t memory_capacity = 256;
    double recent_ratio = 0.25;
    double error_ratio = 0.25;
    double novelty_ratio = 0.25;
    double historical_ratio = 0.25;
    double novelty_threshold = 0.0;
    double prioritized_alpha = 0.6;
    // Correction du biais d'echantillonnage du prioritized replay
    // (importance-sampling exponent) ; reprend le defaut de
    // PrioritizedMemory. 0 desactive la correction (poids toujours 1.0).
    double prioritized_beta = 0.4;
    std::uint32_t seed = 5489u;

    // Précision de stockage des échantillons en mémoire : "float64" est la
    // représentation native actuelle, sans quantification.
    std::string precision = "float64";

    // Scheduling et entraînement. train_every = 1 reproduit le comportement
    // actuel de LearningEngine::learn() (entraînement après chaque
    // observation, équivalent à EverySampleScheduler).
    std::size_t train_every = 1;
    std::size_t batch_size = 8;
    std::size_t epochs = 1;

    // Chemins de persistance. model_path reprend la convention déjà en
    // usage dans la documentation (docs/serialization.md) ; optimizer_path
    // et memory_path suivent la même convention ; metrics_path reprend le
    // nom de fichier déjà produit par `make benchmark`
    // (src/BenchmarkRunner.cpp).
    std::string model_path = "model.gloomy";
    std::string optimizer_path = "optimizer.gloomy";
    std::string memory_path = "memory.gloomy";
    std::string metrics_path = "benchmark_results.csv";

    static const GloomyConfig& defaults();
};

#endif // GLOOMY_CONFIG_H
