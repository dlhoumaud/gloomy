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

    // Taille de la fenetre d'entree des runtimes online/training : nombre de
    // valeurs consecutives de la sequence utilisees comme entree du reseau
    // pour predire la valeur suivante (voir OnlineLearningRuntime.cpp et
    // docs/roadmap.md, « Priorité haute »). window_size = 1 (defaut)
    // reproduit exactement le comportement scalaire historique (une entree,
    // une sortie) ; une valeur plus grande donne au reseau plusieurs pas
    // passes a la fois. Seule la fenetre d'entree est configurable ici : la
    // sortie reste un scalaire (la valeur suivante).
    std::size_t window_size = 1;

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
    // Annealing de prioritized_beta au fil des replays (incremente jusqu'a
    // 1.0 apres chaque sampleIndexed()) ; 0.0 (defaut) desactive l'annealing,
    // beta reste fixe comme avant ce champ.
    double prioritized_beta_annealing_rate = 0.0;
    // Exploration controlee des echantillons de faible priorite du
    // prioritized replay : melange la distribution priorisee avec une
    // distribution uniforme dans une proportion epsilon (0 = distribution
    // priorisee pure, 1 = uniforme pure). 0.0 (defaut) ne change rien au
    // comportement precedent.
    double prioritized_exploration_epsilon = 0.0;
    std::uint32_t seed = 5489u;

    // Poids des cinq composantes du score d'importance (ImportanceScorer),
    // qui determine TrainingSample::priority calcule par
    // LearningEngine::learn()/trainFromMemory(). Le defaut (error=1.0, le
    // reste a 0.0) reproduit exactement le comportement historique
    // (priorite = erreur seule) ; novelty/rarity/recency/diversity sont
    // desormais reellement calculees (voir docs/memory.md, « Score
    // d'importance ») mais n'influencent la priorite que si leur poids est
    // rendu positif ici.
    double importance_weight_error = 1.0;
    double importance_weight_novelty = 0.0;
    double importance_weight_rarity = 0.0;
    double importance_weight_recency = 0.0;
    double importance_weight_diversity = 0.0;

    // Précision de stockage des échantillons en mémoire : "float64" est la
    // représentation native actuelle, sans quantification.
    std::string precision = "float64";

    // Scheduling et entraînement. train_every = 1 reproduit le comportement
    // actuel de LearningEngine::learn() (entraînement après chaque
    // observation, équivalent à EverySampleScheduler).
    std::size_t train_every = 1;
    std::size_t batch_size = 8;
    std::size_t epochs = 1;

    // Chemins de persistance, vides par défaut : la sauvegarde du modèle,
    // de l'optimiseur, de la mémoire et des métriques par les runtimes CLI
    // (`saveTrainingArtifacts`/`saveOnlineArtifacts`) est opt-in, seulement
    // quand le champ correspondant est explicitement renseigné (voir
    // docs/roadmap.md). Un défaut non vide écrirait silencieusement des
    // fichiers dans le répertoire courant à chaque exécution ; pour
    // `metrics_path`, un défaut à "benchmark_results.csv" écraserait en
    // plus le fichier produit par `make benchmark` (qui n'utilise pas
    // GloomyConfig et n'est donc pas affecté par ce champ).
    std::string model_path;
    std::string optimizer_path;
    std::string memory_path;
    std::string metrics_path;

    // Detection active de concept drift dans le runtime online (voir
    // ConceptDriftDetector.h) : compare la moyenne d'erreur recente a une
    // ligne de base historique et, si elle la depasse de plus de
    // concept_drift_std_devs ecarts-types, declenche immediatement un
    // replay supplementaire depuis la memoire (en plus de la mise a jour
    // normalement planifiee). Desactivee par defaut (false) : ne change
    // rien au comportement existant tant qu'elle n'est pas activee
    // explicitement.
    bool concept_drift_detection = false;
    std::size_t concept_drift_recent_window = 10;
    std::size_t concept_drift_minimum_history = 20;
    double concept_drift_std_devs = 3.0;

    static const GloomyConfig& defaults();
};

#endif // GLOOMY_CONFIG_H
