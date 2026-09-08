#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct BenchmarkResult {
    std::string memory_strategy;
    std::string precision;
    std::string optimizer;
    // Fonction de perte utilisee pour l'entrainement de ce scenario
    // ("mse"/"mae"/"huber"). Vide quand la notion ne s'applique pas
    // (les scenarios qui n'entrainent rien, ex. la baseline naive).
    std::string loss_function;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    double mae = 0.0;
    double rmse = 0.0;
    double training_time_ms = 0.0;
    double inference_time_us = 0.0;
    size_t memory_used_bytes = 0;
    size_t samples_stored = 0;
    size_t parameter_updates = 0;
    double forgetting = 0.0;
    // Capacite de la memoire d'apprentissage utilisee pour ce scenario (0
    // si la notion ne s'applique pas, ex. les lignes de forgetting).
    size_t memory_capacity = 0;
    // mae / mae_du_scenario_full_dataset (meme optimiseur et meme perte),
    // pour comparer directement le cout d'une memoire bornee au dataset
    // complet. Laisse a 0 quand aucune reference full_dataset n'est
    // disponible.
    double mae_ratio_to_full_dataset = 0.0;
    // Seed utilisee pour l'initialisation des poids et le generateur de la
    // memoire d'apprentissage sur ce run (0 si la notion ne s'applique pas,
    // ex. un scenario qui n'est pas repete sur plusieurs seeds).
    std::uint32_t seed = 0;
    // Moyenne et ecart-type (population, diviseur N) du MAE sur l'ensemble
    // des seeds d'un meme scenario (memory_strategy/optimizer/loss/
    // capacity) ; identiques sur chaque ligne du groupe. 0 quand un seul
    // run existe.
    double mae_mean = 0.0;
    double mae_stddev = 0.0;
    // Demi-largeur de l'intervalle de confiance a 95% sur mae_mean
    // (mae_mean +/- mae_ci95_margin), calculee via la loi de Student pour
    // le nombre de seeds du groupe. 0 quand un seul run existe ou que le
    // nombre de seeds ne correspond a aucune valeur critique connue (voir
    // BenchmarkRunner.cpp).
    double mae_ci95_margin = 0.0;
    // Nombre approximatif d'operations multiplication-accumulation (MAC)
    // pour l'ensemble de la procedure d'entrainement (forward + backward,
    // facteur x3 approximatif sur les MAC du seul forward — voir
    // BenchmarkRunner.cpp). Grossier par construction : ignore le cout de
    // l'optimiseur et des activations. 0 quand la notion ne s'applique pas.
    double approximate_macs = 0.0;
    // Debit mesure : nombre d'echantillons traites et de mises a jour de
    // poids par seconde, a partir de training_time_ms. 0 quand
    // training_time_ms vaut 0 (mesure indisponible ou non applicable).
    double samples_per_second = 0.0;
    double updates_per_second = 0.0;
};

class BenchmarkCsv {
public:
    static void write(
        const std::string& path,
        const std::vector<BenchmarkResult>& results
    );
};

#endif // BENCHMARK_H
