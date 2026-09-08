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
    // mae / mae_du_scenario_full_dataset (meme optimiseur), pour comparer
    // directement le cout d'une memoire bornee au dataset complet. Laisse a
    // 0 quand aucune reference full_dataset n'est disponible.
    double mae_ratio_to_full_dataset = 0.0;
    // Seed utilisee pour l'initialisation des poids et le generateur de la
    // memoire d'apprentissage sur ce run (0 si la notion ne s'applique pas,
    // ex. un scenario qui n'est pas repete sur plusieurs seeds).
    std::uint32_t seed = 0;
    // Moyenne et ecart-type (population, diviseur N) du MAE sur l'ensemble
    // des seeds d'un meme scenario (memory_strategy/optimizer/capacity) ;
    // identiques sur chaque ligne du groupe. 0 quand un seul run existe.
    double mae_mean = 0.0;
    double mae_stddev = 0.0;
};

class BenchmarkCsv {
public:
    static void write(
        const std::string& path,
        const std::vector<BenchmarkResult>& results
    );
};

#endif // BENCHMARK_H
