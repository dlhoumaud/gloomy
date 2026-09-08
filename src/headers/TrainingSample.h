#ifndef TRAINING_SAMPLE_H
#define TRAINING_SAMPLE_H

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

struct TrainingSample {
    std::vector<double> input;
    std::vector<double> target;
    double priority = 0.0;
    double error = 0.0;
    double novelty = 0.0;
    double rarity = 0.0;
    double recency = 0.0;
    double diversity = 0.0;
    std::size_t age = 0;
    std::size_t usage_count = 0;
};

// Valide qu'un echantillon est utilisable par une memoire d'apprentissage :
// input/target non vides et entierement finis (pas de NaN/infini), sur le
// meme principe que LossFunction::validateInputs et DenseLayer::forward/
// backward (voir docs/roadmap.md, « Priorité moyenne : robustesse
// mathématique »). Ne verifie pas la coherence dimensionnelle entre
// echantillons successifs : chaque memoire qui l'exige (NoveltyMemory,
// QuantizedFIFOMemory, QuantizedInt8FIFOMemory) le fait deja elle-meme.
// inline : TrainingSample.h n'a pas de .cpp associe, et cette fonction est
// appelee depuis plusieurs unites de compilation (une par strategie de
// memoire).
inline void validateTrainingSampleVectors(const TrainingSample& sample) {
    if (sample.input.empty() || sample.target.empty()) {
        throw std::invalid_argument("Training sample input and target cannot be empty");
    }
    for (double value : sample.input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Training sample input must be finite");
        }
    }
    for (double value : sample.target) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Training sample target must be finite");
        }
    }
}

#endif // TRAINING_SAMPLE_H
