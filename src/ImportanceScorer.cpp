#include "headers/ImportanceScorer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
double clampUnit(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Importance component must be finite");
    }
    return std::clamp(value, 0.0, 1.0);
}

void validateWeights(const ImportanceWeights& weights) {
    const double values[] = {
        weights.error,
        weights.novelty,
        weights.rarity,
        weights.recency,
        weights.diversity
    };
    double total = 0.0;
    for (double value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("Importance weights must be finite and non-negative");
        }
        total += value;
    }
    if (total <= 0.0) {
        throw std::invalid_argument("At least one importance weight must be positive");
    }
}
}

ImportanceScorer::ImportanceScorer(ImportanceWeights weights) {
    setWeights(weights);
}

double ImportanceScorer::score(const ImportanceComponents& components) const {
    const double weighted_sum =
        importance_weights.error * clampUnit(components.error) +
        importance_weights.novelty * clampUnit(components.novelty) +
        importance_weights.rarity * clampUnit(components.rarity) +
        importance_weights.recency * clampUnit(components.recency) +
        importance_weights.diversity * clampUnit(components.diversity);

    const double weight_sum =
        importance_weights.error + importance_weights.novelty +
        importance_weights.rarity + importance_weights.recency +
        importance_weights.diversity;
    return weighted_sum / weight_sum;
}

const ImportanceWeights& ImportanceScorer::weights() const {
    return importance_weights;
}

void ImportanceScorer::setWeights(const ImportanceWeights& weights) {
    validateWeights(weights);
    importance_weights = weights;
}
