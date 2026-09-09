#include "headers/ConceptDriftDetector.h"
#include <cmath>
#include <stdexcept>

ConceptDriftDetector::ConceptDriftDetector(
    std::size_t recent_window,
    std::size_t minimum_history,
    double num_std_devs
)
    : recent_window_size(recent_window),
      minimum_history_count(minimum_history),
      std_dev_threshold(num_std_devs) {
    if (recent_window == 0) {
        throw std::invalid_argument("Concept drift detector recent window must be positive");
    }
    if (num_std_devs < 0.0 || !std::isfinite(num_std_devs)) {
        throw std::invalid_argument("Concept drift detector threshold must be finite and non-negative");
    }
    recent_errors.reserve(recent_window);
}

bool ConceptDriftDetector::update(double error) {
    if (!std::isfinite(error)) {
        throw std::invalid_argument("Concept drift detector received a non-finite error value");
    }

    if (recent_errors.size() < recent_window_size) {
        recent_errors.push_back(error);
    } else {
        recent_errors[recent_next_index] = error;
    }
    recent_next_index = (recent_next_index + 1) % recent_window_size;

    drift_flag = false;
    if (recent_errors.size() == recent_window_size && baseline_count >= minimum_history_count) {
        double recent_sum = 0.0;
        for (double value : recent_errors) recent_sum += value;
        const double recent_mean = recent_sum / static_cast<double>(recent_window_size);
        const double variance = baseline_m2 / static_cast<double>(baseline_count);
        const double std_dev = std::sqrt(variance);
        const double threshold = baseline_mean + std_dev_threshold * std_dev;
        drift_flag = recent_mean > threshold;
    }

    // Ligne de base (Welford) mise a jour apres le calcul de detection.
    ++baseline_count;
    const double delta = error - baseline_mean;
    baseline_mean += delta / static_cast<double>(baseline_count);
    const double delta2 = error - baseline_mean;
    baseline_m2 += delta * delta2;

    return drift_flag;
}

bool ConceptDriftDetector::driftDetected() const {
    return drift_flag;
}

double ConceptDriftDetector::recentMean() const {
    if (recent_errors.empty()) return 0.0;
    double sum = 0.0;
    for (double value : recent_errors) sum += value;
    return sum / static_cast<double>(recent_errors.size());
}

double ConceptDriftDetector::baselineMean() const {
    return baseline_mean;
}

double ConceptDriftDetector::baselineStdDev() const {
    if (baseline_count == 0) return 0.0;
    return std::sqrt(baseline_m2 / static_cast<double>(baseline_count));
}

std::size_t ConceptDriftDetector::baselineCount() const {
    return baseline_count;
}

void ConceptDriftDetector::reset() {
    recent_errors.clear();
    recent_next_index = 0;
    baseline_count = 0;
    baseline_mean = 0.0;
    baseline_m2 = 0.0;
    drift_flag = false;
}
