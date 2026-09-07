#include "headers/Metrics.h"
#include <cmath>
#include <stdexcept>

namespace {
void validate(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) {
    if (prediction.empty()) {
        throw std::invalid_argument("Metric vectors cannot be empty");
    }
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Metric vector sizes must match");
    }
}
}

RegressionMetrics Metrics::regression(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) {
    validate(prediction, target);
    double absolute_error = 0.0;
    double squared_error = 0.0;
    for (size_t index = 0; index < prediction.size(); ++index) {
        const double error = prediction[index] - target[index];
        absolute_error += std::abs(error);
        squared_error += error * error;
    }

    const double count = static_cast<double>(prediction.size());
    return {absolute_error / count, std::sqrt(squared_error / count)};
}

double Metrics::forgetting(
    double performance_before,
    double performance_after
) {
    if (!std::isfinite(performance_before) || !std::isfinite(performance_after)) {
        throw std::invalid_argument("Forgetting metrics must be finite");
    }
    return performance_before - performance_after;
}
