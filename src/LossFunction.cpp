#include "headers/LossFunction.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {
void validateInputs(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) {
    if (prediction.empty()) {
        throw std::invalid_argument("Prediction cannot be empty");
    }
    if (prediction.size() != target.size()) {
        throw std::invalid_argument("Prediction and target sizes must match");
    }
    for (double value : prediction) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Prediction values must be finite");
        }
    }
    for (double value : target) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Target values must be finite");
        }
    }
}
}

double MSELoss::compute(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);

    double squaredError = 0.0;
    for (size_t index = 0; index < prediction.size(); ++index) {
        const double error = prediction[index] - target[index];
        squaredError += error * error;
    }

    return squaredError / static_cast<double>(prediction.size());
}

std::vector<double> MSELoss::gradient(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);

    const double scale = 2.0 / static_cast<double>(prediction.size());
    std::vector<double> result(prediction.size());
    for (size_t index = 0; index < prediction.size(); ++index) {
        result[index] = scale * (prediction[index] - target[index]);
    }

    return result;
}

double MAELoss::compute(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);
    double absolute_error = 0.0;
    for (size_t index = 0; index < prediction.size(); ++index) {
        absolute_error += std::abs(prediction[index] - target[index]);
    }
    return absolute_error / static_cast<double>(prediction.size());
}

std::vector<double> MAELoss::gradient(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);
    const double scale = 1.0 / static_cast<double>(prediction.size());
    std::vector<double> result(prediction.size(), 0.0);
    for (size_t index = 0; index < prediction.size(); ++index) {
        const double error = prediction[index] - target[index];
        result[index] = error > 0.0 ? scale : (error < 0.0 ? -scale : 0.0);
    }
    return result;
}

HuberLoss::HuberLoss(double delta)
    : threshold(delta) {
    if (delta <= 0.0 || !std::isfinite(delta)) {
        throw std::invalid_argument("Huber delta must be finite and positive");
    }
}

double HuberLoss::compute(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);
    double total = 0.0;
    for (size_t index = 0; index < prediction.size(); ++index) {
        const double absolute_error = std::abs(prediction[index] - target[index]);
        total += absolute_error <= threshold
            ? 0.5 * absolute_error * absolute_error
            : threshold * (absolute_error - 0.5 * threshold);
    }
    return total / static_cast<double>(prediction.size());
}

std::vector<double> HuberLoss::gradient(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);
    const double scale = 1.0 / static_cast<double>(prediction.size());
    std::vector<double> result(prediction.size(), 0.0);
    for (size_t index = 0; index < prediction.size(); ++index) {
        const double error = prediction[index] - target[index];
        const double absolute_error = std::abs(error);
        if (absolute_error <= threshold) {
            result[index] = scale * error;
        } else {
            result[index] = scale * threshold * (error > 0.0 ? 1.0 : -1.0);
        }
    }
    return result;
}

double HuberLoss::delta() const {
    return threshold;
}

namespace {
constexpr double cross_entropy_epsilon = 1e-12;

void validateProbabilities(const std::vector<double>& prediction) {
    for (double value : prediction) {
        if (value < 0.0) {
            throw std::invalid_argument(
                "CrossEntropyLoss expects non-negative probabilities (post-softmax prediction)"
            );
        }
    }
}
}

double CrossEntropyLoss::compute(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);
    validateProbabilities(prediction);

    double total = 0.0;
    for (size_t index = 0; index < prediction.size(); ++index) {
        if (target[index] == 0.0) continue;
        total -= target[index] * std::log(std::max(prediction[index], cross_entropy_epsilon));
    }
    return total;
}

std::vector<double> CrossEntropyLoss::gradient(
    const std::vector<double>& prediction,
    const std::vector<double>& target
) const {
    validateInputs(prediction, target);
    validateProbabilities(prediction);

    std::vector<double> result(prediction.size());
    for (size_t index = 0; index < prediction.size(); ++index) {
        result[index] = -target[index] / std::max(prediction[index], cross_entropy_epsilon);
    }
    return result;
}
