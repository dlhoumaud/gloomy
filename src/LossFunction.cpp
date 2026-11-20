#include "headers/LossFunction.h"
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
}
}

void LossFunction::gradientInto(
    const std::vector<double>& prediction,
    const std::vector<double>& target,
    std::vector<double>& output
) const {
    output = gradient(prediction, target);
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

    std::vector<double> result;
    gradientInto(prediction, target, result);
    return result;
}

void MSELoss::gradientInto(const std::vector<double>& prediction,
    const std::vector<double>& target, std::vector<double>& output) const {
    validateInputs(prediction, target);
    const double scale = 2.0 / static_cast<double>(prediction.size());
    output.resize(prediction.size());
    for (size_t index = 0; index < prediction.size(); ++index) {
        output[index] = scale * (prediction[index] - target[index]);
    }
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
    std::vector<double> result;
    gradientInto(prediction, target, result);
    return result;
}

void MAELoss::gradientInto(const std::vector<double>& prediction,
    const std::vector<double>& target, std::vector<double>& output) const {
    validateInputs(prediction, target);
    const double scale = 1.0 / static_cast<double>(prediction.size());
    output.resize(prediction.size());
    for (size_t index = 0; index < prediction.size(); ++index) {
        const double error = prediction[index] - target[index];
        output[index] = error > 0.0 ? scale : (error < 0.0 ? -scale : 0.0);
    }
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
    std::vector<double> result;
    gradientInto(prediction, target, result);
    return result;
}

void HuberLoss::gradientInto(const std::vector<double>& prediction,
    const std::vector<double>& target, std::vector<double>& output) const {
    validateInputs(prediction, target);
    const double scale = 1.0 / static_cast<double>(prediction.size());
    output.resize(prediction.size());
    for (size_t index = 0; index < prediction.size(); ++index) {
        const double error = prediction[index] - target[index];
        const double absolute_error = std::abs(error);
        if (absolute_error <= threshold) {
            output[index] = scale * error;
        } else {
            output[index] = scale * threshold * (error > 0.0 ? 1.0 : -1.0);
        }
    }
}

double HuberLoss::delta() const {
    return threshold;
}
