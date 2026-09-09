#include "headers/PageHinkleyDetector.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

PageHinkleyDetector::PageHinkleyDetector(double delta, double lambda)
    : delta_tolerance(delta), lambda_threshold(lambda) {
    if (delta < 0.0 || !std::isfinite(delta)) {
        throw std::invalid_argument("Page-Hinkley delta must be finite and non-negative");
    }
    if (lambda <= 0.0 || !std::isfinite(lambda)) {
        throw std::invalid_argument("Page-Hinkley lambda must be finite and positive");
    }
}

bool PageHinkleyDetector::update(double value) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument("Page-Hinkley detector received a non-finite value");
    }

    ++sample_count;
    running_mean += (value - running_mean) / static_cast<double>(sample_count);
    cumulative_sum += value - running_mean - delta_tolerance;
    minimum_cumulative_sum = std::min(minimum_cumulative_sum, cumulative_sum);

    drift_flag = (cumulative_sum - minimum_cumulative_sum) > lambda_threshold;
    return drift_flag;
}

bool PageHinkleyDetector::driftDetected() const {
    return drift_flag;
}

double PageHinkleyDetector::delta() const {
    return delta_tolerance;
}

double PageHinkleyDetector::lambda() const {
    return lambda_threshold;
}

std::size_t PageHinkleyDetector::count() const {
    return sample_count;
}

void PageHinkleyDetector::reset() {
    running_mean = 0.0;
    sample_count = 0;
    cumulative_sum = 0.0;
    minimum_cumulative_sum = 0.0;
    drift_flag = false;
}
