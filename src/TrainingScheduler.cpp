#include "headers/TrainingScheduler.h"
#include <cmath>
#include <stdexcept>

bool EverySampleScheduler::shouldTrain(double error) {
    if (!std::isfinite(error) || error < 0.0) {
        throw std::invalid_argument("Training error must be finite and non-negative");
    }
    return true;
}

EveryNScheduler::EveryNScheduler(size_t interval)
    : training_interval(interval) {
    if (interval == 0) {
        throw std::invalid_argument("Training interval must be positive");
    }
}

bool EveryNScheduler::shouldTrain(double error) {
    if (!std::isfinite(error) || error < 0.0) {
        throw std::invalid_argument("Training error must be finite and non-negative");
    }
    ++observations;
    if (observations == training_interval) {
        observations = 0;
        return true;
    }
    return false;
}

size_t EveryNScheduler::interval() const {
    return training_interval;
}

OnHighErrorScheduler::OnHighErrorScheduler(double threshold)
    : error_threshold(threshold) {
    if (!std::isfinite(threshold) || threshold < 0.0) {
        throw std::invalid_argument("Error threshold must be finite and non-negative");
    }
}

bool OnHighErrorScheduler::shouldTrain(double error) {
    if (!std::isfinite(error) || error < 0.0) {
        throw std::invalid_argument("Training error must be finite and non-negative");
    }
    return error >= error_threshold;
}

double OnHighErrorScheduler::threshold() const {
    return error_threshold;
}
