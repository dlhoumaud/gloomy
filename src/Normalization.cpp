#include "headers/Normalization.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

StreamingNormalizer::StreamingNormalizer(size_t dimensions)
    : means(dimensions, 0.0),
      moments(dimensions, 0.0),
      minimum_values(dimensions, std::numeric_limits<double>::infinity()),
      maximum_values(dimensions, -std::numeric_limits<double>::infinity()) {
    if (dimensions == 0) {
        throw std::invalid_argument("Normalizer dimensions must be positive");
    }
}

void StreamingNormalizer::update(const std::vector<double>& values) {
    validate(values);
    ++value_count;

    for (size_t index = 0; index < values.size(); ++index) {
        const double delta = values[index] - means[index];
        means[index] += delta / static_cast<double>(value_count);
        const double updated_delta = values[index] - means[index];
        moments[index] += delta * updated_delta;
        minimum_values[index] = std::min(minimum_values[index], values[index]);
        maximum_values[index] = std::max(maximum_values[index], values[index]);
    }
}

std::vector<double> StreamingNormalizer::normalize(const std::vector<double>& values) const {
    validate(values);
    if (value_count == 0) {
        throw std::logic_error("Cannot normalize without statistics");
    }

    std::vector<double> normalized(values.size(), 0.0);
    for (size_t index = 0; index < values.size(); ++index) {
        const double variance_value = moments[index] / static_cast<double>(value_count);
        const double standard_deviation = std::sqrt(variance_value);
        normalized[index] = standard_deviation > 0.0
            ? (values[index] - means[index]) / standard_deviation
            : 0.0;
    }
    return normalized;
}

void StreamingNormalizer::clear() {
    value_count = 0;
    std::fill(means.begin(), means.end(), 0.0);
    std::fill(moments.begin(), moments.end(), 0.0);
    std::fill(
        minimum_values.begin(),
        minimum_values.end(),
        std::numeric_limits<double>::infinity()
    );
    std::fill(
        maximum_values.begin(),
        maximum_values.end(),
        -std::numeric_limits<double>::infinity()
    );
}

void StreamingNormalizer::restore(
    size_t count,
    const std::vector<double>& restored_means,
    const std::vector<double>& variances,
    const std::vector<double>& restored_minimum,
    const std::vector<double>& restored_maximum
) {
    if (count == 0 || restored_means.size() != means.size() ||
        variances.size() != means.size() || restored_minimum.size() != means.size() ||
        restored_maximum.size() != means.size()) {
        throw std::invalid_argument("Invalid normalizer statistics");
    }
    for (size_t index = 0; index < means.size(); ++index) {
        if (!std::isfinite(restored_means[index]) || !std::isfinite(variances[index]) ||
            variances[index] < 0.0 || !std::isfinite(restored_minimum[index]) ||
            !std::isfinite(restored_maximum[index]) || restored_minimum[index] > restored_maximum[index]) {
            throw std::invalid_argument("Normalizer statistics must be finite and consistent");
        }
    }
    value_count = count;
    means = restored_means;
    moments.resize(variances.size());
    for (size_t index = 0; index < variances.size(); ++index) {
        moments[index] = variances[index] * static_cast<double>(count);
    }
    minimum_values = restored_minimum;
    maximum_values = restored_maximum;
}

size_t StreamingNormalizer::count() const {
    return value_count;
}

size_t StreamingNormalizer::dimensions() const {
    return means.size();
}

const std::vector<double>& StreamingNormalizer::mean() const {
    return means;
}

std::vector<double> StreamingNormalizer::variance() const {
    std::vector<double> result(moments.size(), 0.0);
    if (value_count == 0) {
        return result;
    }
    for (size_t index = 0; index < moments.size(); ++index) {
        result[index] = moments[index] / static_cast<double>(value_count);
    }
    return result;
}

const std::vector<double>& StreamingNormalizer::minimum() const {
    return minimum_values;
}

const std::vector<double>& StreamingNormalizer::maximum() const {
    return maximum_values;
}

void StreamingNormalizer::validate(const std::vector<double>& values) const {
    if (values.size() != means.size()) {
        throw std::invalid_argument("Normalizer input dimensions must match");
    }
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("Normalizer values must be finite");
        }
    }
}
