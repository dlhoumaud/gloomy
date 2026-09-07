#ifndef TRAINING_SAMPLE_H
#define TRAINING_SAMPLE_H

#include <cstddef>
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

#endif // TRAINING_SAMPLE_H
