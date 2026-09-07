#ifndef TRAINING_SAMPLE_SERIALIZATION_H
#define TRAINING_SAMPLE_SERIALIZATION_H

#include "TrainingSample.h"
#include <string>
#include <vector>

class TrainingSampleSerialization {
public:
    static void save(
        const std::string& path,
        const std::vector<TrainingSample>& samples
    );

    static std::vector<TrainingSample> load(const std::string& path);
};

#endif // TRAINING_SAMPLE_SERIALIZATION_H
