#ifndef LEARNING_MEMORY_SERIALIZATION_H
#define LEARNING_MEMORY_SERIALIZATION_H

#include "LearningMemory.h"
#include <memory>
#include <string>

class LearningMemorySerialization {
public:
    static void save(const std::string& path, const LearningMemory& memory);

    // Reconstructs the concrete strategy that produced the file (FIFO,
    // Reservoir, Prioritized, Novelty or Hybrid) with its capacity,
    // hyperparameters, stored samples and RNG state, when the strategy uses
    // one.
    static std::unique_ptr<LearningMemory> load(const std::string& path);
};

#endif // LEARNING_MEMORY_SERIALIZATION_H
