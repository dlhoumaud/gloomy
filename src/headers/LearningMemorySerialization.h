#ifndef LEARNING_MEMORY_SERIALIZATION_H
#define LEARNING_MEMORY_SERIALIZATION_H

#include "LearningMemory.h"
#include <memory>
#include <string>

class LearningMemorySerialization {
public:
    static std::string serialize(const LearningMemory& memory);
    static std::unique_ptr<LearningMemory> deserialize(const std::string& bytes);
    static void validateDimensions(const LearningMemory& memory, size_t input_size, size_t target_size);
};

#endif
