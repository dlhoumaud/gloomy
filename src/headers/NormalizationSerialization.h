#ifndef NORMALIZATION_SERIALIZATION_H
#define NORMALIZATION_SERIALIZATION_H

#include "Normalization.h"
#include <string>

class NormalizationSerialization {
public:
    static void save(const std::string& path, const StreamingNormalizer& normalizer);
    static void load(const std::string& path, StreamingNormalizer& normalizer);
};

#endif // NORMALIZATION_SERIALIZATION_H
