#ifndef OPTIMIZER_SERIALIZATION_H
#define OPTIMIZER_SERIALIZATION_H

#include "Optimizer.h"
#include <string>

class OptimizerSerialization {
public:
    static void save(const std::string& path, const Optimizer& optimizer);
    static void load(const std::string& path, Optimizer& optimizer, const std::vector<DenseLayer>& layers);
};

#endif
