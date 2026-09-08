#ifndef OPTIMIZER_SERIALIZATION_H
#define OPTIMIZER_SERIALIZATION_H

#include "DenseLayer.h"
#include "Optimizer.h"
#include <memory>
#include <string>
#include <vector>

class OptimizerSerialization {
public:
    static void save(const std::string& path, const Optimizer& optimizer);

    // Reconstructs the optimizer that produced the file and restores its
    // internal state. `layers` is the network the optimizer will operate on:
    // its shape is checked against the saved state before restoring it, so
    // that a buffer computed for a different architecture is rejected
    // instead of silently corrupting training.
    static std::unique_ptr<Optimizer> load(
        const std::string& path,
        const std::vector<DenseLayer>& layers
    );
};

#endif // OPTIMIZER_SERIALIZATION_H
