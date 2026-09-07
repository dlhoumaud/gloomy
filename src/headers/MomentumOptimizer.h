#ifndef MOMENTUM_OPTIMIZER_H
#define MOMENTUM_OPTIMIZER_H

#include "Optimizer.h"
#include <cstddef>
#include <vector>

class MomentumOptimizer final : public Optimizer {
public:
    MomentumOptimizer(double learning_rate, double momentum = 0.9);

    void update(
        std::vector<DenseLayer>& layers,
        double gradient_scale = 1.0
    ) override;

    double learningRate() const;
    double momentum() const;
    size_t stateBytes() const;

private:
    struct LayerState {
        std::vector<std::vector<double>> weight_velocity;
        std::vector<double> bias_velocity;
    };

    void ensureState(const std::vector<DenseLayer>& layers);

    double learning_rate;
    double momentum_factor;
    std::vector<LayerState> states;
};

#endif // MOMENTUM_OPTIMIZER_H
