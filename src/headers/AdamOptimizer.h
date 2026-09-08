#ifndef ADAM_OPTIMIZER_H
#define ADAM_OPTIMIZER_H

#include "Optimizer.h"
#include <cstddef>
#include <vector>

class AdamOptimizer final : public Optimizer {
public:
    AdamOptimizer(
        double learning_rate,
        double beta1 = 0.9,
        double beta2 = 0.999,
        double epsilon = 1e-8
    );

    void update(
        std::vector<DenseLayer>& layers,
        double gradient_scale = 1.0
    ) override;

    double learningRate() const;
    double beta1() const;
    double beta2() const;
    size_t stateBytes() const;

private:
    friend class OptimizerSerialization;

    struct LayerState {
        std::vector<std::vector<double>> first_moment;
        std::vector<std::vector<double>> second_moment;
        std::vector<double> bias_first_moment;
        std::vector<double> bias_second_moment;
    };

    void ensureState(const std::vector<DenseLayer>& layers);

    double learning_rate;
    double first_decay;
    double second_decay;
    double epsilon;
    std::size_t update_count = 0;
    std::vector<LayerState> states;
};

#endif // ADAM_OPTIMIZER_H
