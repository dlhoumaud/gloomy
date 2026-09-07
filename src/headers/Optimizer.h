#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "DenseLayer.h"
#include <vector>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(
        std::vector<DenseLayer>& layers,
        double gradient_scale = 1.0
    ) = 0;
};

class SGDOptimizer final : public Optimizer {
public:
    explicit SGDOptimizer(double learning_rate);

    void update(
        std::vector<DenseLayer>& layers,
        double gradient_scale = 1.0
    ) override;
    double learningRate() const;
    void setLearningRate(double learning_rate);

private:
    double learning_rate;
};

#endif // OPTIMIZER_H
