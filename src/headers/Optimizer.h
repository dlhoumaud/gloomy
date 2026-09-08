#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "DenseLayer.h"
#include <vector>

// Valide que tous les gradients (poids et biais) des couches fournies sont
// finis, avant qu'un optimiseur ne les applique. Rejette NaN/infini avec un
// message clair, sur le meme principe que LossFunction::validateInputs et
// DenseLayer::forward/backward (voir docs/roadmap.md, « Priorité moyenne :
// robustesse mathématique »). Partagee par SGD/Momentum/Adam pour eviter la
// triplication ; definie une seule fois dans Optimizer.cpp.
void validateFiniteGradients(const std::vector<DenseLayer>& layers);

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
