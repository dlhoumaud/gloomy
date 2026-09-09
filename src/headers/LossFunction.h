#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include <vector>

class LossFunction {
public:
    virtual ~LossFunction() = default;

    virtual double compute(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const = 0;

    virtual std::vector<double> gradient(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const = 0;
};

class MSELoss final : public LossFunction {
public:
    double compute(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;

    std::vector<double> gradient(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;
};

class MAELoss final : public LossFunction {
public:
    double compute(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;

    std::vector<double> gradient(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;
};

class HuberLoss final : public LossFunction {
public:
    explicit HuberLoss(double delta = 1.0);

    double compute(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;

    std::vector<double> gradient(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;

    double delta() const;

private:
    double threshold;
};

// Perte d'entropie croisee categorielle : suppose `prediction` deja passee
// par softmax (post_algorithm="softmax") et `target` une distribution
// (typiquement one-hot). Contrairement a MSELoss/MAELoss/HuberLoss, ne
// divise PAS par le nombre de classes : cette division a un sens pour une
// erreur de regression par dimension, pas pour une distribution de
// probabilite ou elle changerait arbitrairement l'echelle de la perte selon
// K (voir docs/losses.md). gradient() retourne dL/dp_i = -target_i/p_i,
// destine a etre compose avec le jacobien softmax deja implemente dans
// DenseLayer::backward (meme principe que testSoftmaxGradientCheck).
class CrossEntropyLoss final : public LossFunction {
public:
    double compute(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;

    std::vector<double> gradient(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    ) const override;
};

#endif // LOSS_FUNCTION_H
