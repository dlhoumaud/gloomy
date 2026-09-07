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

#endif // LOSS_FUNCTION_H
