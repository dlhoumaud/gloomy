#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "DenseLayer.h"
#include <vector>
#include <istream>
#include <ostream>
#include <string>

class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void update(
        std::vector<DenseLayer>& layers,
        double gradient_scale = 1.0
    ) = 0;
    virtual std::string checkpointType() const = 0;
    virtual void writeCheckpoint(std::ostream& stream) const = 0;
    virtual void readCheckpoint(std::istream& stream, const std::vector<DenseLayer>& layers) = 0;
    virtual std::size_t allocatedBytes() const { return 0; }
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
    std::string checkpointType() const override;
    void writeCheckpoint(std::ostream& stream) const override;
    void readCheckpoint(std::istream& stream, const std::vector<DenseLayer>& layers) override;
    std::size_t allocatedBytes() const override;

private:
    double learning_rate;
};

#endif // OPTIMIZER_H
