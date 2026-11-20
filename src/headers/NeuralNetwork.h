/**
 * @ Author: GloomShade
 * @ Create Time: 2025-01-03 08:20:18
 * @ Modified by: GloomShade
 * @ Modified time: 2025-01-03 12:03:08
 * @ Description:
 */

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "DenseLayer.h"
#include <vector>
#include <string>

class NeuralNetwork {
public:
    std::string algorithm = "none";
    std::string post_algorithm = "none";

    void addLayer(int input_size, int output_size);
    std::vector<double> forward(const std::vector<double>& inputs);
    const std::vector<double>& forwardRef(const std::vector<double>& inputs);
    const std::vector<double>& forwardInferenceRef(const std::vector<double>& inputs);
    std::vector<double> backward(const std::vector<double>& gradient_output);
    const std::vector<double>& backwardRef(const std::vector<double>& gradient_output);
    void zeroGradients();
    std::vector<DenseLayer>& layers();
    const std::vector<DenseLayer>& layers() const;
    double predict(const std::vector<double>& sequence);
    void clear();
    std::size_t allocatedBytes() const;

private:
    std::vector<DenseLayer> layers_data;
    const std::vector<double>& forwardImpl(const std::vector<double>& inputs, bool cache_inputs);
};

#endif // NEURAL_NETWORK_H
