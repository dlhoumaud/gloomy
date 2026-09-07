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
    double predict(std::vector<double> &sequence);
    void clear();

private:
    std::vector<DenseLayer> layers;
};

#endif // NEURAL_NETWORK_H
