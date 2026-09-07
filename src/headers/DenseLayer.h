/**
 * @ Author: GloomShade
 * @ Create Time: 2025-01-03 08:20:18
 * @ Modified by: GloomShade
 * @ Modified time: 2025-01-03 12:03:53
 * @ Description:
 */

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>
#include <string>

class DenseLayer {
public:
    DenseLayer(int input_size, int output_size);
    void set_algorithm(std::string algo);
    void set_post_algorithm(std::string algo);
    std::vector<double> forward(const std::vector<double> &inputs);
    double sigmoid_derivative(double x);

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> bias;
    std::vector<double> inputs;
    std::string algorithm = "none";
    std::string post_algorithm = "none";

    static double sigmoid(double x);
    static double relu(double x);
    static double tanhActivation(double x);
    static double tanhDerivative(double x);
    static double leakyRelu(double x, double alpha = 0.01);
    static std::vector<double> softmax(const std::vector<double> &inputs);
};

#endif // DENSE_LAYER_H
