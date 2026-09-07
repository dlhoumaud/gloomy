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
    std::vector<double> forward(const std::vector<double>& inputs);
    std::vector<double> backward(const std::vector<double>& gradient_output);
    void zeroGradients();

    std::vector<std::vector<double>>& weights();
    const std::vector<std::vector<double>>& weights() const;
    std::vector<double>& bias();
    const std::vector<double>& bias() const;
    const std::vector<std::vector<double>>& weightGradients() const;
    const std::vector<double>& biasGradients() const;
    double sigmoid_derivative(double x);

private:
    std::vector<std::vector<double>> weights_data;
    std::vector<double> bias_data;
    std::vector<double> inputs;
    std::vector<double> preActivations;
    std::vector<double> outputs;
    std::vector<std::vector<double>> weight_gradients;
    std::vector<double> bias_gradients;
    bool has_forward_cache = false;
    std::string algorithm = "none";
    std::string post_algorithm = "none";

    static double sigmoid(double x);
    static double relu(double x);
    static double tanhActivation(double x);
    static double tanhDerivative(double x);
    static double leakyRelu(double x, double alpha = 0.01);
    static double activationDerivative(const std::string& algorithm, double x);
    static std::vector<double> softmax(const std::vector<double> &inputs);
};

#endif // DENSE_LAYER_H
