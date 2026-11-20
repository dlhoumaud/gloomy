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
#include <cstddef>

class DenseMatrixRowView {
public:
    DenseMatrixRowView(double* values, std::size_t size) : values(values), row_size(size) {}
    double& operator[](std::size_t column) const { return values[column]; }
    double* begin() const { return values; }
    double* end() const { return values + row_size; }
    std::size_t size() const { return row_size; }

private:
    double* values;
    std::size_t row_size;
};

class ConstDenseMatrixRowView {
public:
    ConstDenseMatrixRowView(const double* values, std::size_t size) : values(values), row_size(size) {}
    const double& operator[](std::size_t column) const { return values[column]; }
    const double* begin() const { return values; }
    const double* end() const { return values + row_size; }
    std::size_t size() const { return row_size; }

private:
    const double* values;
    std::size_t row_size;
};

class DenseMatrixView {
public:
    DenseMatrixView(double* values, std::size_t row_count, std::size_t column_count)
        : values(values), rows(row_count), columns(column_count) {}
    DenseMatrixRowView operator[](std::size_t row) const {
        return DenseMatrixRowView(values + row * columns, columns);
    }
    std::size_t size() const { return rows; }
    std::size_t columnsCount() const { return columns; }

private:
    double* values;
    std::size_t rows;
    std::size_t columns;
};

class ConstDenseMatrixView {
public:
    ConstDenseMatrixView(const double* values, std::size_t row_count, std::size_t column_count)
        : values(values), rows(row_count), columns(column_count) {}
    ConstDenseMatrixRowView operator[](std::size_t row) const {
        return ConstDenseMatrixRowView(values + row * columns, columns);
    }
    std::size_t size() const { return rows; }
    std::size_t columnsCount() const { return columns; }

private:
    const double* values;
    std::size_t rows;
    std::size_t columns;
};

class DenseLayer {
public:
    DenseLayer(int input_size, int output_size);
    void set_algorithm(std::string algo);
    void set_post_algorithm(std::string algo);
    std::vector<double> forward(const std::vector<double>& inputs);
    const std::vector<double>& forwardRef(const std::vector<double>& inputs);
    const std::vector<double>& forwardInferenceRef(const std::vector<double>& inputs);
    std::vector<double> backward(const std::vector<double>& gradient_output);
    const std::vector<double>& backwardRef(const std::vector<double>& gradient_output);
    void zeroGradients();

    DenseMatrixView weights();
    ConstDenseMatrixView weights() const;
    std::vector<double>& weightValues();
    const std::vector<double>& weightValues() const;
    std::vector<double>& bias();
    const std::vector<double>& bias() const;
    ConstDenseMatrixView weightGradients() const;
    const std::vector<double>& weightGradientValues() const;
    const std::vector<double>& biasGradients() const;
    double sigmoid_derivative(double x);

private:
    std::size_t input_size = 0;
    std::size_t output_size = 0;
    std::vector<double> weights_data;
    std::vector<double> bias_data;
    std::vector<double> inputs;
    std::vector<double> preActivations;
    std::vector<double> outputs;
    std::vector<double> gradient_after_activation;
    std::vector<double> gradient_inputs;
    std::vector<double> weight_gradients;
    std::vector<double> bias_gradients;
    bool has_forward_cache = false;
    std::string algorithm = "none";
    std::string post_algorithm = "none";

    const std::vector<double>& forwardImpl(const std::vector<double>& inputs, bool cache_inputs);

    static double sigmoid(double x);
    static double relu(double x);
    static double tanhActivation(double x);
    static double tanhDerivative(double x);
    static double leakyRelu(double x, double alpha = 0.01);
    static double activationDerivative(const std::string& algorithm, double x);
    static void softmax(std::vector<double>& values);
};

#endif // DENSE_LAYER_H
