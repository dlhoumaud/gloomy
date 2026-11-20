#include "headers/LearningEngine.h"
#include "headers/LossFunction.h"
#include "headers/NeuralNetwork.h"
#include "headers/Optimizer.h"
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>
#include <sys/resource.h>
#include <vector>

namespace {
struct alignas(std::max_align_t) AllocationHeader {
    std::size_t size;
    bool measured;
};

bool measure_allocations = false;
std::size_t allocation_count = 0;
std::size_t allocated_bytes = 0;
std::size_t live_measured_bytes = 0;
std::size_t peak_live_measured_bytes = 0;

std::vector<TrainingSample> makeDataset(std::size_t count, std::size_t start) {
    std::vector<TrainingSample> samples;
    samples.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const double x = static_cast<double>(index + start) / 1000.0;
        samples.push_back({{x}, {2.0 * x + 1.0}});
    }
    return samples;
}

double validationMetrics(
    NeuralNetwork& network,
    const LossFunction& loss,
    const std::vector<TrainingSample>& validation,
    double& mae,
    double& rmse
) {
    double total_loss = 0.0;
    double absolute_error = 0.0;
    double squared_error = 0.0;
    for (const TrainingSample& sample : validation) {
        const std::vector<double> prediction = network.forward(sample.input);
        const double error = prediction.front() - sample.target.front();
        total_loss += loss.compute(prediction, sample.target);
        absolute_error += std::abs(error);
        squared_error += error * error;
    }
    const double count = static_cast<double>(validation.size());
    mae = absolute_error / count;
    rmse = std::sqrt(squared_error / count);
    return total_loss / count;
}

std::size_t processPeakRssBytes() {
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__)
    return static_cast<std::size_t>(usage.ru_maxrss);
#else
    return static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
}
}

void* operator new(std::size_t requested_size) {
    const std::size_t size = requested_size == 0 ? 1 : requested_size;
    if (size > static_cast<std::size_t>(-1) - sizeof(AllocationHeader)) throw std::bad_alloc();
    auto* header = static_cast<AllocationHeader*>(std::malloc(sizeof(AllocationHeader) + size));
    if (!header) throw std::bad_alloc();
    header->size = size;
    header->measured = measure_allocations;
    if (measure_allocations) {
        ++allocation_count;
        allocated_bytes += size;
        live_measured_bytes += size;
        if (live_measured_bytes > peak_live_measured_bytes)
            peak_live_measured_bytes = live_measured_bytes;
    }
    return header + 1;
}

void* operator new[](std::size_t size) {
    return ::operator new(size);
}

[[gnu::noinline]] void releaseAllocation(void* memory) noexcept {
    if (!memory) return;
    auto* header = static_cast<AllocationHeader*>(memory) - 1;
    if (header->measured) live_measured_bytes -= header->size;
    std::free(header);
}

void operator delete(void* memory) noexcept { releaseAllocation(memory); }
void operator delete[](void* memory) noexcept { releaseAllocation(memory); }
void operator delete(void* memory, std::size_t) noexcept { releaseAllocation(memory); }
void operator delete[](void* memory, std::size_t) noexcept { releaseAllocation(memory); }

int main(int argc, char* argv[]) {
    if (argc != 3) return 2;
    const std::string revision = argv[1];
    const unsigned int seed = static_cast<unsigned int>(std::stoul(argv[2]));
    const auto training = makeDataset(400, 0);
    const auto validation = makeDataset(80, 400);

    std::srand(seed);
    NeuralNetwork network;
    network.algorithm = "none";
    network.addLayer(1, 8);
    network.addLayer(8, 1);
    MSELoss loss;
    SGDOptimizer optimizer(0.01);
    LearningEngine engine(network, loss, optimizer);

    measure_allocations = true;
    const auto start = std::chrono::steady_clock::now();
    engine.train(training, 1, 8);
    const auto end = std::chrono::steady_clock::now();
    measure_allocations = false;

    double mae = 0.0;
    double rmse = 0.0;
    const double validation_loss = validationMetrics(network, loss, validation, mae, rmse);
    const double training_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << revision << ',' << seed << ',' << training_time_ms << ',' << validation_loss << ','
              << mae << ',' << rmse << ',' << live_measured_bytes << ',' << allocation_count << ','
              << allocated_bytes << ',' << peak_live_measured_bytes << ',' << processPeakRssBytes()
              << ',' << ((training.size() + 7) / 8) << '\n';
    return 0;
}
