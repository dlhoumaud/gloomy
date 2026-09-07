#ifndef METRICS_H
#define METRICS_H

#include <vector>

struct RegressionMetrics {
    double mae;
    double rmse;
};

class Metrics {
public:
    static RegressionMetrics regression(
        const std::vector<double>& prediction,
        const std::vector<double>& target
    );

    static double forgetting(
        double performance_before,
        double performance_after
    );
};

#endif // METRICS_H
