#ifndef TRAINING_SCHEDULER_H
#define TRAINING_SCHEDULER_H

#include <cstddef>

class TrainingScheduler {
public:
    virtual ~TrainingScheduler() = default;
    virtual bool shouldTrain(double error) = 0;
};

class EverySampleScheduler final : public TrainingScheduler {
public:
    bool shouldTrain(double error) override;
};

class EveryNScheduler final : public TrainingScheduler {
public:
    explicit EveryNScheduler(size_t interval);

    bool shouldTrain(double error) override;
    size_t interval() const;

private:
    size_t training_interval;
    size_t observations = 0;
};

class OnHighErrorScheduler final : public TrainingScheduler {
public:
    explicit OnHighErrorScheduler(double threshold);

    bool shouldTrain(double error) override;
    double threshold() const;

private:
    double error_threshold;
};

#endif // TRAINING_SCHEDULER_H
