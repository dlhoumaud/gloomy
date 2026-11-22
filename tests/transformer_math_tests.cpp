#include "TransformerLM.h"

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
TransformerConfig smallConfig() {
    TransformerConfig config;
    config.context_length = 2;
    config.model_dimension = 4;
    config.attention_heads = 2;
    config.layer_count = 1;
    config.seed = 73;
    config.learning_rate = 0.001;
    return config;
}

std::string checkpoint(const TransformerLM& model) {
    std::ostringstream stream(std::ios::binary);
    model.writeCheckpoint(stream);
    return stream.str();
}

void testTransformerGradients() {
    TransformerLM model(259, smallConfig());
    LanguageSequence sequence{{0, 3}, {3, 1}, {1, 1}};
    const double maximum_relative_error = model.gradientCheck({sequence});
    assert(std::isfinite(maximum_relative_error));
    assert(maximum_relative_error < 5e-3);
}

void testTransformerStabilityAndReproducibility() {
    TransformerLM first(259, smallConfig());
    TransformerLM second(259, smallConfig());
    LanguageSequence sequence{{0, 3}, {3, 1}, {1, 1}};
    const auto logits = first.nextTokenLogits(sequence.input);
    for (double value : logits) assert(std::isfinite(value));

    for (int update = 0; update < 3; ++update) {
        const double first_loss = first.trainBatch({sequence});
        const double second_loss = second.trainBatch({sequence});
        assert(std::isfinite(first_loss));
        assert(first_loss == second_loss);
    }
    assert(checkpoint(first) == checkpoint(second));
}

void testNonFiniteUpdateIsRejectedAtomically() {
    auto config = smallConfig();
    config.learning_rate = 1e308;
    config.weight_decay = 1e308;
    TransformerLM model(259, config);
    LanguageSequence sequence{{0, 3}, {3, 1}, {1, 1}};
    const auto before = checkpoint(model);
    bool rejected = false;
    try {
        model.trainBatch({sequence});
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
    assert(model.updateCount() == 0);
    assert(checkpoint(model) == before);
}

void testReducedPrecisionInferenceModes() {
    TransformerLM model(259, smallConfig());
    LanguageSequence sequence{{0, 3}, {3, 1}, {1, 1}};
    for (std::size_t bits : {8U, 16U, 32U}) {
        model.setInferencePrecision(bits);
        const auto metrics = model.evaluate({sequence});
        assert(std::isfinite(metrics.cross_entropy));
        const auto logits = model.nextTokenLogits(sequence.input);
        for (double value : logits) assert(std::isfinite(value));
        bool training_rejected = false;
        try {
            model.trainBatch({sequence});
        } catch (const std::logic_error&) {
            training_rejected = true;
        }
        assert(training_rejected);
    }
    model.setInferencePrecision(0);
    assert(std::isfinite(model.trainBatch({sequence})));
    bool invalid_precision_rejected = false;
    try {
        model.setInferencePrecision(4);
    } catch (const std::invalid_argument&) {
        invalid_precision_rejected = true;
    }
    assert(invalid_precision_rejected);
}

void testCompactCheckpointRoundTrip() {
    TransformerLM model(259, smallConfig());
    std::ostringstream compact(std::ios::binary);
    const auto metrics = model.writeCompactCheckpoint(compact, 8);
    assert(metrics.bits == 8);
    assert(std::isfinite(metrics.maximum_absolute_error));
    std::istringstream input(compact.str(), std::ios::binary);
    auto restored = TransformerLM::readCheckpoint(input);
    const auto before = model.nextTokenLogits({0, 3});
    const auto after = restored.nextTokenLogits({0, 3});
    assert(before.size() == after.size());
    for (std::size_t i = 0; i < before.size(); ++i) assert(std::isfinite(after[i]));
    bool rejected = false;
    try { restored.trainBatch({LanguageSequence{{0, 3}, {3, 1}, {1, 1}}}); }
    catch (const std::logic_error&) { rejected = true; }
    assert(rejected);
}
}

int main() {
    testTransformerGradients();
    testTransformerStabilityAndReproducibility();
    testNonFiniteUpdateIsRejectedAtomically();
    testReducedPrecisionInferenceModes();
    testCompactCheckpointRoundTrip();
}
