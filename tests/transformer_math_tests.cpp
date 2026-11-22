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
}

int main() {
    testTransformerGradients();
    testTransformerStabilityAndReproducibility();
    testNonFiniteUpdateIsRejectedAtomically();
}
