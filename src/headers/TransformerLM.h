#ifndef TRANSFORMER_LM_H
#define TRANSFORMER_LM_H

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <vector>

struct TransformerConfig {
    std::size_t context_length = 32;
    std::size_t model_dimension = 32;
    std::size_t attention_heads = 4;
    std::size_t layer_count = 1;
    double learning_rate = 0.001;
    double weight_decay = 0.01;
    std::uint32_t seed = 1234;
};

struct LanguageSequence {
    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> target;
    std::vector<std::uint8_t> loss_mask;
};

struct LanguageMetrics {
    double cross_entropy = 0.0;
    double perplexity = 0.0;
    std::size_t token_count = 0;
};

struct WeightQuantizationMetrics {
    std::size_t bits = 0;
    double maximum_absolute_error = 0.0;
    double mean_absolute_error = 0.0;
};

class TransformerLM {
public:
    TransformerLM(std::size_t vocabulary_size, TransformerConfig config);
    double trainBatch(const std::vector<LanguageSequence>& batch);
    double gradientCheck(const std::vector<LanguageSequence>& batch, double epsilon = 1e-5) const;
    WeightQuantizationMetrics quantizeWeights(std::size_t bits);
    void shuffleTrainingSequences(std::vector<LanguageSequence>& sequences);
    void shuffleSequenceIndices(std::vector<std::size_t>& indices);
    LanguageMetrics evaluate(const std::vector<LanguageSequence>& sequences) const;
    std::vector<double> nextTokenLogits(const std::vector<std::uint32_t>& context) const;
    std::size_t vocabularySize() const;
    std::size_t parameterCount() const;
    std::size_t contextLength() const;
    const TransformerConfig& config() const;
    std::uint64_t updateCount() const;
    void writeCheckpoint(std::ostream& stream) const;
    static TransformerLM readCheckpoint(std::istream& stream);

private:
    friend struct TransformerExecutor;
    struct Parameter {
        std::vector<double> value, first_moment, second_moment;
        bool decay = true;
    };
    struct BlockParameters {
        std::size_t norm1_scale, norm1_bias;
        std::size_t qkv_weight, qkv_bias;
        std::size_t attention_weight, attention_bias;
        std::size_t norm2_scale, norm2_bias;
        std::size_t feed_forward_in_weight, feed_forward_in_bias;
        std::size_t feed_forward_out_weight, feed_forward_out_bias;
    };
    std::size_t addParameter(std::size_t size, bool decay, double initial_value = 0.0);
    void initializeParameters();
    std::size_t vocabulary_size;
    TransformerConfig model_config;
    std::size_t token_embedding, position_embedding;
    std::vector<BlockParameters> blocks;
    std::size_t final_norm_scale, final_norm_bias, output_weight, output_bias;
    std::vector<Parameter> parameters;
    std::uint64_t update_count = 0;
    mutable std::mt19937 random_generator;
};

#endif
