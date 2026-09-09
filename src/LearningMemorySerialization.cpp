#include "headers/LearningMemorySerialization.h"
#include "headers/FIFOMemory.h"
#include "headers/ReservoirMemory.h"
#include "headers/PrioritizedMemory.h"
#include "headers/NoveltyMemory.h"
#include "headers/HybridMemory.h"
#include "headers/QuantizedFIFOMemory.h"
#include "headers/QuantizedInt8FIFOMemory.h"
#include "headers/TrainingSampleQuantization.h"
#include "headers/Int8TrainingSampleQuantization.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
constexpr char magic[] = "GLOOMYLM";
// Version 4 : ajout de beta_annealing_rate et exploration_epsilon pour
// PrioritizedMemory (annealing de la correction de biais et exploration
// controlee des echantillons de faible priorite, voir docs/memory.md).
// Version 3 : ajout des memoires quantifiees QuantizedFIFOMemory (int16) et
// QuantizedInt8FIFOMemory (int8), avec leurs parametres de calibration
// (scale/zero_point) partages par tous les echantillons stockes (voir
// docs/quantization.md et docs/memory.md). Version 2 : ajout de
// correction_exponent (beta) pour PrioritizedMemory (correction du biais
// d'echantillonnage). Un fichier d'une version anterieure est refuse plutot
// que mal interprete.
constexpr std::uint32_t format_version = 4;
constexpr std::uint32_t strategy_fifo = 0;
constexpr std::uint32_t strategy_reservoir = 1;
constexpr std::uint32_t strategy_prioritized = 2;
constexpr std::uint32_t strategy_novelty = 3;
constexpr std::uint32_t strategy_hybrid = 4;
constexpr std::uint32_t strategy_quantized_fifo_int16 = 5;
constexpr std::uint32_t strategy_quantized_fifo_int8 = 6;
constexpr std::uint64_t maximum_capacity = 1'000'000;
constexpr std::uint64_t maximum_vector_size = 1'000'000;
constexpr std::uint64_t checksum_offset = 1469598103934665603ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;

void writeBytes(std::ostream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write learning memory file");
}

void readBytes(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated learning memory file");
}

std::uint64_t checksum(const std::string& bytes) {
    std::uint64_t result = checksum_offset;
    for (unsigned char byte : bytes) {
        result ^= byte;
        result *= checksum_prime;
    }
    return result;
}

void writeUint32(std::ostream& stream, std::uint32_t value) {
    writeBytes(stream, &value, sizeof(value));
}

void writeUint64(std::ostream& stream, std::uint64_t value) {
    writeBytes(stream, &value, sizeof(value));
}

void writeDouble(std::ostream& stream, double value) {
    writeBytes(stream, &value, sizeof(value));
}

std::uint32_t readUint32(std::istream& stream) {
    std::uint32_t value = 0;
    readBytes(stream, &value, sizeof(value));
    return value;
}

std::uint64_t readUint64(std::istream& stream) {
    std::uint64_t value = 0;
    readBytes(stream, &value, sizeof(value));
    return value;
}

double readDouble(std::istream& stream) {
    double value = 0.0;
    readBytes(stream, &value, sizeof(value));
    return value;
}

void writeString(std::ostream& stream, const std::string& value) {
    const std::uint64_t size = value.size();
    writeBytes(stream, &size, sizeof(size));
    if (!value.empty()) writeBytes(stream, value.data(), value.size());
}

std::string readString(std::istream& stream) {
    std::uint64_t size = 0;
    readBytes(stream, &size, sizeof(size));
    if (size > maximum_vector_size || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Learning memory string size is invalid");
    }
    std::string value(static_cast<size_t>(size), '\0');
    if (!value.empty()) readBytes(stream, value.data(), value.size());
    return value;
}

void writeVector(std::ostream& stream, const std::vector<double>& values) {
    const std::uint64_t size = values.size();
    writeBytes(stream, &size, sizeof(size));
    if (!values.empty()) writeBytes(stream, values.data(), values.size() * sizeof(double));
}

std::vector<double> readVector(std::istream& stream) {
    std::uint64_t size = 0;
    readBytes(stream, &size, sizeof(size));
    if (size > maximum_vector_size || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Learning memory sample vector size is invalid");
    }
    std::vector<double> values(static_cast<size_t>(size));
    if (!values.empty()) readBytes(stream, values.data(), values.size() * sizeof(double));
    return values;
}

void writeSample(std::ostream& stream, const TrainingSample& sample) {
    writeVector(stream, sample.input);
    writeVector(stream, sample.target);
    writeDouble(stream, sample.priority);
    writeDouble(stream, sample.error);
    writeDouble(stream, sample.novelty);
    writeDouble(stream, sample.rarity);
    writeDouble(stream, sample.recency);
    writeDouble(stream, sample.diversity);
    writeUint64(stream, static_cast<std::uint64_t>(sample.age));
    writeUint64(stream, static_cast<std::uint64_t>(sample.usage_count));
}

TrainingSample readSample(std::istream& stream) {
    TrainingSample sample;
    sample.input = readVector(stream);
    sample.target = readVector(stream);
    sample.priority = readDouble(stream);
    sample.error = readDouble(stream);
    sample.novelty = readDouble(stream);
    sample.rarity = readDouble(stream);
    sample.recency = readDouble(stream);
    sample.diversity = readDouble(stream);
    sample.age = static_cast<std::size_t>(readUint64(stream));
    sample.usage_count = static_cast<std::size_t>(readUint64(stream));
    return sample;
}

std::string captureRngState(const std::mt19937& generator) {
    std::mt19937 copy = generator;
    std::ostringstream state_stream;
    state_stream << copy;
    return state_stream.str();
}

void restoreRngState(std::mt19937& generator, const std::string& state) {
    std::istringstream state_stream(state);
    state_stream >> generator;
    if (!state_stream) throw std::runtime_error("Learning memory RNG state is invalid");
}

// Les parametres de calibration (scale/zero_point) sont partages par tous
// les echantillons d'une meme memoire quantifiee (voir TrainingSampleQuantizer
// / Int8TrainingSampleQuantizer : encode() utilise toujours les memes
// parametres, fixes a la construction) : ils ne sont donc ecrits qu'une
// seule fois par memoire, pas repetes par echantillon.
void writeQuantizationParameters(std::ostream& stream, const QuantizationParameters& parameters) {
    writeDouble(stream, parameters.scale);
    writeBytes(stream, &parameters.zero_point, sizeof(parameters.zero_point));
}

QuantizationParameters readQuantizationParameters(std::istream& stream) {
    QuantizationParameters parameters{};
    parameters.scale = readDouble(stream);
    readBytes(stream, &parameters.zero_point, sizeof(parameters.zero_point));
    return parameters;
}

void writeInt16Values(std::ostream& stream, const std::vector<std::int16_t>& values) {
    const std::uint64_t size = values.size();
    writeBytes(stream, &size, sizeof(size));
    if (!values.empty()) writeBytes(stream, values.data(), values.size() * sizeof(std::int16_t));
}

std::vector<std::int16_t> readInt16Values(std::istream& stream) {
    std::uint64_t size = 0;
    readBytes(stream, &size, sizeof(size));
    if (size > maximum_vector_size || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Quantized learning memory vector size is invalid");
    }
    std::vector<std::int16_t> values(static_cast<size_t>(size));
    if (!values.empty()) readBytes(stream, values.data(), values.size() * sizeof(std::int16_t));
    return values;
}

void writeInt8Values(std::ostream& stream, const std::vector<std::int8_t>& values) {
    const std::uint64_t size = values.size();
    writeBytes(stream, &size, sizeof(size));
    if (!values.empty()) writeBytes(stream, values.data(), values.size() * sizeof(std::int8_t));
}

std::vector<std::int8_t> readInt8Values(std::istream& stream) {
    std::uint64_t size = 0;
    readBytes(stream, &size, sizeof(size));
    if (size > maximum_vector_size || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Quantized learning memory vector size is invalid");
    }
    std::vector<std::int8_t> values(static_cast<size_t>(size));
    if (!values.empty()) readBytes(stream, values.data(), values.size() * sizeof(std::int8_t));
    return values;
}

// Metadonnees d'un echantillon quantifie : memes champs que TrainingSample,
// hors input/target (deja geres separement en int16/int8 ci-dessus).
struct QuantizedSampleMetadata {
    double priority = 0.0;
    double error = 0.0;
    double novelty = 0.0;
    double rarity = 0.0;
    double recency = 0.0;
    double diversity = 0.0;
    std::size_t age = 0;
    std::size_t usage_count = 0;
};

void writeQuantizedMetadata(std::ostream& stream, const QuantizedSampleMetadata& metadata) {
    writeDouble(stream, metadata.priority);
    writeDouble(stream, metadata.error);
    writeDouble(stream, metadata.novelty);
    writeDouble(stream, metadata.rarity);
    writeDouble(stream, metadata.recency);
    writeDouble(stream, metadata.diversity);
    writeUint64(stream, static_cast<std::uint64_t>(metadata.age));
    writeUint64(stream, static_cast<std::uint64_t>(metadata.usage_count));
}

QuantizedSampleMetadata readQuantizedMetadata(std::istream& stream) {
    QuantizedSampleMetadata metadata;
    metadata.priority = readDouble(stream);
    metadata.error = readDouble(stream);
    metadata.novelty = readDouble(stream);
    metadata.rarity = readDouble(stream);
    metadata.recency = readDouble(stream);
    metadata.diversity = readDouble(stream);
    metadata.age = static_cast<std::size_t>(readUint64(stream));
    metadata.usage_count = static_cast<std::size_t>(readUint64(stream));
    return metadata;
}
}

void LearningMemorySerialization::save(const std::string& path, const LearningMemory& memory) {
    std::ostringstream payload(std::ios::binary);
    writeBytes(payload, magic, sizeof(magic) - 1);
    writeBytes(payload, &format_version, sizeof(format_version));

    if (const auto* fifo = dynamic_cast<const FIFOMemory*>(&memory)) {
        writeUint32(payload, strategy_fifo);
        writeUint64(payload, fifo->memory_capacity);
        writeUint64(payload, fifo->samples.size());
        for (const TrainingSample& sample : fifo->samples) writeSample(payload, sample);
    } else if (const auto* reservoir = dynamic_cast<const ReservoirMemory*>(&memory)) {
        writeUint32(payload, strategy_reservoir);
        writeUint64(payload, reservoir->memory_capacity);
        writeUint64(payload, static_cast<std::uint64_t>(reservoir->seen_samples));
        writeString(payload, captureRngState(reservoir->generator));
        writeUint64(payload, reservoir->samples.size());
        for (const TrainingSample& sample : reservoir->samples) writeSample(payload, sample);
    } else if (const auto* prioritized = dynamic_cast<const PrioritizedMemory*>(&memory)) {
        writeUint32(payload, strategy_prioritized);
        writeUint64(payload, prioritized->memory_capacity);
        writeDouble(payload, prioritized->priority_exponent);
        writeDouble(payload, prioritized->correction_exponent);
        writeDouble(payload, prioritized->beta_annealing_rate);
        writeDouble(payload, prioritized->exploration_epsilon);
        writeString(payload, captureRngState(prioritized->generator));
        writeUint64(payload, prioritized->samples.size());
        for (const TrainingSample& sample : prioritized->samples) writeSample(payload, sample);
    } else if (const auto* novelty = dynamic_cast<const NoveltyMemory*>(&memory)) {
        writeUint32(payload, strategy_novelty);
        writeUint64(payload, novelty->memory_capacity);
        writeDouble(payload, novelty->distance_threshold);
        writeUint64(payload, novelty->samples.size());
        for (const TrainingSample& sample : novelty->samples) writeSample(payload, sample);
    } else if (const auto* hybrid = dynamic_cast<const HybridMemory*>(&memory)) {
        writeUint32(payload, strategy_hybrid);
        writeUint64(payload, hybrid->memory_capacity);
        const HybridMemoryRatios& ratios = hybrid->memory_ratios;
        writeDouble(payload, ratios.recent);
        writeDouble(payload, ratios.error);
        writeDouble(payload, ratios.novelty);
        writeDouble(payload, ratios.historical);
        writeDouble(payload, hybrid->distance_threshold);
        writeUint64(payload, hybrid->seen_samples);
        writeString(payload, captureRngState(hybrid->generator));
        writeUint64(payload, hybrid->samples.size());
        for (const HybridMemory::StoredSample& stored : hybrid->samples) {
            writeSample(payload, stored.sample);
            writeUint32(payload, static_cast<std::uint32_t>(stored.partition));
        }
    } else if (const auto* quantized16 = dynamic_cast<const QuantizedFIFOMemory*>(&memory)) {
        writeUint32(payload, strategy_quantized_fifo_int16);
        writeUint64(payload, quantized16->memory_capacity);
        writeQuantizationParameters(payload, quantized16->sample_quantizer.inputParameters());
        writeQuantizationParameters(payload, quantized16->sample_quantizer.targetParameters());
        writeUint64(payload, quantized16->input_dimensions);
        writeUint64(payload, quantized16->target_dimensions);
        writeUint64(payload, quantized16->samples.size());
        for (const QuantizedTrainingSample& stored : quantized16->samples) {
            writeInt16Values(payload, stored.input.values);
            writeInt16Values(payload, stored.target.values);
            writeQuantizedMetadata(payload, {
                stored.priority, stored.error, stored.novelty,
                stored.rarity, stored.recency, stored.diversity,
                stored.age, stored.usage_count
            });
        }
    } else if (const auto* quantized8 = dynamic_cast<const QuantizedInt8FIFOMemory*>(&memory)) {
        writeUint32(payload, strategy_quantized_fifo_int8);
        writeUint64(payload, quantized8->memory_capacity);
        writeQuantizationParameters(payload, quantized8->sample_quantizer.inputParameters());
        writeQuantizationParameters(payload, quantized8->sample_quantizer.targetParameters());
        writeUint64(payload, quantized8->input_dimensions);
        writeUint64(payload, quantized8->target_dimensions);
        writeUint64(payload, quantized8->samples.size());
        for (const Int8QuantizedTrainingSample& stored : quantized8->samples) {
            writeInt8Values(payload, stored.input.values);
            writeInt8Values(payload, stored.target.values);
            writeQuantizedMetadata(payload, {
                stored.priority, stored.error, stored.novelty,
                stored.rarity, stored.recency, stored.diversity,
                stored.age, stored.usage_count
            });
        }
    } else {
        throw std::runtime_error("Unsupported learning memory strategy for serialization");
    }

    const std::string bytes = payload.str();
    const std::uint64_t file_checksum = checksum(bytes);
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to open learning memory file for writing");
    writeBytes(stream, bytes.data(), bytes.size());
    writeBytes(stream, &file_checksum, sizeof(file_checksum));
}

std::unique_ptr<LearningMemory> LearningMemorySerialization::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open learning memory file for reading");
    const std::string file_bytes{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    };
    if (file_bytes.size() <= sizeof(std::uint64_t)) {
        throw std::runtime_error("Learning memory file is truncated");
    }

    const size_t payload_size = file_bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, file_bytes.data() + payload_size, sizeof(stored_checksum));
    const std::string payload = file_bytes.substr(0, payload_size);
    if (checksum(payload) != stored_checksum) {
        throw std::runtime_error("Learning memory file checksum mismatch");
    }
    std::istringstream stream(payload, std::ios::binary);

    char file_magic[sizeof(magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) {
        throw std::runtime_error("Invalid learning memory file magic");
    }
    std::uint32_t version = 0;
    readBytes(stream, &version, sizeof(version));
    if (version != format_version) throw std::runtime_error("Unsupported learning memory file version");

    const std::uint32_t strategy = readUint32(stream);
    const std::uint64_t capacity = readUint64(stream);
    if (capacity == 0 || capacity > maximum_capacity) {
        throw std::runtime_error("Learning memory capacity is invalid");
    }

    if (strategy == strategy_fifo) {
        const std::uint64_t sample_count = readUint64(stream);
        if (sample_count > capacity) throw std::runtime_error("Learning memory sample count exceeds capacity");
        auto memory = std::make_unique<FIFOMemory>(static_cast<size_t>(capacity));
        std::vector<TrainingSample> samples;
        samples.reserve(static_cast<size_t>(sample_count));
        for (std::uint64_t index = 0; index < sample_count; ++index) samples.push_back(readSample(stream));
        memory->samples = std::move(samples);
        return memory;
    }

    if (strategy == strategy_reservoir) {
        const std::uint64_t seen_samples = readUint64(stream);
        const std::string rng_state = readString(stream);
        const std::uint64_t sample_count = readUint64(stream);
        if (sample_count > capacity) throw std::runtime_error("Learning memory sample count exceeds capacity");
        auto memory = std::make_unique<ReservoirMemory>(static_cast<size_t>(capacity));
        memory->seen_samples = static_cast<size_t>(seen_samples);
        restoreRngState(memory->generator, rng_state);
        std::vector<TrainingSample> samples;
        samples.reserve(static_cast<size_t>(sample_count));
        for (std::uint64_t index = 0; index < sample_count; ++index) samples.push_back(readSample(stream));
        memory->samples = std::move(samples);
        return memory;
    }

    if (strategy == strategy_prioritized) {
        const double alpha = readDouble(stream);
        const double beta = readDouble(stream);
        const double beta_annealing_rate = readDouble(stream);
        const double exploration_epsilon = readDouble(stream);
        const std::string rng_state = readString(stream);
        const std::uint64_t sample_count = readUint64(stream);
        if (sample_count > capacity) throw std::runtime_error("Learning memory sample count exceeds capacity");
        auto memory = std::make_unique<PrioritizedMemory>(
            static_cast<size_t>(capacity), alpha, 5489u, beta, beta_annealing_rate, exploration_epsilon
        );
        restoreRngState(memory->generator, rng_state);
        std::vector<TrainingSample> samples;
        samples.reserve(static_cast<size_t>(sample_count));
        for (std::uint64_t index = 0; index < sample_count; ++index) samples.push_back(readSample(stream));
        memory->samples = std::move(samples);
        return memory;
    }

    if (strategy == strategy_novelty) {
        const double threshold = readDouble(stream);
        const std::uint64_t sample_count = readUint64(stream);
        if (sample_count > capacity) throw std::runtime_error("Learning memory sample count exceeds capacity");
        auto memory = std::make_unique<NoveltyMemory>(static_cast<size_t>(capacity), threshold);
        std::vector<TrainingSample> samples;
        samples.reserve(static_cast<size_t>(sample_count));
        for (std::uint64_t index = 0; index < sample_count; ++index) {
            TrainingSample sample = readSample(stream);
            if (!samples.empty() && sample.input.size() != samples.front().input.size()) {
                throw std::runtime_error("Learning memory novelty samples have inconsistent input size");
            }
            samples.push_back(std::move(sample));
        }
        memory->samples = std::move(samples);
        return memory;
    }

    if (strategy == strategy_hybrid) {
        HybridMemoryRatios ratios;
        ratios.recent = readDouble(stream);
        ratios.error = readDouble(stream);
        ratios.novelty = readDouble(stream);
        ratios.historical = readDouble(stream);
        const double threshold = readDouble(stream);
        const std::uint64_t seen_samples = readUint64(stream);
        const std::string rng_state = readString(stream);
        const std::uint64_t sample_count = readUint64(stream);
        if (sample_count > capacity) throw std::runtime_error("Learning memory sample count exceeds capacity");

        auto memory = std::make_unique<HybridMemory>(static_cast<size_t>(capacity), ratios, threshold);
        memory->seen_samples = seen_samples;
        restoreRngState(memory->generator, rng_state);

        std::vector<HybridMemory::StoredSample> samples;
        samples.reserve(static_cast<size_t>(sample_count));
        for (std::uint64_t index = 0; index < sample_count; ++index) {
            TrainingSample sample = readSample(stream);
            const std::uint32_t partition_tag = readUint32(stream);
            if (partition_tag > static_cast<std::uint32_t>(HybridMemory::Partition::Historical)) {
                throw std::runtime_error("Learning memory hybrid partition tag is invalid");
            }
            samples.push_back({std::move(sample), static_cast<HybridMemory::Partition>(partition_tag)});
        }
        memory->samples = std::move(samples);
        return memory;
    }

    if (strategy == strategy_quantized_fifo_int16) {
        const QuantizationParameters input_parameters = readQuantizationParameters(stream);
        const QuantizationParameters target_parameters = readQuantizationParameters(stream);
        const std::uint64_t input_dimensions = readUint64(stream);
        const std::uint64_t target_dimensions = readUint64(stream);
        if (input_dimensions > maximum_vector_size || target_dimensions > maximum_vector_size) {
            throw std::runtime_error("Quantized learning memory dimensions are invalid");
        }
        const std::uint64_t sample_count = readUint64(stream);
        if (sample_count > capacity) throw std::runtime_error("Learning memory sample count exceeds capacity");

        auto memory = std::make_unique<QuantizedFIFOMemory>(
            static_cast<size_t>(capacity),
            TrainingSampleQuantizer(input_parameters, target_parameters)
        );
        memory->input_dimensions = static_cast<size_t>(input_dimensions);
        memory->target_dimensions = static_cast<size_t>(target_dimensions);

        std::vector<QuantizedTrainingSample> samples;
        samples.reserve(static_cast<size_t>(sample_count));
        for (std::uint64_t index = 0; index < sample_count; ++index) {
            QuantizedTrainingSample stored;
            stored.input.values = readInt16Values(stream);
            stored.input.parameters = input_parameters;
            stored.target.values = readInt16Values(stream);
            stored.target.parameters = target_parameters;
            if (stored.input.values.size() != input_dimensions ||
                stored.target.values.size() != target_dimensions) {
                throw std::runtime_error("Quantized learning memory sample dimensions do not match");
            }
            const QuantizedSampleMetadata metadata = readQuantizedMetadata(stream);
            stored.priority = metadata.priority;
            stored.error = metadata.error;
            stored.novelty = metadata.novelty;
            stored.rarity = metadata.rarity;
            stored.recency = metadata.recency;
            stored.diversity = metadata.diversity;
            stored.age = metadata.age;
            stored.usage_count = metadata.usage_count;
            samples.push_back(std::move(stored));
        }
        memory->samples = std::move(samples);
        return memory;
    }

    if (strategy == strategy_quantized_fifo_int8) {
        const QuantizationParameters input_parameters = readQuantizationParameters(stream);
        const QuantizationParameters target_parameters = readQuantizationParameters(stream);
        const std::uint64_t input_dimensions = readUint64(stream);
        const std::uint64_t target_dimensions = readUint64(stream);
        if (input_dimensions > maximum_vector_size || target_dimensions > maximum_vector_size) {
            throw std::runtime_error("Quantized learning memory dimensions are invalid");
        }
        const std::uint64_t sample_count = readUint64(stream);
        if (sample_count > capacity) throw std::runtime_error("Learning memory sample count exceeds capacity");

        auto memory = std::make_unique<QuantizedInt8FIFOMemory>(
            static_cast<size_t>(capacity),
            Int8TrainingSampleQuantizer(input_parameters, target_parameters)
        );
        memory->input_dimensions = static_cast<size_t>(input_dimensions);
        memory->target_dimensions = static_cast<size_t>(target_dimensions);

        std::vector<Int8QuantizedTrainingSample> samples;
        samples.reserve(static_cast<size_t>(sample_count));
        for (std::uint64_t index = 0; index < sample_count; ++index) {
            Int8QuantizedTrainingSample stored;
            stored.input.values = readInt8Values(stream);
            stored.input.parameters = input_parameters;
            stored.target.values = readInt8Values(stream);
            stored.target.parameters = target_parameters;
            if (stored.input.values.size() != input_dimensions ||
                stored.target.values.size() != target_dimensions) {
                throw std::runtime_error("Quantized learning memory sample dimensions do not match");
            }
            const QuantizedSampleMetadata metadata = readQuantizedMetadata(stream);
            stored.priority = metadata.priority;
            stored.error = metadata.error;
            stored.novelty = metadata.novelty;
            stored.rarity = metadata.rarity;
            stored.recency = metadata.recency;
            stored.diversity = metadata.diversity;
            stored.age = metadata.age;
            stored.usage_count = metadata.usage_count;
            samples.push_back(std::move(stored));
        }
        memory->samples = std::move(samples);
        return memory;
    }

    throw std::runtime_error("Unknown learning memory strategy in file");
}
