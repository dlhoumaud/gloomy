#include "headers/LearningMemorySerialization.h"
#include "headers/FIFOMemory.h"
#include "headers/ReservoirMemory.h"
#include "headers/PrioritizedMemory.h"
#include "headers/NoveltyMemory.h"
#include "headers/HybridMemory.h"
#include "headers/QuantizedFIFOMemory.h"
#include "headers/QuantizedInt8FIFOMemory.h"
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>

namespace {
constexpr char magic[] = "GLOOMYMEM";
constexpr std::uint32_t version = 1;
constexpr std::uint64_t maximum_capacity = 10'000'000;
constexpr std::uint64_t maximum_vector = 1'000'000;
enum MemoryType : std::uint32_t { Fifo = 1, Reservoir, Prioritized, Novelty, Hybrid, Quantized16, Quantized8 };

void write(std::ostream& out, const void* data, size_t size) {
    out.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!out) throw std::runtime_error("Unable to serialize learning memory");
}
void read(std::istream& in, void* data, size_t size) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!in) throw std::runtime_error("Invalid or truncated learning memory section");
}
void writeSize(std::ostream& out, size_t value) {
    const std::uint64_t saved = value; write(out, &saved, sizeof(saved));
}
size_t readSize(std::istream& in, std::uint64_t limit = maximum_capacity) {
    std::uint64_t value = 0; read(in, &value, sizeof(value));
    if (value > limit || value > std::numeric_limits<size_t>::max())
        throw std::runtime_error("Learning memory size is invalid");
    return static_cast<size_t>(value);
}
void writeVector(std::ostream& out, const std::vector<double>& values) {
    writeSize(out, values.size());
    if (!values.empty()) write(out, values.data(), values.size() * sizeof(double));
}
std::vector<double> readVector(std::istream& in) {
    const size_t size = readSize(in, maximum_vector);
    std::vector<double> values(size);
    if (size) read(in, values.data(), size * sizeof(double));
    for (double value : values) if (!std::isfinite(value)) throw std::runtime_error("Memory sample values must be finite");
    return values;
}
void writeSample(std::ostream& out, const TrainingSample& sample) {
    writeVector(out, sample.input); writeVector(out, sample.target);
    const double metadata[] = {sample.priority, sample.error, sample.novelty, sample.rarity, sample.recency, sample.diversity};
    for (double value : metadata) {
        if (!std::isfinite(value)) throw std::invalid_argument("Memory metadata must be finite");
        write(out, &value, sizeof(value));
    }
    writeSize(out, sample.age); writeSize(out, sample.usage_count);
}
TrainingSample readSample(std::istream& in) {
    TrainingSample sample; sample.input = readVector(in); sample.target = readVector(in);
    double* metadata[] = {&sample.priority, &sample.error, &sample.novelty, &sample.rarity, &sample.recency, &sample.diversity};
    for (double* value : metadata) {
        read(in, value, sizeof(*value));
        if (!std::isfinite(*value)) throw std::runtime_error("Memory metadata must be finite");
    }
    sample.age = readSize(in, std::numeric_limits<std::uint64_t>::max());
    sample.usage_count = readSize(in, std::numeric_limits<std::uint64_t>::max());
    return sample;
}
std::string rngState(const std::mt19937& rng) {
    std::ostringstream out; out << rng;
    if (!out) throw std::runtime_error("Unable to serialize memory random state");
    return out.str();
}
std::mt19937 readRng(std::istream& in) {
    const size_t size = readSize(in, 10000);
    std::string text(size, '\0'); if (size) read(in, text.data(), size);
    std::istringstream state(text);
    std::mt19937 rng; state >> rng;
    if (!state) throw std::runtime_error("Invalid memory random state");
    state >> std::ws;
    if (!state.eof()) throw std::runtime_error("Unexpected memory random state data");
    return rng;
}
void writeRng(std::ostream& out, const std::mt19937& rng) {
    const std::string state = rngState(rng); writeSize(out, state.size());
    if (!state.empty()) write(out, state.data(), state.size());
}
void writeQuantParameters(std::ostream& out, const QuantizationParameters& p) {
    if (!std::isfinite(p.scale) || p.scale <= 0.0) throw std::invalid_argument("Invalid memory quantization scale");
    write(out, &p.scale, sizeof(p.scale)); write(out, &p.zero_point, sizeof(p.zero_point));
}
QuantizationParameters readQuantParameters(std::istream& in) {
    QuantizationParameters p{}; read(in, &p.scale, sizeof(p.scale)); read(in, &p.zero_point, sizeof(p.zero_point));
    if (!std::isfinite(p.scale) || p.scale <= 0.0) throw std::runtime_error("Invalid memory quantization scale");
    return p;
}
void writeQuantizedVector(std::ostream& out, const std::vector<std::int16_t>& values) {
    writeSize(out, values.size()); if (!values.empty()) write(out, values.data(), values.size() * sizeof(values[0]));
}
std::vector<std::int16_t> readQuantized16(std::istream& in, size_t expected) {
    const size_t size = readSize(in, maximum_vector);
    if (size != expected) throw std::runtime_error("Quantized memory dimensions mismatch");
    std::vector<std::int16_t> values(size); if (size) read(in, values.data(), size * sizeof(values[0])); return values;
}
void writeQuantizedVector(std::ostream& out, const std::vector<std::int8_t>& values) {
    writeSize(out, values.size()); if (!values.empty()) write(out, values.data(), values.size() * sizeof(values[0]));
}
std::vector<std::int8_t> readQuantized8(std::istream& in, size_t expected) {
    const size_t size = readSize(in, maximum_vector);
    if (size != expected) throw std::runtime_error("Quantized memory dimensions mismatch");
    std::vector<std::int8_t> values(size); if (size) read(in, values.data(), size * sizeof(values[0])); return values;
}
void writeQuantizedMetadata(std::ostream& out, const QuantizedTrainingSample& s) {
    const double values[] = {s.priority, s.error, s.novelty, s.rarity, s.recency, s.diversity};
    for (double value : values) { if (!std::isfinite(value)) throw std::invalid_argument("Memory metadata must be finite"); write(out, &value, sizeof(value)); }
    writeSize(out, s.age); writeSize(out, s.usage_count);
}
void writeQuantizedMetadata(std::ostream& out, const Int8QuantizedTrainingSample& s) {
    const double values[] = {s.priority, s.error, s.novelty, s.rarity, s.recency, s.diversity};
    for (double value : values) { if (!std::isfinite(value)) throw std::invalid_argument("Memory metadata must be finite"); write(out, &value, sizeof(value)); }
    writeSize(out, s.age); writeSize(out, s.usage_count);
}
void readQuantizedMetadata(std::istream& in, QuantizedTrainingSample& s) {
    double* values[] = {&s.priority, &s.error, &s.novelty, &s.rarity, &s.recency, &s.diversity};
    for (double* value : values) { read(in, value, sizeof(*value)); if (!std::isfinite(*value)) throw std::runtime_error("Memory metadata must be finite"); }
    s.age = readSize(in, std::numeric_limits<std::uint64_t>::max());
    s.usage_count = readSize(in, std::numeric_limits<std::uint64_t>::max());
}
void readQuantizedMetadata(std::istream& in, Int8QuantizedTrainingSample& s) {
    double* values[] = {&s.priority, &s.error, &s.novelty, &s.rarity, &s.recency, &s.diversity};
    for (double* value : values) { read(in, value, sizeof(*value)); if (!std::isfinite(*value)) throw std::runtime_error("Memory metadata must be finite"); }
    s.age = readSize(in, std::numeric_limits<std::uint64_t>::max());
    s.usage_count = readSize(in, std::numeric_limits<std::uint64_t>::max());
}
}

std::string LearningMemorySerialization::serialize(const LearningMemory& memory) {
    std::ostringstream out(std::ios::binary);
    write(out, magic, sizeof(magic) - 1); write(out, &version, sizeof(version));
    const std::uint64_t capacity = memory.capacity(), count = memory.size();
    if (capacity == 0 || capacity > maximum_capacity || count > capacity) throw std::invalid_argument("Learning memory capacity is invalid");
    write(out, &capacity, sizeof(capacity)); write(out, &count, sizeof(count));

    MemoryType type = Fifo;
    if (dynamic_cast<const FIFOMemory*>(&memory)) type = Fifo;
    else if (dynamic_cast<const ReservoirMemory*>(&memory)) type = Reservoir;
    else if (dynamic_cast<const PrioritizedMemory*>(&memory)) type = Prioritized;
    else if (dynamic_cast<const NoveltyMemory*>(&memory)) type = Novelty;
    else if (dynamic_cast<const HybridMemory*>(&memory)) type = Hybrid;
    else if (dynamic_cast<const QuantizedFIFOMemory*>(&memory)) type = Quantized16;
    else if (dynamic_cast<const QuantizedInt8FIFOMemory*>(&memory)) type = Quantized8;
    else throw std::invalid_argument("Unsupported learning memory implementation");
    write(out, &type, sizeof(type));

    if (const auto* m = dynamic_cast<const ReservoirMemory*>(&memory)) {
        write(out, &m->initial_seed, sizeof(m->initial_seed));
        writeSize(out, m->seen_samples); writeRng(out, m->generator);
        for (const auto& sample : m->samples) writeSample(out, sample);
    } else if (const auto* m = dynamic_cast<const PrioritizedMemory*>(&memory)) {
        write(out, &m->priority_exponent, sizeof(m->priority_exponent));
        write(out, &m->initial_seed, sizeof(m->initial_seed)); writeRng(out, m->generator);
        for (const auto& sample : m->samples) writeSample(out, sample);
    } else if (const auto* m = dynamic_cast<const NoveltyMemory*>(&memory)) {
        write(out, &m->distance_threshold, sizeof(m->distance_threshold));
        for (const auto& sample : m->samples) writeSample(out, sample);
    } else if (const auto* m = dynamic_cast<const HybridMemory*>(&memory)) {
        write(out, &m->memory_ratios.recent, sizeof(double));
        write(out, &m->memory_ratios.error, sizeof(double));
        write(out, &m->memory_ratios.novelty, sizeof(double));
        write(out, &m->memory_ratios.historical, sizeof(double));
        writeSize(out, m->recent_capacity); writeSize(out, m->error_capacity);
        writeSize(out, m->novelty_capacity); writeSize(out, m->historical_capacity);
        write(out, &m->distance_threshold, sizeof(m->distance_threshold));
        write(out, &m->seen_samples, sizeof(m->seen_samples));
        write(out, &m->initial_seed, sizeof(m->initial_seed)); writeRng(out, m->generator);
        for (const auto& stored : m->samples) {
            const std::uint32_t partition = static_cast<std::uint32_t>(stored.partition);
            write(out, &partition, sizeof(partition)); writeSample(out, stored.sample);
        }
    } else if (const auto* m = dynamic_cast<const QuantizedFIFOMemory*>(&memory)) {
        writeSize(out, m->input_dimensions); writeSize(out, m->target_dimensions);
        writeQuantParameters(out, m->sample_quantizer.inputParameters());
        writeQuantParameters(out, m->sample_quantizer.targetParameters());
        for (const auto& sample : m->samples) {
            writeQuantizedVector(out, sample.input.values); writeQuantizedVector(out, sample.target.values);
            writeQuantizedMetadata(out, sample);
        }
    } else if (const auto* m = dynamic_cast<const QuantizedInt8FIFOMemory*>(&memory)) {
        writeSize(out, m->input_dimensions); writeSize(out, m->target_dimensions);
        writeQuantParameters(out, m->sample_quantizer.inputParameters());
        writeQuantParameters(out, m->sample_quantizer.targetParameters());
        for (const auto& sample : m->samples) {
            writeQuantizedVector(out, sample.input.values); writeQuantizedVector(out, sample.target.values);
            writeQuantizedMetadata(out, sample);
        }
    } else {
        const auto* fifo = dynamic_cast<const FIFOMemory*>(&memory);
        for (const auto& sample : fifo->samples) writeSample(out, sample);
    }
    return out.str();
}

std::unique_ptr<LearningMemory> LearningMemorySerialization::deserialize(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::binary);
    char file_magic[sizeof(magic) - 1]; read(in, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) throw std::runtime_error("Invalid learning memory magic");
    std::uint32_t file_version = 0; read(in, &file_version, sizeof(file_version));
    if (file_version != version) throw std::runtime_error("Unsupported learning memory version");
    const size_t capacity = readSize(in); const size_t count = readSize(in);
    if (!capacity || count > capacity) throw std::runtime_error("Learning memory count exceeds capacity");
    MemoryType type{}; read(in, &type, sizeof(type));
    std::unique_ptr<LearningMemory> result;
    if (type == Fifo) result = std::make_unique<FIFOMemory>(capacity);
    else if (type == Reservoir) result = std::make_unique<ReservoirMemory>(capacity);
    else if (type == Prioritized) result = nullptr;
    else if (type == Novelty) result = nullptr;
    else if (type == Hybrid) result = nullptr;
    else if (type == Quantized16 || type == Quantized8) result = nullptr;
    else throw std::runtime_error("Unknown learning memory type");

    if (type == Fifo) {
        auto& m = static_cast<FIFOMemory&>(*result); m.samples.reserve(capacity);
        for (size_t i = 0; i < count; ++i) m.samples.push_back(readSample(in));
    } else if (type == Reservoir) {
        std::uint32_t seed = 0; read(in, &seed, sizeof(seed));
        result = std::make_unique<ReservoirMemory>(capacity, seed);
        auto& m = static_cast<ReservoirMemory&>(*result); m.initial_seed = seed; m.seen_samples = readSize(in, std::numeric_limits<size_t>::max()); m.generator = readRng(in); m.samples.reserve(capacity);
        if (m.seen_samples < count) throw std::runtime_error("Reservoir observation count is invalid");
        for (size_t i = 0; i < count; ++i) m.samples.push_back(readSample(in));
    } else if (type == Prioritized) {
        double alpha = 0.0; std::uint32_t seed = 0;
        read(in, &alpha, sizeof(alpha)); read(in, &seed, sizeof(seed));
        result = std::make_unique<PrioritizedMemory>(capacity, alpha, seed);
        auto& m = static_cast<PrioritizedMemory&>(*result); m.initial_seed = seed; m.generator = readRng(in); m.samples.reserve(capacity);
        for (size_t i = 0; i < count; ++i) m.samples.push_back(readSample(in));
    } else if (type == Novelty) {
        double threshold = 0.0; read(in, &threshold, sizeof(threshold));
        result = std::make_unique<NoveltyMemory>(capacity, threshold);
        auto& m = static_cast<NoveltyMemory&>(*result); m.samples.reserve(capacity);
        for (size_t i = 0; i < count; ++i) m.samples.push_back(readSample(in));
        for (const auto& sample : m.samples) {
            if (sample.input.empty() || sample.input.size() != m.samples.front().input.size())
                throw std::runtime_error("Novelty memory input dimensions are inconsistent");
        }
    } else if (type == Hybrid) {
        HybridMemoryRatios ratios{}; double threshold = 0.0; std::uint64_t seen = 0;
        read(in, &ratios.recent, sizeof(double)); read(in, &ratios.error, sizeof(double));
        read(in, &ratios.novelty, sizeof(double)); read(in, &ratios.historical, sizeof(double));
        const size_t recent_capacity = readSize(in), error_capacity = readSize(in);
        const size_t novelty_capacity = readSize(in), historical_capacity = readSize(in);
        read(in, &threshold, sizeof(threshold)); read(in, &seen, sizeof(seen));
        std::uint32_t seed = 0; read(in, &seed, sizeof(seed));
        result = std::make_unique<HybridMemory>(capacity, ratios, threshold, seed);
        auto& m = static_cast<HybridMemory&>(*result);
        if (recent_capacity + error_capacity + novelty_capacity + historical_capacity != capacity)
            throw std::runtime_error("Hybrid memory partition capacities are invalid");
        m.recent_capacity = recent_capacity; m.error_capacity = error_capacity;
        m.novelty_capacity = novelty_capacity; m.historical_capacity = historical_capacity;
        m.seen_samples = seen; m.initial_seed = seed; m.generator = readRng(in); m.samples.reserve(capacity);
        for (size_t i = 0; i < count; ++i) {
            std::uint32_t partition = 0; read(in, &partition, sizeof(partition));
            if (partition > static_cast<std::uint32_t>(HybridMemory::Partition::Historical)) throw std::runtime_error("Invalid hybrid memory partition");
            m.samples.push_back({readSample(in), static_cast<HybridMemory::Partition>(partition)});
        }
        const auto sizes = m.partitionSizes();
        const auto limits = {m.recent_capacity, m.error_capacity, m.novelty_capacity, m.historical_capacity};
        size_t index = 0; for (size_t limit : limits) if (sizes[index++] > limit) throw std::runtime_error("Hybrid memory partition exceeds capacity");
    } else if (type == Quantized16) {
        const size_t inputs = readSize(in, maximum_vector), targets = readSize(in, maximum_vector);
        if (count && (!inputs || !targets)) throw std::runtime_error("Quantized memory dimensions are invalid");
        auto ip = readQuantParameters(in), tp = readQuantParameters(in);
        result = std::make_unique<QuantizedFIFOMemory>(capacity, TrainingSampleQuantizer(ip, tp));
        auto& m = static_cast<QuantizedFIFOMemory&>(*result); m.input_dimensions = inputs; m.target_dimensions = targets; m.samples.reserve(capacity);
        for (size_t i = 0; i < count; ++i) {
            QuantizedTrainingSample s{}; s.input.parameters = ip; s.target.parameters = tp;
            s.input.values = readQuantized16(in, inputs); s.target.values = readQuantized16(in, targets); readQuantizedMetadata(in, s); m.samples.push_back(std::move(s));
        }
    } else {
        const size_t inputs = readSize(in, maximum_vector), targets = readSize(in, maximum_vector);
        if (count && (!inputs || !targets)) throw std::runtime_error("Quantized memory dimensions are invalid");
        auto ip = readQuantParameters(in), tp = readQuantParameters(in);
        result = std::make_unique<QuantizedInt8FIFOMemory>(capacity, Int8TrainingSampleQuantizer(ip, tp));
        auto& m = static_cast<QuantizedInt8FIFOMemory&>(*result); m.input_dimensions = inputs; m.target_dimensions = targets; m.samples.reserve(capacity);
        for (size_t i = 0; i < count; ++i) {
            Int8QuantizedTrainingSample s{}; s.input.parameters = ip; s.target.parameters = tp;
            s.input.values = readQuantized8(in, inputs); s.target.values = readQuantized8(in, targets); readQuantizedMetadata(in, s); m.samples.push_back(std::move(s));
        }
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected learning memory data");
    return result;
}

void LearningMemorySerialization::validateDimensions(
    const LearningMemory& memory,
    size_t input_size,
    size_t target_size
) {
    auto validate = [input_size, target_size](const TrainingSample& sample) {
        if (sample.input.size() != input_size || sample.target.size() != target_size)
            throw std::runtime_error("Learning memory sample dimensions do not match model");
    };
    if (const auto* m = dynamic_cast<const FIFOMemory*>(&memory)) {
        for (const auto& sample : m->samples) validate(sample);
    } else if (const auto* m = dynamic_cast<const ReservoirMemory*>(&memory)) {
        for (const auto& sample : m->samples) validate(sample);
    } else if (const auto* m = dynamic_cast<const PrioritizedMemory*>(&memory)) {
        for (const auto& sample : m->samples) validate(sample);
    } else if (const auto* m = dynamic_cast<const NoveltyMemory*>(&memory)) {
        for (const auto& sample : m->samples) validate(sample);
    } else if (const auto* m = dynamic_cast<const HybridMemory*>(&memory)) {
        for (const auto& stored : m->samples) validate(stored.sample);
    } else if (const auto* m = dynamic_cast<const QuantizedFIFOMemory*>(&memory)) {
        if ((m->input_dimensions && m->input_dimensions != input_size) ||
            (m->target_dimensions && m->target_dimensions != target_size) ||
            (memory.size() && (!m->input_dimensions || !m->target_dimensions)))
            throw std::runtime_error("Quantized memory dimensions do not match model");
    } else if (const auto* m = dynamic_cast<const QuantizedInt8FIFOMemory*>(&memory)) {
        if ((m->input_dimensions && m->input_dimensions != input_size) ||
            (m->target_dimensions && m->target_dimensions != target_size) ||
            (memory.size() && (!m->input_dimensions || !m->target_dimensions)))
            throw std::runtime_error("Quantized memory dimensions do not match model");
    } else {
        throw std::runtime_error("Unsupported learning memory implementation");
    }
}
