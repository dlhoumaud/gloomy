#include "headers/GloomyModelSerialization.h"
#include "headers/LearningMemorySerialization.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <set>
#include <unordered_map>
#include <utility>

namespace {
constexpr char magic[] = "GLOOMY_MODEL";
constexpr char inference_magic[] = "GLOOMY_INFER";
constexpr std::uint32_t format_version = 1;
constexpr std::uint32_t network_section = 1;
constexpr std::uint32_t normalizer_section = 2;
constexpr std::uint32_t optimizer_section = 3;
constexpr std::uint32_t memory_section = 4;
constexpr std::uint32_t maximum_sections = 32;
constexpr std::uint64_t maximum_section_size = 256ULL * 1024ULL * 1024ULL;
constexpr std::uint64_t hash_offset = 1469598103934665603ULL;
constexpr std::uint64_t hash_prime = 1099511628211ULL;
constexpr std::uint64_t maximum_layers = 1024;
constexpr std::uint64_t maximum_dimension = 1'000'000;

std::uint64_t checksum(const std::string& bytes) {
    std::uint64_t value = hash_offset;
    for (unsigned char byte : bytes) { value ^= byte; value *= hash_prime; }
    return value;
}
void write(std::ostream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write GLOOMY_MODEL file");
}
void read(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated GLOOMY_MODEL file");
}
void writeString(std::ostream& stream, const std::string& value) {
    const std::uint64_t size = value.size();
    write(stream, &size, sizeof(size));
    if (!value.empty()) write(stream, value.data(), value.size());
}
std::string readString(std::istream& stream, std::uint64_t maximum) {
    std::uint64_t size = 0; read(stream, &size, sizeof(size));
    if (size > maximum || size > std::numeric_limits<size_t>::max())
        throw std::runtime_error("GLOOMY_MODEL string size is invalid");
    std::string value(static_cast<size_t>(size), '\0');
    if (!value.empty()) read(stream, value.data(), value.size());
    return value;
}
void writeSection(std::ostream& output, std::uint32_t id, const std::string& bytes) {
    if (bytes.size() > maximum_section_size) throw std::runtime_error("GLOOMY_MODEL section is too large");
    const std::uint64_t size = bytes.size();
    write(output, &id, sizeof(id)); write(output, &size, sizeof(size));
    if (!bytes.empty()) write(output, bytes.data(), bytes.size());
}

std::string serializeNetwork(const NeuralNetwork& network) {
    if (network.layers().empty() || network.layers().size() > maximum_layers)
        throw std::invalid_argument("Network layer count is invalid");
    std::ostringstream out(std::ios::binary);
    writeString(out, network.algorithm); writeString(out, network.post_algorithm);
    const std::uint64_t count = network.layers().size(); write(out, &count, sizeof(count));
    for (const DenseLayer& layer : network.layers()) {
        const std::uint64_t inputs = layer.weights().size(), outputs = layer.bias().size();
        if (!inputs || !outputs || inputs > maximum_dimension || outputs > maximum_dimension)
            throw std::invalid_argument("Network layer dimensions are invalid");
        write(out, &inputs, sizeof(inputs)); write(out, &outputs, sizeof(outputs));
        const auto& weights = layer.weightValues();
        if (weights.size() != inputs * outputs) throw std::invalid_argument("Network weight shape is invalid");
        for (double value : weights) if (!std::isfinite(value)) throw std::invalid_argument("Network parameters must be finite");
        write(out, weights.data(), weights.size() * sizeof(double));
        for (double value : layer.bias()) if (!std::isfinite(value)) throw std::invalid_argument("Network parameters must be finite");
        write(out, layer.bias().data(), layer.bias().size() * sizeof(double));
    }
    return out.str();
}

NeuralNetwork deserializeNetwork(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::binary);
    NeuralNetwork result;
    result.algorithm = readString(in, maximum_dimension);
    result.post_algorithm = readString(in, maximum_dimension);
    std::uint64_t count = 0; read(in, &count, sizeof(count));
    if (!count || count > maximum_layers) throw std::runtime_error("GLOOMY_MODEL network layer count is invalid");
    std::uint64_t previous_outputs = 0;
    for (std::uint64_t index = 0; index < count; ++index) {
        std::uint64_t inputs = 0, outputs = 0;
        read(in, &inputs, sizeof(inputs)); read(in, &outputs, sizeof(outputs));
        if (!inputs || !outputs || inputs > maximum_dimension || outputs > maximum_dimension ||
            (index && inputs != previous_outputs))
            throw std::runtime_error("GLOOMY_MODEL network dimensions are invalid");
        previous_outputs = outputs;
        result.addLayer(static_cast<int>(inputs), static_cast<int>(outputs));
        DenseLayer& layer = result.layers().back();
        auto& weights = layer.weightValues();
        read(in, weights.data(), weights.size() * sizeof(double));
        for (double value : weights) if (!std::isfinite(value)) throw std::runtime_error("GLOOMY_MODEL parameters must be finite");
        read(in, layer.bias().data(), layer.bias().size() * sizeof(double));
        for (double value : layer.bias()) if (!std::isfinite(value)) throw std::runtime_error("GLOOMY_MODEL parameters must be finite");
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected network section data");
    return result;
}

std::string serializeNormalizer(const StreamingNormalizer& normalizer) {
    if (normalizer.count() == 0) throw std::invalid_argument("Normalizer has no statistics to save");
    std::ostringstream out(std::ios::binary);
    const std::uint64_t dimensions = normalizer.dimensions(), count = normalizer.count();
    write(out, &dimensions, sizeof(dimensions)); write(out, &count, sizeof(count));
    const auto means = normalizer.mean(); const auto variances = normalizer.variance();
    for (const auto* values : {&means, &variances, &normalizer.minimum(), &normalizer.maximum()})
        write(out, values->data(), values->size() * sizeof(double));
    return out.str();
}
StreamingNormalizer deserializeNormalizer(const std::string& bytes) {
    std::istringstream in(bytes, std::ios::binary);
    std::uint64_t dimensions = 0, count = 0; read(in, &dimensions, sizeof(dimensions)); read(in, &count, sizeof(count));
    if (!dimensions || dimensions > maximum_dimension || !count || count > std::numeric_limits<size_t>::max())
        throw std::runtime_error("GLOOMY_MODEL normalizer dimensions or count are invalid");
    std::vector<double> means(dimensions), variances(dimensions), minimum(dimensions), maximum(dimensions);
    for (auto* values : {&means, &variances, &minimum, &maximum})
        read(in, values->data(), values->size() * sizeof(double));
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected normalizer section data");
    StreamingNormalizer result(static_cast<size_t>(dimensions));
    result.restore(static_cast<size_t>(count), means, variances, minimum, maximum);
    return result;
}

std::string serializeOptimizer(const Optimizer& optimizer) {
    std::ostringstream out(std::ios::binary);
    writeString(out, optimizer.checkpointType());
    optimizer.writeCheckpoint(out);
    return out.str();
}
void deserializeOptimizer(const std::string& bytes, Optimizer& optimizer, const std::vector<DenseLayer>& layers) {
    std::istringstream in(bytes, std::ios::binary);
    const std::string type = readString(in, 64);
    if (type != optimizer.checkpointType()) throw std::runtime_error("GLOOMY_MODEL optimizer type mismatch");
    optimizer.readCheckpoint(in, layers);
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected optimizer section data");
}
}

bool GloomyModelSerialization::isInferenceArtifact(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open model file for reading");
    char header[sizeof(inference_magic) - 1]{};
    file.read(header, sizeof(header));
    return file.gcount() == static_cast<std::streamsize>(sizeof(header)) &&
        std::string(header, sizeof(header)) == std::string(inference_magic, sizeof(inference_magic) - 1);
}

void GloomyModelSerialization::saveInference(
    const std::string& path,
    const NeuralNetwork& network,
    const StreamingNormalizer& normalizer
) {
    if (network.layers().empty() ||
        normalizer.dimensions() != network.layers().front().weights().size())
        throw std::invalid_argument("Inference artifact normalizer dimensions do not match network input");
    std::ostringstream payload(std::ios::binary);
    write(payload, inference_magic, sizeof(inference_magic) - 1);
    write(payload, &format_version, sizeof(format_version));
    const std::uint32_t section_count = 2;
    write(payload, &section_count, sizeof(section_count));
    writeSection(payload, network_section, serializeNetwork(network));
    writeSection(payload, normalizer_section, serializeNormalizer(normalizer));
    const std::string bytes = payload.str();
    const std::uint64_t hash = checksum(bytes);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Unable to open inference artifact for writing");
    write(file, bytes.data(), bytes.size());
    write(file, &hash, sizeof(hash));
}

void GloomyModelSerialization::loadInference(
    const std::string& path,
    NeuralNetwork& network,
    StreamingNormalizer& normalizer
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open inference artifact for reading");
    const std::string file_bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    constexpr size_t minimum_size = sizeof(inference_magic) - 1 + sizeof(format_version) +
        sizeof(std::uint32_t) + sizeof(std::uint64_t);
    if (file_bytes.size() < minimum_size) throw std::runtime_error("Inference artifact is truncated");
    const size_t payload_size = file_bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_hash = 0;
    std::memcpy(&stored_hash, file_bytes.data() + payload_size, sizeof(stored_hash));
    const std::string payload = file_bytes.substr(0, payload_size);
    if (checksum(payload) != stored_hash) throw std::runtime_error("Inference artifact checksum mismatch");
    std::istringstream in(payload, std::ios::binary);
    char file_magic[sizeof(inference_magic) - 1];
    read(in, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(inference_magic, sizeof(inference_magic) - 1))
        throw std::runtime_error("Invalid inference artifact magic");
    std::uint32_t version = 0, count = 0;
    read(in, &version, sizeof(version));
    read(in, &count, sizeof(count));
    if (version != format_version) throw std::runtime_error("Unsupported inference artifact version");
    if (count != 2) throw std::runtime_error("Inference artifact section count is invalid");
    std::unordered_map<std::uint32_t, std::string> sections;
    for (std::uint32_t index = 0; index < count; ++index) {
        std::uint32_t id = 0;
        std::uint64_t size = 0;
        read(in, &id, sizeof(id));
        read(in, &size, sizeof(size));
        if (size > maximum_section_size || size > std::numeric_limits<size_t>::max())
            throw std::runtime_error("Inference artifact section size is invalid");
        if (id != network_section && id != normalizer_section)
            throw std::runtime_error("Unexpected inference artifact section");
        if (sections.count(id)) throw std::runtime_error("Duplicate inference artifact section");
        std::string bytes(static_cast<size_t>(size), '\0');
        if (size) read(in, bytes.data(), bytes.size());
        sections.emplace(id, std::move(bytes));
    }
    if (in.peek() != std::char_traits<char>::eof() || !sections.count(network_section) ||
        !sections.count(normalizer_section))
        throw std::runtime_error("Inference artifact sections are invalid");
    NeuralNetwork restored_network = deserializeNetwork(sections.at(network_section));
    StreamingNormalizer restored_normalizer = deserializeNormalizer(sections.at(normalizer_section));
    if (restored_normalizer.dimensions() != restored_network.layers().front().weights().size())
        throw std::runtime_error("Inference artifact normalizer dimensions do not match network input");
    network = std::move(restored_network);
    normalizer = std::move(restored_normalizer);
}

void GloomyModelSerialization::save(
    const std::string& path,
    const NeuralNetwork& network,
    const StreamingNormalizer& normalizer,
    const Optimizer& optimizer,
    const LearningMemory* memory
) {
    std::ostringstream payload(std::ios::binary);
    write(payload, magic, sizeof(magic) - 1); write(payload, &format_version, sizeof(format_version));
    const std::uint32_t section_count = memory ? 4 : 3; write(payload, &section_count, sizeof(section_count));
    writeSection(payload, network_section, serializeNetwork(network));
    writeSection(payload, normalizer_section, serializeNormalizer(normalizer));
    writeSection(payload, optimizer_section, serializeOptimizer(optimizer));
    if (memory) writeSection(payload, memory_section, LearningMemorySerialization::serialize(*memory));
    const std::string bytes = payload.str(); const std::uint64_t hash = checksum(bytes);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Unable to open GLOOMY_MODEL file for writing");
    write(file, bytes.data(), bytes.size()); write(file, &hash, sizeof(hash));
}

void GloomyModelSerialization::load(
    const std::string& path,
    NeuralNetwork& network,
    StreamingNormalizer& normalizer,
    Optimizer& optimizer,
    std::unique_ptr<LearningMemory>* memory
) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open GLOOMY_MODEL file for reading");
    const std::string file_bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (file_bytes.size() < sizeof(magic) - 1 + sizeof(format_version) + sizeof(std::uint32_t) + sizeof(std::uint64_t))
        throw std::runtime_error("GLOOMY_MODEL file is truncated");
    const size_t payload_size = file_bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_hash = 0; std::memcpy(&stored_hash, file_bytes.data() + payload_size, sizeof(stored_hash));
    const std::string payload = file_bytes.substr(0, payload_size);
    if (checksum(payload) != stored_hash) throw std::runtime_error("GLOOMY_MODEL checksum mismatch");
    std::istringstream in(payload, std::ios::binary);
    char file_magic[sizeof(magic) - 1]; read(in, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1))
        throw std::runtime_error("Invalid GLOOMY_MODEL magic");
    std::uint32_t version = 0, count = 0; read(in, &version, sizeof(version)); read(in, &count, sizeof(count));
    if (version != format_version) throw std::runtime_error("Unsupported GLOOMY_MODEL version");
    if (!count || count > maximum_sections) throw std::runtime_error("GLOOMY_MODEL section count is invalid");
    std::unordered_map<std::uint32_t, std::string> sections;
    std::set<std::uint32_t> section_ids;
    for (std::uint32_t index = 0; index < count; ++index) {
        std::uint32_t id = 0; std::uint64_t size = 0; read(in, &id, sizeof(id)); read(in, &size, sizeof(size));
        if (size > maximum_section_size || size > std::numeric_limits<size_t>::max())
            throw std::runtime_error("GLOOMY_MODEL section size is invalid");
        if (!section_ids.insert(id).second) throw std::runtime_error("Duplicate GLOOMY_MODEL section");
        const bool supported_section = id >= network_section && id <= optimizer_section;
        const bool requested_memory = id == memory_section && memory != nullptr;
        if (supported_section || requested_memory) {
            std::string bytes(static_cast<size_t>(size), '\0');
            if (size) read(in, bytes.data(), bytes.size());
            sections.emplace(id, std::move(bytes));
        } else {
            in.ignore(static_cast<std::streamsize>(size));
            if (static_cast<std::uint64_t>(in.gcount()) != size)
                throw std::runtime_error("Invalid or truncated GLOOMY_MODEL section");
        }
    }
    if (in.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected GLOOMY_MODEL data");
    if (!sections.count(network_section) || !sections.count(normalizer_section) || !sections.count(optimizer_section))
        throw std::runtime_error("GLOOMY_MODEL required section is missing");

    NeuralNetwork restored_network = deserializeNetwork(sections.at(network_section));
    StreamingNormalizer restored_normalizer = deserializeNormalizer(sections.at(normalizer_section));
    if (restored_normalizer.dimensions() != restored_network.layers().front().weights().size())
        throw std::runtime_error("GLOOMY_MODEL normalizer dimensions do not match network input");
    std::istringstream optimizer_validation(sections.at(optimizer_section), std::ios::binary);
    if (readString(optimizer_validation, 64) != optimizer.checkpointType())
        throw std::runtime_error("GLOOMY_MODEL optimizer type mismatch");
    std::unique_ptr<LearningMemory> restored_memory;
    const auto memory_section_iterator = sections.find(memory_section);
    if (memory_section_iterator != sections.end()) {
        restored_memory = LearningMemorySerialization::deserialize(memory_section_iterator->second);
        LearningMemorySerialization::validateDimensions(
            *restored_memory,
            restored_network.layers().front().weights().size(),
            restored_network.layers().back().bias().size()
        );
    }

    deserializeOptimizer(sections.at(optimizer_section), optimizer, restored_network.layers());
    network = std::move(restored_network);
    normalizer = std::move(restored_normalizer);
    if (memory) *memory = std::move(restored_memory);
}
