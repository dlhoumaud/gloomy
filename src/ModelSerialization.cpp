#include "headers/ModelSerialization.h"
#include "headers/NetworkSerialization.h"
#include "headers/NormalizationSerialization.h"
#include "headers/OptimizerSerialization.h"
#include "headers/LearningMemorySerialization.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {
constexpr char magic[] = "GLOOMYMD";
constexpr std::uint32_t format_version = 1;
constexpr std::uint64_t maximum_section_size = 100'000'000;
constexpr std::uint64_t checksum_offset = 1469598103934665603ULL;
constexpr std::uint64_t checksum_prime = 1099511628211ULL;
constexpr char normalization_magic[] = "GLOOMYNM";
constexpr std::uint32_t normalization_version = 1;
constexpr std::uint64_t maximum_dimensions = 1'000'000;

void writeBytes(std::ostream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write model file");
}

void readBytes(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated model file");
}

std::uint64_t checksum(const std::string& bytes) {
    std::uint64_t result = checksum_offset;
    for (unsigned char byte : bytes) {
        result ^= byte;
        result *= checksum_prime;
    }
    return result;
}

void writeUint64(std::ostream& stream, std::uint64_t value) {
    writeBytes(stream, &value, sizeof(value));
}

std::uint64_t readUint64(std::istream& stream) {
    std::uint64_t value = 0;
    readBytes(stream, &value, sizeof(value));
    return value;
}

void writeString(std::ostream& stream, const std::string& value) {
    const std::uint64_t size = value.size();
    writeUint64(stream, size);
    if (!value.empty()) writeBytes(stream, value.data(), value.size());
}

std::string readString(std::istream& stream) {
    const std::uint64_t size = readUint64(stream);
    if (size > maximum_section_size || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Model section name size is invalid");
    }
    std::string value(static_cast<size_t>(size), '\0');
    if (!value.empty()) readBytes(stream, value.data(), value.size());
    return value;
}

// tmpnam() genere un nom sans creer le fichier : une autre execution peut
// s'y glisser entre-temps (TOCTOU), et glibc le signale explicitement comme
// dangereux a la liaison. mkstemp() (POSIX) cree et ouvre le fichier de
// facon atomique ; on referme aussitot le descripteur puisque les appelants
// ne manipulent que le chemin (via les serializers existants, bases sur des
// chemins de fichiers).
std::string temporaryPath() {
    std::string pattern = (std::filesystem::temp_directory_path() / "gloomy_model_XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const int descriptor = mkstemp(buffer.data());
    if (descriptor == -1) {
        throw std::runtime_error("Unable to create a temporary file");
    }
    ::close(descriptor);
    return std::string(buffer.data());
}

std::string readFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open temporary file for reading");
    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

void writeFile(const std::string& path, const std::string& bytes) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to open temporary file for writing");
    writeBytes(stream, bytes.data(), bytes.size());
}

template <typename SaveFn>
std::string serializeComponent(SaveFn save_fn) {
    const std::string temp = temporaryPath();
    save_fn(temp);
    std::string bytes = readFile(temp);
    std::remove(temp.c_str());
    return bytes;
}

void writeSection(std::ostream& stream, const std::string& section_name, const std::string& section_bytes) {
    writeString(stream, section_name);
    writeUint64(stream, static_cast<std::uint64_t>(section_bytes.size()));
    if (!section_bytes.empty()) writeBytes(stream, section_bytes.data(), section_bytes.size());
}

std::pair<std::string, std::string> readSection(std::istream& stream) {
    const std::string section_name = readString(stream);
    const std::uint64_t size = readUint64(stream);
    if (size > maximum_section_size || size > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Model section size is invalid");
    }
    std::string section_bytes(static_cast<size_t>(size), '\0');
    if (!section_bytes.empty()) {
        readBytes(stream, section_bytes.data(), section_bytes.size());
    }
    return {section_name, section_bytes};
}

std::unique_ptr<StreamingNormalizer> loadNormalizerFromBytes(const std::string& bytes) {
    std::istringstream stream(bytes, std::ios::binary);
    char file_magic[sizeof(normalization_magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(normalization_magic, sizeof(normalization_magic) - 1)) {
        throw std::runtime_error("Invalid model normalization section magic");
    }

    std::uint32_t file_version = 0;
    readBytes(stream, &file_version, sizeof(file_version));
    if (file_version != normalization_version) {
        throw std::runtime_error("Unsupported model normalization section version");
    }

    std::uint64_t dimensions = 0;
    std::uint64_t count = 0;
    readBytes(stream, &dimensions, sizeof(dimensions));
    readBytes(stream, &count, sizeof(count));
    if (dimensions == 0 || dimensions > maximum_dimensions ||
        count == 0 || count > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("Model normalization section is invalid");
    }

    std::vector<double> means(static_cast<size_t>(dimensions));
    std::vector<double> variances(static_cast<size_t>(dimensions));
    std::vector<double> minimum(static_cast<size_t>(dimensions));
    std::vector<double> maximum(static_cast<size_t>(dimensions));

    readBytes(stream, means.data(), means.size() * sizeof(double));
    readBytes(stream, variances.data(), variances.size() * sizeof(double));
    readBytes(stream, minimum.data(), minimum.size() * sizeof(double));
    readBytes(stream, maximum.data(), maximum.size() * sizeof(double));

    auto normalizer = std::make_unique<StreamingNormalizer>(static_cast<size_t>(dimensions));
    normalizer->restore(static_cast<size_t>(count), means, variances, minimum, maximum);
    return normalizer;
}
}

void ModelSerialization::save(
    const std::string& path,
    const NeuralNetwork& network,
    const StreamingNormalizer& normalizer,
    const Optimizer& optimizer,
    const LearningMemory& memory
) {
    const std::string network_bytes = serializeComponent(
        [&](const std::string& temp_path) {
            NetworkSerialization::save(temp_path, network);
        }
    );

    const std::string normalization_bytes = serializeComponent(
        [&](const std::string& temp_path) {
            NormalizationSerialization::save(temp_path, normalizer);
        }
    );

    const std::string optimizer_bytes = serializeComponent(
        [&](const std::string& temp_path) {
            OptimizerSerialization::save(temp_path, optimizer);
        }
    );

    const std::string memory_bytes = serializeComponent(
        [&](const std::string& temp_path) {
            LearningMemorySerialization::save(temp_path, memory);
        }
    );

    std::ostringstream payload(std::ios::binary);
    writeBytes(payload, magic, sizeof(magic) - 1);
    writeBytes(payload, &format_version, sizeof(format_version));
    writeSection(payload, "NETWORK", network_bytes);
    writeSection(payload, "NORMALIZATION", normalization_bytes);
    writeSection(payload, "OPTIMIZER", optimizer_bytes);
    writeSection(payload, "MEMORY", memory_bytes);

    const std::string payload_bytes = payload.str();
    const std::uint64_t file_checksum = checksum(payload_bytes);

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Unable to open model file for writing");
    writeBytes(stream, payload_bytes.data(), payload_bytes.size());
    writeBytes(stream, &file_checksum, sizeof(file_checksum));
}

ModelSerialization::LoadedModel ModelSerialization::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open model file for reading");
    const std::string file_bytes{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    };
    if (file_bytes.size() <= sizeof(std::uint64_t)) {
        throw std::runtime_error("Model file is truncated");
    }

    const size_t payload_size = file_bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_checksum = 0;
    std::memcpy(&stored_checksum, file_bytes.data() + payload_size, sizeof(stored_checksum));
    const std::string payload = file_bytes.substr(0, payload_size);
    if (checksum(payload) != stored_checksum) {
        throw std::runtime_error("Model file checksum mismatch");
    }

    std::istringstream stream(payload, std::ios::binary);

    char file_magic[sizeof(magic) - 1];
    readBytes(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1)) {
        throw std::runtime_error("Invalid model file magic");
    }

    std::uint32_t version = 0;
    readBytes(stream, &version, sizeof(version));
    if (version != format_version) {
        throw std::runtime_error("Unsupported model file version");
    }

    LoadedModel model;
    bool has_network = false;
    bool has_normalization = false;
    bool has_optimizer = false;
    bool has_memory = false;

    while (stream.peek() != std::char_traits<char>::eof()) {
        const auto section = readSection(stream);

        if (section.first == "NETWORK") {
            const std::string temp_path = temporaryPath();
            writeFile(temp_path, section.second);
            NetworkSerialization::load(temp_path, model.network);
            std::remove(temp_path.c_str());
            has_network = true;
            continue;
        }

        if (section.first == "NORMALIZATION") {
            model.normalizer = loadNormalizerFromBytes(section.second);
            has_normalization = true;
            continue;
        }

        if (section.first == "OPTIMIZER") {
            const std::string temp_path = temporaryPath();
            writeFile(temp_path, section.second);
            model.optimizer = OptimizerSerialization::load(temp_path, model.network.layers());
            std::remove(temp_path.c_str());
            has_optimizer = true;
            continue;
        }

        if (section.first == "MEMORY") {
            const std::string temp_path = temporaryPath();
            writeFile(temp_path, section.second);
            model.memory = LearningMemorySerialization::load(temp_path);
            std::remove(temp_path.c_str());
            has_memory = true;
            continue;
        }

        throw std::runtime_error("Unknown model file section");
    }

    if (!has_network || !has_normalization || !has_optimizer || !has_memory) {
        throw std::runtime_error("Model file is missing required sections");
    }

    return model;
}
