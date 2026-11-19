#include "headers/OptimizerSerialization.h"
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace {
constexpr char magic[] = "GLOOMYOP";
constexpr std::uint32_t version = 1;
constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
constexpr std::uint64_t fnv_prime = 1099511628211ULL;
std::uint64_t checksum(const std::string& bytes) {
    std::uint64_t value = fnv_offset;
    for (unsigned char byte : bytes) { value ^= byte; value *= fnv_prime; }
    return value;
}
void write(std::ostream& stream, const void* data, size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to write optimizer checkpoint");
}
void read(std::istream& stream, void* data, size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated optimizer checkpoint");
}
}
void OptimizerSerialization::save(const std::string& path, const Optimizer& optimizer) {
    std::ostringstream payload(std::ios::binary);
    write(payload, magic, sizeof(magic) - 1); write(payload, &version, sizeof(version));
    const std::string type = optimizer.checkpointType();
    const std::uint64_t type_size = type.size(); write(payload, &type_size, sizeof(type_size));
    write(payload, type.data(), type.size()); optimizer.writeCheckpoint(payload);
    const std::string bytes = payload.str(); const std::uint64_t hash = checksum(bytes);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) throw std::runtime_error("Unable to open optimizer checkpoint for writing");
    write(file, bytes.data(), bytes.size()); write(file, &hash, sizeof(hash));
}
void OptimizerSerialization::load(const std::string& path, Optimizer& optimizer, const std::vector<DenseLayer>& layers) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Unable to open optimizer checkpoint for reading");
    const std::string bytes{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (bytes.size() < sizeof(magic) - 1 + sizeof(version) + sizeof(std::uint64_t) * 2)
        throw std::runtime_error("Optimizer checkpoint is truncated");
    const size_t payload_size = bytes.size() - sizeof(std::uint64_t);
    std::uint64_t stored_hash = 0; std::memcpy(&stored_hash, bytes.data() + payload_size, sizeof(stored_hash));
    const std::string payload = bytes.substr(0, payload_size);
    if (checksum(payload) != stored_hash) throw std::runtime_error("Optimizer checkpoint checksum mismatch");
    std::istringstream stream(payload, std::ios::binary);
    char file_magic[sizeof(magic) - 1]; read(stream, file_magic, sizeof(file_magic));
    if (std::string(file_magic, sizeof(file_magic)) != std::string(magic, sizeof(magic) - 1))
        throw std::runtime_error("Invalid optimizer checkpoint magic");
    std::uint32_t file_version = 0; read(stream, &file_version, sizeof(file_version));
    if (file_version != version) throw std::runtime_error("Unsupported optimizer checkpoint version");
    std::uint64_t type_size = 0; read(stream, &type_size, sizeof(type_size));
    if (type_size == 0 || type_size > 64) throw std::runtime_error("Invalid optimizer checkpoint type");
    std::string type(static_cast<size_t>(type_size), '\0'); read(stream, type.data(), type.size());
    if (type != optimizer.checkpointType()) throw std::runtime_error("Optimizer checkpoint type mismatch");
    optimizer.readCheckpoint(stream, layers);
    if (stream.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Unexpected optimizer checkpoint data");
}
