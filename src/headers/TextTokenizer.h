#ifndef TEXT_TOKENIZER_H
#define TEXT_TOKENIZER_H

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

class TextTokenizer {
public:
    static constexpr std::uint32_t bosId = 0;
    static constexpr std::uint32_t eosId = 1;
    static constexpr std::uint32_t unknownId = 2;
    static constexpr std::uint32_t version = 3;

    static TextTokenizer train(const std::vector<std::string>& texts, std::size_t maximum_vocabulary);
    std::vector<std::uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<std::uint32_t>& tokens) const;
    std::size_t vocabularySize() const;
    const std::vector<std::string>& pieces() const;
    const std::vector<std::uint32_t>& mergeLeft() const;
    const std::vector<std::uint32_t>& mergeRight() const;
    const std::vector<std::uint32_t>& mergeOutput() const;
    void write(std::ostream& stream) const;
    static TextTokenizer read(std::istream& stream);

    static std::string normalizeText(const std::string& text);

private:
    void rebuildIndex();
    std::vector<std::string> token_pieces;
    std::vector<std::uint32_t> merge_left;
    std::vector<std::uint32_t> merge_right;
    std::vector<std::uint32_t> merge_output;
    std::unordered_map<std::string, std::uint32_t> piece_ids;
};

#endif // TEXT_TOKENIZER_H
