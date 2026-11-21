#include "headers/TextTokenizer.h"
#include "UnicodeNormalizationData.h"
#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>

namespace {
std::vector<std::string> splitUtf8(const std::string& text) {
    std::vector<std::string> characters;
    for (std::size_t index = 0; index < text.size();) {
        const unsigned char first = static_cast<unsigned char>(text[index]);
        std::size_t width = 0;
        if (first <= 0x7f) width = 1;
        else if (first >= 0xc2 && first <= 0xdf) width = 2;
        else if (first >= 0xe0 && first <= 0xef) width = 3;
        else if (first >= 0xf0 && first <= 0xf4) width = 4;
        else throw std::invalid_argument("Text is not valid UTF-8");
        if (index + width > text.size()) throw std::invalid_argument("Text ends inside a UTF-8 character");
        for (std::size_t offset = 1; offset < width; ++offset) {
            const unsigned char continuation = static_cast<unsigned char>(text[index + offset]);
            if ((continuation & 0xc0) != 0x80) throw std::invalid_argument("Text is not valid UTF-8");
        }
        if (width == 3) {
            const unsigned char second = static_cast<unsigned char>(text[index + 1]);
            if ((first == 0xe0 && second < 0xa0) || (first == 0xed && second >= 0xa0))
                throw std::invalid_argument("Text contains an invalid UTF-8 code point");
        }
        if (width == 4) {
            const unsigned char second = static_cast<unsigned char>(text[index + 1]);
            if ((first == 0xf0 && second < 0x90) || (first == 0xf4 && second >= 0x90))
                throw std::invalid_argument("Text contains an invalid UTF-8 code point");
        }
        characters.push_back(text.substr(index, width));
        index += width;
    }
    return characters;
}

void writeBytes(std::ostream& stream, const void* data, std::size_t size) {
    stream.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Unable to serialize text tokenizer");
}

void readBytes(std::istream& stream, void* data, std::size_t size) {
    stream.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (!stream) throw std::runtime_error("Invalid or truncated text tokenizer");
}

void writeString(std::ostream& stream, const std::string& value) {
    const std::uint64_t size = value.size();
    writeBytes(stream, &size, sizeof(size));
    if (size) writeBytes(stream, value.data(), value.size());
}

std::string readString(std::istream& stream) {
    std::uint64_t size = 0;
    readBytes(stream, &size, sizeof(size));
    if (size > 1'000'000) throw std::runtime_error("Tokenizer piece exceeds size limit");
    std::string value(static_cast<std::size_t>(size), '\0');
    if (size) readBytes(stream, value.data(), value.size());
    return value;
}

std::uint64_t pairKey(std::uint32_t left, std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

std::string safeUtf8(const std::string& input) {
    std::string output;
    for (std::size_t index = 0; index < input.size();) {
        const auto first=static_cast<unsigned char>(input[index]);
        std::size_t width=first<=0x7f?1:first>=0xc2&&first<=0xdf?2:first>=0xe0&&first<=0xef?3:first>=0xf0&&first<=0xf4?4:0;
        bool valid=width&&index+width<=input.size();
        for(std::size_t j=1;valid&&j<width;++j)valid=(static_cast<unsigned char>(input[index+j])&0xc0)==0x80;
        if(valid&&width==3){const auto second=static_cast<unsigned char>(input[index+1]);valid=!((first==0xe0&&second<0xa0)||(first==0xed&&second>=0xa0));}
        if(valid&&width==4){const auto second=static_cast<unsigned char>(input[index+1]);valid=!((first==0xf0&&second<0x90)||(first==0xf4&&second>=0x90));}
        if(valid){output.append(input,index,width);index+=width;}
        else{output+="\xef\xbf\xbd";++index;}
    }
    return output;
}

std::uint32_t codepoint(const std::string& bytes) {
    const auto first = static_cast<unsigned char>(bytes[0]);
    if (bytes.size() == 1) return first;
    std::uint32_t value = first & (bytes.size() == 2 ? 0x1f : bytes.size() == 3 ? 0x0f : 0x07);
    for (std::size_t index = 1; index < bytes.size(); ++index)
        value = (value << 6) | (static_cast<unsigned char>(bytes[index]) & 0x3f);
    return value;
}

void appendUtf8(std::string& output, std::uint32_t value) {
    if (value <= 0x7f) output.push_back(static_cast<char>(value));
    else if (value <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (value >> 6)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else if (value <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (value >> 12)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (value >> 18)));
        output.push_back(static_cast<char>(0x80 | ((value >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((value >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (value & 0x3f)));
    }
}

std::uint8_t combiningClass(std::uint32_t value) {
    const auto* first = std::begin(gloomy_unicode::combining_classes);
    const auto* last = std::end(gloomy_unicode::combining_classes);
    const auto* found = std::lower_bound(first, last, value,
        [](const gloomy_unicode::CombiningClass& entry, std::uint32_t key) { return entry.code < key; });
    return found != last && found->code == value ? found->value : 0;
}

void canonicalDecompose(std::uint32_t value, std::vector<std::uint32_t>& output) {
    constexpr std::uint32_t s_base=0xac00, l_base=0x1100, v_base=0x1161, t_base=0x11a7;
    constexpr std::uint32_t l_count=19, v_count=21, t_count=28, n_count=v_count*t_count, s_count=l_count*n_count;
    if (value >= s_base && value < s_base+s_count) {
        const auto index=value-s_base;
        output.push_back(l_base+index/n_count);
        output.push_back(v_base+(index%n_count)/t_count);
        if (index%t_count) output.push_back(t_base+index%t_count);
        return;
    }
    const auto* first=std::begin(gloomy_unicode::decompositions);
    const auto* last=std::end(gloomy_unicode::decompositions);
    const auto* found=std::lower_bound(first,last,value,
        [](const gloomy_unicode::Decomposition& entry,std::uint32_t key){return entry.code<key;});
    if(found==last||found->code!=value){output.push_back(value);return;}
    for(std::size_t i=0;i<found->length;++i)
        canonicalDecompose(gloomy_unicode::decomposition_values[found->offset+i],output);
}

std::uint32_t compose(std::uint32_t first, std::uint32_t second) {
    constexpr std::uint32_t s_base=0xac00, l_base=0x1100, v_base=0x1161, t_base=0x11a7;
    constexpr std::uint32_t l_count=19, v_count=21, t_count=28, n_count=v_count*t_count, s_count=l_count*n_count;
    if(first>=l_base&&first<l_base+l_count&&second>=v_base&&second<v_base+v_count)
        return s_base+((first-l_base)*v_count+(second-v_base))*t_count;
    if(first>=s_base&&first<s_base+s_count&&(first-s_base)%t_count==0&&second>t_base&&second<t_base+t_count)
        return first+(second-t_base);
    const auto* begin=std::begin(gloomy_unicode::compositions);
    const auto* end=std::end(gloomy_unicode::compositions);
    const auto* found=std::lower_bound(begin,end,std::pair<std::uint32_t,std::uint32_t>{first,second},
        [](const gloomy_unicode::Composition& entry,const auto& key){return entry.first<key.first||(entry.first==key.first&&entry.second<key.second);});
    return found!=end&&found->first==first&&found->second==second?found->result:0;
}

std::vector<std::uint32_t> normalizeNfc(const std::vector<std::uint32_t>& input) {
    std::vector<std::uint32_t> decomposed;
    for(auto value:input)canonicalDecompose(value,decomposed);
    for(std::size_t i=1;i<decomposed.size();++i){const auto current=combiningClass(decomposed[i]);if(!current)continue;std::size_t j=i;while(j>0){const auto previous=combiningClass(decomposed[j-1]);if(previous==0||previous<=current)break;std::swap(decomposed[j],decomposed[j-1]);--j;}}
    if(decomposed.empty())return decomposed;
    std::vector<std::uint32_t> composed;composed.reserve(decomposed.size());composed.push_back(decomposed[0]);
    std::size_t starter_position=0;std::uint32_t starter=decomposed[0];std::uint8_t previous_class=0;
    for(std::size_t i=1;i<decomposed.size();++i){const auto value=decomposed[i];const auto current_class=combiningClass(value);const auto combined=compose(starter,value);
        if(combined&&(previous_class<current_class||previous_class==0)){composed[starter_position]=combined;starter=combined;}
        else{if(current_class==0){starter_position=composed.size();starter=value;}composed.push_back(value);previous_class=current_class;}}
    return composed;
}
}

std::string TextTokenizer::normalizeText(const std::string& text) {
    const std::vector<std::string> characters = splitUtf8(text);
    std::vector<std::uint32_t> codepoints;
    codepoints.reserve(characters.size());
    for (std::size_t index = 0; index < characters.size(); ++index) {
        const auto value=codepoint(characters[index]);
        if(value=='\r'){codepoints.push_back('\n');if(index+1<characters.size()&&characters[index+1]=="\n")++index;}
        else codepoints.push_back(value);
    }
    if(!codepoints.empty()&&codepoints.front()==0xfeff)codepoints.erase(codepoints.begin());
    const auto canonical=normalizeNfc(codepoints);std::string normalized;normalized.reserve(text.size());
    for(auto value:canonical)appendUtf8(normalized,value);
    return normalized;
}

TextTokenizer TextTokenizer::train(const std::vector<std::string>& texts, std::size_t maximum_vocabulary) {
    if (texts.empty()) throw std::invalid_argument("Tokenizer training corpus is empty");
    if (maximum_vocabulary < 259 || maximum_vocabulary > 1'000'000)
        throw std::invalid_argument("Byte-level BPE vocabulary size must be between 259 and 1000000");

    std::vector<std::string> normalized_texts;
    for (const std::string& text : texts) {
        normalized_texts.push_back(normalizeText(text));
    }

    TextTokenizer tokenizer;
    tokenizer.token_pieces = {"<bos>", "<eos>", "<unk>"};
    for (std::size_t byte = 0; byte < 256; ++byte) tokenizer.token_pieces.emplace_back(1, static_cast<char>(byte));
    tokenizer.rebuildIndex();

    std::vector<std::vector<std::uint32_t>> sequences;
    sequences.reserve(normalized_texts.size());
    for (const auto& text : normalized_texts) {
        std::vector<std::uint32_t> ids;
        ids.reserve(text.size());
        for (unsigned char byte : text) ids.push_back(static_cast<std::uint32_t>(byte)+3);
        sequences.push_back(std::move(ids));
    }

    while (tokenizer.token_pieces.size() < maximum_vocabulary) {
        std::map<std::uint64_t, std::size_t> counts;
        for (const auto& sequence : sequences)
            for (std::size_t index = 0; index + 1 < sequence.size(); ++index)
                ++counts[pairKey(sequence[index], sequence[index + 1])];
        if (counts.empty()) break;
        const auto best = std::max_element(counts.begin(), counts.end(),
            [](const auto& left, const auto& right) {
                return left.second < right.second ||
                    (left.second == right.second && left.first > right.first);
            });
        if (best->second < 2) break;
        const std::uint32_t left = static_cast<std::uint32_t>(best->first >> 32);
        const std::uint32_t right = static_cast<std::uint32_t>(best->first & 0xffffffffu);
        const std::string merged_piece = tokenizer.token_pieces.at(left) + tokenizer.token_pieces.at(right);
        if (merged_piece.size() > 1'000'000) throw std::runtime_error("Tokenizer merge exceeds size limit");
        const std::uint32_t merged = static_cast<std::uint32_t>(tokenizer.token_pieces.size());
        tokenizer.token_pieces.push_back(merged_piece);
        tokenizer.merge_left.push_back(left);
        tokenizer.merge_right.push_back(right);
        tokenizer.merge_output.push_back(merged);
        tokenizer.piece_ids.emplace(merged_piece, merged);

        for (auto& sequence : sequences) {
            std::vector<std::uint32_t> output;
            output.reserve(sequence.size());
            for (std::size_t index = 0; index < sequence.size();) {
                if (index + 1 < sequence.size() && sequence[index] == left && sequence[index + 1] == right) {
                    output.push_back(merged);
                    index += 2;
                } else {
                    output.push_back(sequence[index++]);
                }
            }
            sequence.swap(output);
        }
    }
    return tokenizer;
}

std::vector<std::uint32_t> TextTokenizer::encode(const std::string& text) const {
    const auto normalized = normalizeText(text);
    std::vector<std::uint32_t> ids;
    ids.reserve(normalized.size());
    for (unsigned char byte : normalized) ids.push_back(static_cast<std::uint32_t>(byte)+3);
    for (std::size_t merge = 0; merge < merge_left.size(); ++merge) {
        std::vector<std::uint32_t> output;
        output.reserve(ids.size());
        for (std::size_t index = 0; index < ids.size();) {
            if (index + 1 < ids.size() && ids[index] == merge_left[merge] && ids[index + 1] == merge_right[merge]) {
                output.push_back(merge_output[merge]);
                index += 2;
            } else {
                output.push_back(ids[index++]);
            }
        }
        ids.swap(output);
    }
    return ids;
}

std::string TextTokenizer::decode(const std::vector<std::uint32_t>& tokens) const {
    std::string text;
    for (std::uint32_t token : tokens) {
        if (token == bosId || token == eosId) continue;
        if (token >= token_pieces.size()) throw std::invalid_argument("Token id is outside vocabulary");
        if (token != unknownId) text += token_pieces[token];
        else text += "�";
    }
    return safeUtf8(text);
}

std::size_t TextTokenizer::vocabularySize() const { return token_pieces.size(); }
const std::vector<std::string>& TextTokenizer::pieces() const { return token_pieces; }
const std::vector<std::uint32_t>& TextTokenizer::mergeLeft() const { return merge_left; }
const std::vector<std::uint32_t>& TextTokenizer::mergeRight() const { return merge_right; }
const std::vector<std::uint32_t>& TextTokenizer::mergeOutput() const { return merge_output; }

void TextTokenizer::write(std::ostream& stream) const {
    const std::uint32_t tokenizer_version = version;
    const std::uint64_t piece_count = token_pieces.size();
    const std::uint64_t merge_count = merge_left.size();
    writeBytes(stream, &tokenizer_version, sizeof(tokenizer_version));
    writeString(stream, gloomy_unicode::data_version);
    writeBytes(stream, &piece_count, sizeof(piece_count));
    for (const std::string& piece : token_pieces) writeString(stream, piece);
    writeBytes(stream, &merge_count, sizeof(merge_count));
    for (std::size_t index = 0; index < merge_left.size(); ++index) {
        writeBytes(stream, &merge_left[index], sizeof(std::uint32_t));
        writeBytes(stream, &merge_right[index], sizeof(std::uint32_t));
        writeBytes(stream, &merge_output[index], sizeof(std::uint32_t));
    }
}

TextTokenizer TextTokenizer::read(std::istream& stream) {
    TextTokenizer tokenizer;
    std::uint32_t tokenizer_version = 0;
    std::uint64_t piece_count = 0;
    readBytes(stream, &tokenizer_version, sizeof(tokenizer_version));
    const std::string unicode_version = readString(stream);
    readBytes(stream, &piece_count, sizeof(piece_count));
    if (tokenizer_version != version || unicode_version != gloomy_unicode::data_version || piece_count < 259 || piece_count > 1'000'000)
        throw std::runtime_error("Unsupported or invalid tokenizer version/vocabulary");
    tokenizer.token_pieces.reserve(static_cast<std::size_t>(piece_count));
    for (std::uint64_t index = 0; index < piece_count; ++index)
        tokenizer.token_pieces.push_back(readString(stream));
    std::uint64_t merge_count = 0;
    readBytes(stream, &merge_count, sizeof(merge_count));
    if (merge_count > piece_count - 259) throw std::runtime_error("Tokenizer merge count is invalid");
    tokenizer.merge_left.resize(static_cast<std::size_t>(merge_count));
    tokenizer.merge_right.resize(static_cast<std::size_t>(merge_count));
    tokenizer.merge_output.resize(static_cast<std::size_t>(merge_count));
    for (std::size_t index = 0; index < merge_count; ++index) {
        readBytes(stream, &tokenizer.merge_left[index], sizeof(std::uint32_t));
        readBytes(stream, &tokenizer.merge_right[index], sizeof(std::uint32_t));
        readBytes(stream, &tokenizer.merge_output[index], sizeof(std::uint32_t));
        const std::uint32_t expected_output=static_cast<std::uint32_t>(259+index);
        if (tokenizer.merge_left[index] >= expected_output || tokenizer.merge_right[index] >= expected_output ||
            tokenizer.merge_output[index] != expected_output)
            throw std::runtime_error("Tokenizer merge token id is invalid");
    }
    if(tokenizer.token_pieces[0]!="<bos>"||tokenizer.token_pieces[1]!="<eos>"||tokenizer.token_pieces[2]!="<unk>")
        throw std::runtime_error("Tokenizer special tokens are invalid");
    for(std::size_t byte=0;byte<256;++byte)if(tokenizer.token_pieces[byte+3]!=std::string(1,static_cast<char>(byte)))
        throw std::runtime_error("Tokenizer byte fallback vocabulary is invalid");
    tokenizer.rebuildIndex();
    return tokenizer;
}

void TextTokenizer::rebuildIndex() {
    piece_ids.clear();
    for (std::size_t index = 0; index < token_pieces.size(); ++index) {
        if (!piece_ids.emplace(token_pieces[index], static_cast<std::uint32_t>(index)).second && index < 3)
            throw std::runtime_error("Tokenizer special token collision");
    }
}
