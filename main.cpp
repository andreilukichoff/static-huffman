/*
 * The MIT License
 *
 * Copyright (c) 2022 Andrei Lukichev
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <fmt/format.h>

/*
 * Static huffman encoder/decoder
 *
 * The implementation of the decoding algorithm is academic and not optimized
 * for performance. The encoding algorithm is much simpler and does not require
 * any major optimization.
 */

// TODO: canonical huffman codebook
// TODO: block encoding instead of entire file encoding
// TODO: optimize decoder for performance

#define BITS_IN_BYTE 8

using namespace std;
using namespace std::chrono;

// power of 2 constant expression
template<class T>
constexpr T pwrtwo(T exponent) {
    return (T(1) << exponent);
}

// frequency counter type
using freq_t = uint32_t;

// symbol type (usually single byte)
using symbol_t = uint8_t;

// bit accumulator type
using bitbuf_t = uint64_t;

enum compmethod_t : uint8_t {
    StaticHuffman = 1,
    BlockStaticHuffman,
    DynamicHuffman
};

struct header_t {
    union {
        char data_[14];
        struct {
            char signature[4];
            uint8_t version;
            compmethod_t comp_method;
            uint64_t source_length;
        };
    };
};

// extract file name from a full path
std::string basename(const std::string &filename) {
    if (filename.empty()) {
        return {};
    }

    auto len = filename.length();
    auto index = filename.find_last_of("/\\");

    if (index == std::string::npos) {
        return filename;
    }

    if (index + 1 >= len) {

        len--;
        index = filename.substr(0, len).find_last_of("/\\");

        if (len == 0) {
            return filename;
        }

        if (index == 0) {
            return filename.substr(1, len - 1);
        }

        if (index == std::string::npos) {
            return filename.substr(0, len);
        }

        return filename.substr(index + 1, len - index - 1);
    }

    return filename.substr(index + 1, len - index);
}

string btos(uint64_t b, uint8_t len) {
    auto s = fmt::format("{:b}", b);
    for (int i = s.length(); i < len; i++)
        s = "0" + s;
    return s;
}

class Node {
private:
    // save huffman code by recursive tree traversal
    void create_code(unsigned char c, uint64_t &code, uint8_t &len) {
        if (parent_ == nullptr) {
            code = code >> (32 - len);
            return;
        }

        if (parent_->left_ == this)
            code |= (uint32_t) 1 << (31 - len);

        len++;
        return parent_->create_code(c, code, len);
    }

public:
    Node() {
    }

    ~Node() {
        delete left_;
        delete right_;
    }

    Node(symbol_t s) {
        symbol = s;
        freq = 0;
    }

    Node(Node *left, Node *right) {
        left_ = left;
        right_ = right;
        freq = left->freq + right->freq;
    }

    Node *right_;
    Node *left_;
    Node *parent_;
    freq_t freq;
    symbol_t symbol;
    uint64_t code_;
    uint8_t len_;

    void create_code() {
        create_code(symbol, code_, len_);
    }
};

class HuffmanEncoder {
private:
    static constexpr const char sig[] = "huff";

    // max symbols in alphabet, if we use single byte symbol then it is 256
    static constexpr uint_fast32_t MaxSymbols = (pwrtwo(sizeof(symbol_t) * BITS_IN_BYTE));

    // sum of all symbol frequencies must not overflow freq value while building the tree, so we
    // rescale if frequency of any particular symbol exceeds 1/MaxSymbols of max frequency value
    static constexpr uint_fast32_t RescaleThreshold = ((pwrtwo(sizeof(freq_t) * BITS_IN_BYTE) - 1) / MaxSymbols);

    // how many bits we can accumulate before moving to write buffer
    static constexpr uint_fast32_t BitBufferSize = sizeof(bitbuf_t) * BITS_IN_BYTE;

    // how many bits to move at once when writing to buffer
    static constexpr uint_fast32_t BitFlushSize = BitBufferSize / 2;

    static constexpr uint_fast32_t ReadBufferSize = 2 * 1024;
    static constexpr uint_fast32_t WriteBufferSize = 2 * 1024;
    static constexpr uint_least8_t Version = 1;
    static constexpr uint64_t mask = ~0;

    istream *input_stream_;
    ostream *output_stream_;

    fpos<mbstate_t> input_stream_size_;

    // performance counters
    time_point<steady_clock, nanoseconds> start_time, encoding_start_time;

    // stream buffers
    char read_buf[ReadBufferSize];

    vector<Node *> symbols;

    // symbols in alphabet
    uint64_t alphabet_size;

    auto get_input_stream_size() {
        input_stream_->clear();
        input_stream_->seekg(0, ios::end);
        auto result = input_stream_->tellg();
        reset_input_stream();
        return result;
    }

    void reset_input_stream() {
        input_stream_->clear();
        input_stream_->seekg(0, ios::beg);
    }

    void rescale_freq_table(vector<Node *> &symbols) {
        for (int j = 0; j < MaxSymbols; j++) {
            if (symbols[j]->freq > 1)
                symbols[j]->freq /= 2;
        }
    }

    void build_freq_table() {
        symbols.clear();

        for (int c = 0; c < MaxSymbols; c++) {
            Node *node = new Node((symbol_t) c);
            node->symbol = (symbol_t) c;
            symbols.push_back(node);
        }

        reset_input_stream();

        // build tree:
        while (input_stream_->peek() != EOF) {
            input_stream_->read(read_buf, ReadBufferSize);
            auto bytes_read = input_stream_->gcount();
            for (int i = 0; i < bytes_read; i++) {
                symbols[(unsigned char) read_buf[i]]->freq++;
                if (symbols[(symbol_t) read_buf[i]]->freq == RescaleThreshold) {
                    rescale_freq_table(symbols);
                }
            }
        }
    }

    // todo: create progress struct and make this method external
    void report_progress(bool decoding = false, uint64_t source_length = 0) {
        auto now = steady_clock::now();
        auto total_elapsed_ns = (now - start_time).count();
        auto encoding_elapsed_ns = (now - encoding_start_time).count();

        auto progress = ((double) input_stream_->tellg() / (double) input_stream_size_) * 100;
        auto enc_mb_per_sec =
                (double) input_stream_->tellg() / ((double) encoding_elapsed_ns / 1000000000) / 1024 / 1024;
        auto total_mb_per_sec =
                (double) input_stream_->tellg() / ((double) total_elapsed_ns / 1000000000) / 1024 / 1024;
        auto ns_per_byte = (double) encoding_elapsed_ns / (double) input_stream_->tellg();
        auto output_size = output_stream_->tellp();

        double ratio;
        if (decoding) {
            ratio = 100 - 100 * (double) input_stream_size_ / source_length;
        } else {
            ratio = 100 - 100 * (double) output_size / input_stream_->tellg();
        }

        cout << fmt::format("progress: {:.2f}%, speed: {:.2f} MiB/s ({:.2f} ns/byte), "
                            "total speed: {:.2f} MiB/s, output size: {} MiB, ratio: {:.2f}%",
                            progress, enc_mb_per_sec, ns_per_byte, total_mb_per_sec, (output_size / 1024 / 1024), ratio)
             << endl;
    }

    void build_tree(vector<Node *> &tree) {
        // sort by frequency
        sort(tree.begin(), tree.end(), [](auto &a, auto &b) {
            return a->freq > b->freq;
        });

        // delete symbols with zero frequency
        while (!tree.empty() && !tree.back()->freq)
            tree.pop_back();

        alphabet_size = tree.size();

        while (tree.size() > 1) {
            Node *right = tree.back();
            tree.pop_back();
            Node *left = tree.back();
            tree.pop_back();
            Node *parent = new Node(left, right);
            left->parent_ = parent;
            right->parent_ = parent;
            tree.push_back(parent);

            sort(tree.begin(), tree.end(), [](auto &a, auto &b) {
                return a->freq > b->freq;
            });
        }

        // create codes
        for (int j = 0; j < MaxSymbols; j++) {
            if (symbols[j]->freq) {
                symbols[j]->create_code();
            }
        }
    }

    void encode_proc(const vector<Node *> &symbols) {
        auto current = start_time;
        auto now = steady_clock::now();
        encoding_start_time = now;
        auto total_bytes_read = 0;
        bitbuf_t bit_buf = 0;
        uint8_t bitIndex = 0;
        int writeBufIndex = 0;

        char write_buf[WriteBufferSize];

        while (input_stream_->peek() != EOF) {
            input_stream_->read(read_buf, ReadBufferSize);
            auto bytes_read = input_stream_->gcount();
            total_bytes_read += bytes_read;
            for (int i = 0; i < bytes_read; i++) {
                uint8_t symbol = read_buf[i];
                bit_buf = (bit_buf & ~(mask << bitIndex)) |
                          (symbols[symbol]->code_ << bitIndex);
                bitIndex += symbols[symbol]->len_;

                if (bitIndex >= BitFlushSize) {
                    *((uint32_t *) &write_buf[writeBufIndex]) = (uint32_t) bit_buf;

                    writeBufIndex += BitFlushSize / 8;

                    if (writeBufIndex >= WriteBufferSize) {
                        output_stream_->write(write_buf, writeBufIndex);
                        writeBufIndex = 0;
                    }

                    bitIndex -= BitFlushSize;
                    bit_buf >>= BitFlushSize;
                }
            }

            now = steady_clock::now();
            auto elapsed_ms = duration_cast<milliseconds>(now - current).count();
            if (elapsed_ms >= 500) {
                report_progress();
                current = now;
            }
        }

        // flush remaining bits
        if (bitIndex) {
            *((uint32_t *) &write_buf[writeBufIndex]) = (uint32_t) bit_buf;
            writeBufIndex += BitFlushSize / 8;
            output_stream_->write(write_buf, writeBufIndex);
        }

        input_stream_->clear();
        report_progress();
    }

    void read_freq_table() {
        symbols.clear();
        input_stream_->read(reinterpret_cast<char *>(&alphabet_size), sizeof(alphabet_size));

        for (int i = 0; i < MaxSymbols; i++) {
            Node *node = new Node((symbol_t) i);
            symbols.push_back(node);
        }

        for (int i = 0; i < alphabet_size; i++) {
            symbol_t s = input_stream_->get();
            input_stream_->read(reinterpret_cast<char *>(&symbols[s]->freq), sizeof(symbols[s]->freq));
        }
    }

    void decode_static_huffman_proc(uint64_t length) {
        read_freq_table();

        auto tree = symbols;

        build_tree(tree);

        uint64_t total_bytes_read = 0;
        auto now = start_time;
        auto current = now;
        uint_fast32_t write_buf_pos = 0;
        uint64_t decodedSymbols = 0;
        Node *root = tree.front();
        Node *nodeptr = root;

        char write_buf[WriteBufferSize] = {0};

        while (input_stream_->peek() != EOF) {
            input_stream_->read(read_buf, ReadBufferSize);
            auto bytes_read = input_stream_->gcount();

            if (!bytes_read)
                break;

            total_bytes_read += bytes_read;
            for (uint_fast32_t byte_index = 0; byte_index < bytes_read; byte_index++) {
                for (uint_fast8_t bit_index = 0; bit_index < BITS_IN_BYTE; bit_index++) {
                    uint_fast8_t bit = read_buf[byte_index] & 1 << bit_index;

                    if (bit) {
                        nodeptr = nodeptr->left_;
                    } else {
                        nodeptr = nodeptr->right_;
                    }

                    if (nodeptr->left_ == nodeptr->right_) {
                        write_buf[write_buf_pos] = nodeptr->symbol;
                        nodeptr = root;
                        if (++write_buf_pos == WriteBufferSize) {
                            output_stream_->write(write_buf, WriteBufferSize);
                            write_buf_pos = 0;
                        }
                        if (++decodedSymbols >= length)
                            break;
                    }

                }
                if (decodedSymbols >= length)
                    break;
            }

            now = steady_clock::now();
            auto elapsed_ms = duration_cast<milliseconds>(now - current).count();
            if (elapsed_ms >= 500) {
                report_progress(true, length);
                current = now;
            }
        }

        // flush buffer
        if (write_buf_pos) {
            output_stream_->write(write_buf, write_buf_pos);
        }

        input_stream_->clear();
        report_progress(true, length);
    }

    void write_header(uint8_t version, compmethod_t comp_method) {
        header_t header{
                .comp_method = comp_method,
                .version = version,
                .source_length = static_cast<uint64_t>(input_stream_size_)
        };
        strncpy(header.signature, sig, sizeof(header.signature));

        output_stream_->write(reinterpret_cast<const char *>(&header), sizeof(header));

        // write freq table
        output_stream_->write(reinterpret_cast<const char *>(&alphabet_size), sizeof(alphabet_size));
        for (int i = 0; i < symbols.size(); i++) {
            if (symbols[i]->freq) {
                output_stream_->put((char) i);
                output_stream_->write(reinterpret_cast<const char *>(&symbols[i]->freq), sizeof(symbols[i]->freq));
            }
        }
    }

public:
    static constexpr int ERR_INVALID_FILE_FORMAT = 101;
    static constexpr int ERR_INVALID_FILE_SIGNATURE = 102;
    static constexpr int ERR_UNSUPPORTED_FILE_VERSION = 103;
    static constexpr int ERR_UNSUPPORTED_COMP_METHOD = 104;
    header_t header;

    HuffmanEncoder(istream &input_stream, ostream &output_stream) {
        input_stream_ = &input_stream;
        output_stream_ = &output_stream;
    }

    int encode() {
        input_stream_size_ = get_input_stream_size();
        start_time = steady_clock::now();

        build_freq_table();

        // preserve original array for encoding
        auto tree = symbols;
        build_tree(tree);

        reset_input_stream();

        // write header:
        write_header(Version, compmethod_t::StaticHuffman);

        // encode:
        encode_proc(symbols);

        return EXIT_SUCCESS;
    }

    int decode() {
        input_stream_size_ = get_input_stream_size();
        start_time = steady_clock::now();
        encoding_start_time = start_time;

        input_stream_->read(reinterpret_cast<char *>(&header), sizeof(header));

        if (input_stream_->gcount() != sizeof(header)) {
            return ERR_INVALID_FILE_FORMAT;
        }

        if (strncmp(sig, header.signature, sizeof(header.signature))) {
            return ERR_INVALID_FILE_SIGNATURE;
        }

        if (header.version != Version) {
            return ERR_UNSUPPORTED_FILE_VERSION;
        }

        switch (header.comp_method) {
            case StaticHuffman:
                decode_static_huffman_proc(header.source_length);
                break;
            default:
                return ERR_UNSUPPORTED_COMP_METHOD;
                break;
        }

        return EXIT_SUCCESS;
    }
};

int main(int argc, char *argv[]) {
    static const int ERR_INVALID_ARGUMENTS = 1;
    static const int ERR_UNKNOWN_COMMAND = 2;
    static const int ERR_UNABLE_TO_OPEN_INPUT_FILE = 3;
    static const int ERR_UNABLE_TO_OPEN_OUTPUT_FILE = 4;
    static const int ERR_INVALID_FILE_FORMAT = 101;
    static const int ERR_INVALID_FILE_SIGNATURE = 102;
    static const int ERR_UNSUPPORTED_FILE_VERSION = 103;
    static const int ERR_UNSUPPORTED_COMP_METHOD = 104;
    static const int ERR_UNKNOWN = 255;

    if (argc != 4) {
        cout << "usage: " << basename(argv[0]) << " encode|decode [input_file] [output_file]" << endl;
        return ERR_INVALID_ARGUMENTS;
    }

    ifstream input_file;
    ofstream output_file;

    input_file.open(argv[2], ios::in | ios::binary);

    if (input_file.is_open()) {
        output_file.open(argv[3], ios::out | ios::trunc | ios::binary);
        if (output_file.is_open()) {
            if (!strcmp(argv[1], "encode")) {
                HuffmanEncoder encoder(input_file, output_file);
                int result = encoder.encode();
                if(result != EXIT_SUCCESS) {
                    switch(result) {
                        default:
                            cout << "Unknown error occured" << endl;
                            break;
                    }
                    return ERR_UNKNOWN;
                }
            } else if (!strcmp(argv[1], "decode")) {
                HuffmanEncoder encoder(input_file, output_file);
                int result = encoder.decode();
                if(result != EXIT_SUCCESS) {
                    output_file.close();
                    input_file.close();
                    switch(result) {
                        case HuffmanEncoder::ERR_INVALID_FILE_FORMAT:
                            cout << "Invalid file format" << endl;
                            return ERR_INVALID_FILE_FORMAT;
                            break;
                        case HuffmanEncoder::ERR_INVALID_FILE_SIGNATURE:
                            cout << "Invalid signature" << endl;
                            return ERR_INVALID_FILE_SIGNATURE;
                            break;
                        case HuffmanEncoder::ERR_UNSUPPORTED_COMP_METHOD:
                            cout << "Unsupported compression method: " << encoder.header.comp_method << endl;
                            return ERR_UNSUPPORTED_COMP_METHOD;
                            break;
                        case HuffmanEncoder::ERR_UNSUPPORTED_FILE_VERSION:
                            cout << "Unsupported version: " << encoder.header.version << endl;
                            return ERR_UNSUPPORTED_FILE_VERSION;
                            break;
                        default:
                            cout << "Unknown error occured: " << result << endl;
                            break;
                    }
                    return ERR_UNKNOWN;
                }
            } else {
                cout << "Unknown command: " << argv[1] << endl;
                return ERR_UNKNOWN_COMMAND;
            }
        } else {
            cout << "Unable to open output file: " << basename(argv[3]) << endl;
            input_file.close();
            return ERR_UNABLE_TO_OPEN_INPUT_FILE;
        }
    } else {
        cout << "Unable to open input file: " << basename(argv[2]) << endl;
        return ERR_UNABLE_TO_OPEN_OUTPUT_FILE;
    }

    output_file.close();
    input_file.close();

    cout << "done" << endl;

    return EXIT_SUCCESS;
}
