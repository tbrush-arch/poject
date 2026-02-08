// ParallelParser.cpp

#include "pch.h"
#include "ParallelParser.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>
#include <cstdlib>
#include <thread>
#include <future>
#include <cstring>
#include <cmath>
#include <cctype>

// --- Google Highway Includes ---
#include "hwy/highway.h"
#include "hwy/contrib/algo/find-inl.h"

// --- Linux mmap Stuff ---
#include <sys/mman.h>   // mmap, munmap
#include <sys/stat.h>   // fstat
#include <fcntl.h>      // open
#include <unistd.h>     // close, sysconf

// --- Lookup Tables for Fast Math ---
alignas(64) const double POW10_TABLE[16] = {
    1.0e0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7,
    1.0e8, 1.0e9, 1.0e10, 1.0e11, 1.0e12, 1.0e13, 1.0e14, 1.0e15
};

// --- SIMD Lookup Table for Right Alignment ---
alignas(64) const int8_t RIGHT_ALIGN_LUT[17][16] = {
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 1, 2},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 1, 2, 3},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 1, 2, 3, 4},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 1, 2, 3, 4, 5},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6},
    {-1,-1,-1,-1,-1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7},
    {-1,-1,-1,-1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
    {-1,-1,-1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    {-1,-1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10},
    {-1,-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11},
    {-1,-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12},
    {-1,-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13},
    {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14},
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15}
};

// --- Helper classes for Linux Resource Management ---

class ScopedFile {
public:
    int fd;
    ScopedFile(const std::string& filename) {
        fd = open(filename.c_str(), O_RDONLY);
    }
    ~ScopedFile() {
        if (fd != -1) close(fd);
    }
    bool isValid() const { return fd != -1; }
};

class ScopedMap {
public:
    void* addr;
    size_t length;

    ScopedMap(int fd, size_t len) : length(len) {
        addr = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
    }

    ~ScopedMap() {
        if (addr != MAP_FAILED && addr != nullptr) {
            munmap(addr, length);
        }
    }

    bool isValid() const { return addr != MAP_FAILED && addr != nullptr; }
    char* get() const { return static_cast<char*>(addr); }
};

// ---------------------------------------------------------
// SIMD Kernels using Google Highway
// ---------------------------------------------------------

HWY_BEFORE_NAMESPACE();
namespace hwy_parser {
    namespace HWY_NAMESPACE {

        namespace hn = hwy::HWY_NAMESPACE;

        /**
         * @brief Scans for the next newline character.
         */
        inline char* FindNextNewlineHwy(char* p, const char* pEnd) {
            void* found = std::memchr(p, '\n', pEnd - p);
            return found ? static_cast<char*>(found) + 1 : const_cast<char*>(pEnd);
        }

        /**
         * @brief Skips whitespace characters using Highway.
         */
        inline char* SkipWhitespaceHwy(char* p, const char* pEnd) {
            const hn::ScalableTag<uint8_t> d;
            const auto space_limit = hn::Set(d, 32);
            while (p + hn::Lanes(d) <= pEnd) {
                auto chunk = hn::LoadU(d, reinterpret_cast<const uint8_t*>(p));
                auto mask = chunk > space_limit;
                intptr_t pos = hn::FindFirstTrue(d, mask);
                if (pos >= 0) return p + pos;
                p += hn::Lanes(d);
            }
            while (p < pEnd && (unsigned char)*p <= 32) p++;
            return const_cast<char*>(p);
        }

        /**
         * @brief Scans for the next delimiter using Highway.
         */
        inline char* FindDelimiterHwy(char* p, const char* pEnd) {
            const hn::ScalableTag<uint8_t> d;
            const auto space = hn::Set(d, ' ');
            const auto newline = hn::Set(d, '\n');
            const auto cr = hn::Set(d, '\r');
            const auto tab = hn::Set(d, '\t');

            while (p + hn::Lanes(d) <= pEnd) {
                auto chunk = hn::LoadU(d, reinterpret_cast<const uint8_t*>(p));
                auto combined = hn::Or(hn::Or(chunk == space, chunk == newline),
                    hn::Or(chunk == cr, chunk == tab));
                intptr_t pos = hn::FindFirstTrue(d, combined);
                if (pos >= 0) return p + pos;
                p += hn::Lanes(d);
            }
            while (p < pEnd && !(*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
            return const_cast<char*>(p);
        }

        /**
         * @brief Simultaneous SIMD Parser using Highway.
         */
        void ParseChunkTrueSimdHwy(char* pStart, char* pEnd, bool is_real_or_int, MatrixEntry* output_ptr) {
            char* p = pStart;
            size_t idx = 0;

            const hn::ScalableTag<uint8_t> d8;
            const hn::ScalableTag<uint16_t> d16;
            const hn::ScalableTag<int32_t> d32;

            const auto v_zero_char = hn::Set(d8, '0');
            const auto v_mult_pairs = hn::Set(d16, 0x010A); // (10, 1)
            const auto v_mult_quads = hn::Set(d32, 0x00010064); // (100, 1)

            while (p + 64 <= pEnd) {
                auto chunk = hn::LoadU(d8, reinterpret_cast<const uint8_t*>(p));
                auto newline_mask = (chunk == hn::Set(d8, '\n'));

                size_t processed_offset = 0;

                while (hn::CountTrue(d8, newline_mask) > 0) {
                    intptr_t relative_nl_idx = hn::FindFirstTrue(d8, newline_mask);
                    int line_len = relative_nl_idx - processed_offset;
                    char* line_ptr = p + processed_offset;

                    if (line_len > 0 && *line_ptr != '%') {
                        auto line_vec = hn::LoadU(d8, reinterpret_cast<const uint8_t*>(line_ptr));
                        auto space_mask = (line_vec == hn::Set(d8, ' '));

                        intptr_t space1 = hn::FindFirstTrue(d8, space_mask);
                        auto space_mask2 = hn::AndNot(hn::FirstN(d8, space1 + 1), space_mask);
                        intptr_t space2 = hn::FindFirstTrue(d8, space_mask2);

                        int off_r = 0; while (off_r < space1 && line_ptr[off_r] == ' ') off_r++;
                        int len_r = (int)space1 - off_r;
                        int off_c = (int)space1 + 1; while (off_c < space2 && line_ptr[off_c] == ' ') off_c++;
                        int len_c = (int)space2 - off_c;
                        int off_v = (int)space2 + 1; while (off_v < line_len && line_ptr[off_v] == ' ') off_v++;
                        int len_v = line_len - off_v;

                        bool neg_v = (len_v > 0 && line_ptr[off_v] == '-');

                        // STEP 3: SIMULTANEOUS GATHER (Permute substrings into 16-byte lanes)
                        alignas(64) int8_t idx_map[64] = { 0 };
                        for (int i = 0; i < 16; ++i) {
                            idx_map[i] = (int8_t)(off_r + i);       // Lane 0: Row
                            idx_map[i + 16] = (int8_t)(off_c + i);  // Lane 1: Col
                            idx_map[i + 32] = (int8_t)(off_v + i);  // Lane 2: Value
                        }
                        auto v_permute_ctrl = hn::Load(d8, reinterpret_cast<const uint8_t*>(idx_map));
                        auto v_batch = hn::TableLookupBytes(line_vec, v_permute_ctrl);

                        // STEP 4: RIGHT-ALIGN (Shuffle digits to fixed positions in lanes)
                        alignas(64) int8_t combined_shuf[64] = { 0 };
                        std::memcpy(combined_shuf, RIGHT_ALIGN_LUT[len_r], 16);
                        std::memcpy(combined_shuf + 16, RIGHT_ALIGN_LUT[len_c], 16);
                        std::memcpy(combined_shuf + 32, RIGHT_ALIGN_LUT[len_v], 16);
                        auto v_shuffles = hn::Load(d8, reinterpret_cast<const uint8_t*>(combined_shuf));
                        v_batch = hn::TableLookupBytes(v_batch, v_shuffles);

                        // STEP 5: CONVERT digits to integers simultaneously
                        auto v_nums = hn::Sub(v_batch, v_zero_char);
                        v_nums = hn::Max(v_nums, hn::Set(d8, 0)); // Clamp negatives/padding to 0

                        // Pairwise multiply-add: Digit*10 + Digit -> uint16_t
                        auto v_pairs = hn::SumOfMul(v_nums, hn::BitCast(d8, v_mult_pairs));
                        // Quadwise multiply-add: Pair*100 + Pair -> uint32_t
                        auto v_quads = hn::SumOfMul(v_pairs, hn::BitCast(d16, v_mult_quads));

                        alignas(64) int32_t final_parts[16];
                        hn::Store(v_quads, d32, final_parts);

                        auto combine_lane = [&](int lane_idx) -> int {
                            int off = lane_idx * 4;
                            return final_parts[off] * 1000000000000LL +
                                final_parts[off + 1] * 100000000 +
                                final_parts[off + 2] * 10000 +
                                final_parts[off + 3];
                            };

                        output_ptr[idx].row = combine_lane(0);
                        output_ptr[idx].col = combine_lane(1);

                        if (!is_real_or_int) {
                            int v_int = combine_lane(2);
                            output_ptr[idx].value = neg_v ? -v_int : v_int;
                        }
                        else {
                            output_ptr[idx].value = std::strtod(line_ptr + off_v, nullptr);
                        }
                        idx++;
                    }
                    processed_offset = relative_nl_idx + 1;
                    newline_mask = hn::AndNot(hn::FirstN(d8, processed_offset), newline_mask);
                }
                p += processed_offset;
            }

            // Scalar fallback for the tail
            while (p < pEnd) {
                char* next_nl = p;
                while (next_nl < pEnd && *next_nl != '\n') next_nl++;
                if (next_nl > p && *p != '%') {
                    char* end_ptr;
                    output_ptr[idx].row = (int)std::strtol(p, &end_ptr, 10);
                    output_ptr[idx].col = (int)std::strtol(end_ptr, &end_ptr, 10);
                    output_ptr[idx].value = std::strtod(end_ptr, nullptr);
                    idx++;
                }
                p = (next_nl < pEnd) ? next_nl + 1 : pEnd;
            }
        }

        /**
         * @brief Scans a chunk to count valid entries.
         */
        size_t CountEntriesInChunkHwy(char* pStart, char* pEnd) {
            size_t count = 0;
            char* p = pStart;
            while (p < pEnd) {
                p = SkipWhitespaceHwy(p, pEnd);
                if (p >= pEnd) break;
                if (*p == '%') { p = FindNextNewlineHwy(p, pEnd); continue; }
                count++;
                p = FindNextNewlineHwy(p, pEnd);
            }
            return count;
        }

    } // namespace HWY_NAMESPACE
} // namespace hwy_parser
HWY_AFTER_NAMESPACE();

// ---------------------------------------------------------
// Top-Level Functions
// ---------------------------------------------------------

namespace hn = hwy_parser::HWY_NAMESPACE;

bool parallelParser(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries) {
    rows = 0; cols = 0; entries.clear();

    ScopedFile file(filename);
    if (!file.isValid()) return false;
    struct stat sb;
    if (fstat(file.fd, &sb) == -1 || sb.st_size == 0) return false;

    ScopedMap mappedFile(file.fd, sb.st_size);
    if (!mappedFile.isValid()) return false;

    char* pCurrent = mappedFile.get();
    const char* const pEnd = pCurrent + sb.st_size;

    // --- Header Parsing ---
    bool banner_found = false;
    std::string field;
    while (pCurrent < pEnd) {
        if (*pCurrent == '%' && pCurrent + 1 < pEnd && *(pCurrent + 1) == '%') {
            char* line_end = hn::FindNextNewlineHwy(pCurrent, pEnd);
            std::string line(pCurrent, line_end - pCurrent);
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            std::stringstream ss(line);
            std::string tok, obj, fmt, sym;
            ss >> tok >> obj >> fmt >> field >> sym;
            if (tok == "%%MatrixMarket" && obj == "matrix" && fmt == "coordinate") {
                std::transform(field.begin(), field.end(), field.begin(), ::tolower);
                banner_found = true;
                pCurrent = line_end;
                break;
            }
        }
        pCurrent = hn::FindNextNewlineHwy(pCurrent, pEnd);
    }
    if (!banner_found) return false;

    // --- Dimension Parsing ---
    int num_entries = 0;
    while (pCurrent < pEnd) {
        pCurrent = hn::SkipWhitespaceHwy(pCurrent, pEnd);
        if (pCurrent >= pEnd) break;
        if (*pCurrent == '%') { pCurrent = hn::FindNextNewlineHwy(pCurrent, pEnd); continue; }

        char* pNext = hn::FindDelimiterHwy(pCurrent, pEnd);
        rows = (int)std::strtol(pCurrent, nullptr, 10);
        pCurrent = hn::SkipWhitespaceHwy(pNext, pEnd);
        pNext = hn::FindDelimiterHwy(pCurrent, pEnd);
        cols = (int)std::strtol(pCurrent, nullptr, 10);
        pCurrent = hn::SkipWhitespaceHwy(pNext, pEnd);
        pNext = hn::FindDelimiterHwy(pCurrent, pEnd);
        num_entries = (int)std::strtol(pCurrent, nullptr, 10);
        pCurrent = hn::FindNextNewlineHwy(pNext, pEnd);
        break;
    }

    // --- Parallel Threading ---
    unsigned int num_threads = std::thread::hardware_concurrency();
    long long chunk_size = (pEnd - pCurrent) / num_threads;
    std::vector<char*> starts, ends;
    char* chunk_start = pCurrent;

    for (unsigned int i = 0; i < num_threads; ++i) {
        if (chunk_start >= pEnd) break;
        char* chunk_end = (i == num_threads - 1) ? const_cast<char*>(pEnd) : chunk_start + chunk_size;
        chunk_end = hn::FindNextNewlineHwy(chunk_end, pEnd);
        starts.push_back(chunk_start); ends.push_back(chunk_end);
        chunk_start = chunk_end;
    }

    // Pass 1: Count entries per chunk
    std::vector<std::future<size_t>> count_futures;
    for (size_t i = 0; i < starts.size(); ++i)
        count_futures.push_back(std::async(std::launch::async, hn::CountEntriesInChunkHwy, starts[i], ends[i]));

    std::vector<size_t> offsets;
    size_t total_found = 0;
    for (auto& fut : count_futures) {
        offsets.push_back(total_found);
        total_found += fut.get();
    }
    entries.resize(total_found);

    // Pass 2: Simultaneous SIMD Parse
    bool is_real_or_int = (field == "real" || field == "integer");
    std::vector<std::future<void>> parse_futures;
    for (size_t i = 0; i < starts.size(); ++i)
        parse_futures.push_back(std::async(std::launch::async, hn::ParseChunkTrueSimdHwy,
            starts[i], ends[i], is_real_or_int, entries.data() + offsets[i]));

    for (auto& fut : parse_futures) fut.get();
    return true;
}

bool parallelParserSingleThread(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries) {
    rows = 0; cols = 0; entries.clear();

    ScopedFile file(filename);
    if (!file.isValid()) return false;
    struct stat sb;
    if (fstat(file.fd, &sb) == -1 || sb.st_size == 0) return false;

    ScopedMap mappedFile(file.fd, sb.st_size);
    if (!mappedFile.isValid()) return false;

    char* pCurrent = mappedFile.get();
    const char* const pEnd = pCurrent + sb.st_size;

    // --- Header Parsing ---
    bool banner_found = false;
    std::string field;
    while (pCurrent < pEnd) {
        if (*pCurrent == '%' && pCurrent + 1 < pEnd && *(pCurrent + 1) == '%') {
            char* line_end = hn::FindNextNewlineHwy(pCurrent, pEnd);
            std::string line(pCurrent, line_end - pCurrent);
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            std::stringstream ss(line);
            std::string tok, obj, fmt, sym;
            ss >> tok >> obj >> fmt >> field >> sym;
            if (tok == "%%MatrixMarket" && obj == "matrix" && fmt == "coordinate") {
                std::transform(field.begin(), field.end(), field.begin(), ::tolower);
                banner_found = true;
                pCurrent = line_end;
                break;
            }
        }
        pCurrent = hn::FindNextNewlineHwy(pCurrent, pEnd);
    }
    if (!banner_found) return false;

    // --- Dimension Parsing ---
    int num_entries = 0;
    while (pCurrent < pEnd) {
        pCurrent = hn::SkipWhitespaceHwy(pCurrent, pEnd);
        if (pCurrent >= pEnd) break;
        if (*pCurrent == '%') { pCurrent = hn::FindNextNewlineHwy(pCurrent, pEnd); continue; }

        char* pNext = hn::FindDelimiterHwy(pCurrent, pEnd);
        rows = (int)std::strtol(pCurrent, nullptr, 10);
        pCurrent = hn::SkipWhitespaceHwy(pNext, pEnd);
        pNext = hn::FindDelimiterHwy(pCurrent, pEnd);
        cols = (int)std::strtol(pCurrent, nullptr, 10);
        pCurrent = hn::SkipWhitespaceHwy(pNext, pEnd);
        pNext = hn::FindDelimiterHwy(pCurrent, pEnd);
        num_entries = (int)std::strtol(pCurrent, nullptr, 10);
        pCurrent = hn::FindNextNewlineHwy(pNext, pEnd);
        break;
    }

    std::cout << "Starting single-threaded simultaneous Highway parse." << std::endl;
    entries.resize(num_entries);
    bool is_real_or_int = (field == "real" || field == "integer");

    // Call Kernel on the entire file range
    hn::ParseChunkTrueSimdHwy(pCurrent, const_cast<char*>(pEnd), is_real_or_int, entries.data());

    return true;
}
