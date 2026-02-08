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
#include <immintrin.h>  // AVX-512 intrinsics
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

// --- SIMD Lookup Table ---
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

// Automatically closes the file descriptor when it goes out of scope
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

// Automatically unmaps memory when it goes out of scope
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

/**
 * @brief Safely loads 64 bytes into an AVX-512 register.
 * Checks against 4KB page boundaries to prevent segfaults.
 */
inline __m512i load_safe_avx512(const char* p, const char* pEnd) {
    if (p >= pEnd) return _mm512_setzero_si512();

    // Calculate bytes remaining in current 4KB page
    const size_t bytes_on_page = 4096 - (reinterpret_cast<uintptr_t>(p) & 0xFFF);

    // Case 1: The 64-byte chunk fits entirely in the current page.
    if (bytes_on_page >= 64) {
        return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
    }
    else {
        // Case 2: Crosses page boundary, but next page is within file bounds
        if (p + 64 <= pEnd) {
            return _mm512_loadu_si512(reinterpret_cast<const void*>(p));
        }
        // Case 3: End of file/buffer. Safe copy.
        alignas(64) char buffer[64] = { 0 };
        size_t remaining = pEnd - p;
        if (remaining > 0) {
            // Cap at 64 to be safe, though logic implies remaining < 64
            if (remaining > 64) remaining = 64;
            std::memcpy(buffer, p, remaining);
        }
        return _mm512_load_si512(reinterpret_cast<const void*>(buffer));
    }
}

/**
 * @brief Scans 64 bytes at a time to find the next delimiter.
 * Uses AVX-512 mask registers (__mmask64).
 */
inline char* find_delimiter_avx512(char* p, const char* pEnd) {
    const __m512i space = _mm512_set1_epi8(' ');
    const __m512i newline = _mm512_set1_epi8('\n');
    const __m512i cr = _mm512_set1_epi8('\r');
    const __m512i tab = _mm512_set1_epi8('\t');

    while (p < pEnd) {
        __m512i chunk = load_safe_avx512(p, pEnd);

        // Comparisons return a 64-bit mask directly
        __mmask64 m1 = _mm512_cmpeq_epi8_mask(chunk, space);
        __mmask64 m2 = _mm512_cmpeq_epi8_mask(chunk, newline);
        __mmask64 m3 = _mm512_cmpeq_epi8_mask(chunk, cr);
        __mmask64 m4 = _mm512_cmpeq_epi8_mask(chunk, tab);

        // Combine results using bitwise OR on the masks
        __mmask64 combined = m1 | m2 | m3 | m4;

        if (combined != 0) {
            // Count Trailing Zeros on 64-bit integer
            return p + _tzcnt_u64(combined);
        }

        p += 64; // Advance by 64 bytes
        if (p > pEnd) return const_cast<char*>(pEnd);
    }
    return const_cast<char*>(pEnd);
}

/**
 * @brief Scans 64 bytes at a time to skip over whitespace.
 */
inline char* skip_whitespace_avx512(char* p, const char* pEnd) {
    // Treats anything <= 32 as whitespace.
    const __m512i limit = _mm512_set1_epi8(32);

    while (p < pEnd) {
        __m512i chunk = load_safe_avx512(p, pEnd);

        // Returns mask where byte > 32 (valid data)
        __mmask64 non_space = _mm512_cmpgt_epi8_mask(chunk, limit);

        if (non_space != 0) {
            return p + _tzcnt_u64(non_space);
        }

        p += 64;
        if (p > pEnd) return const_cast<char*>(pEnd);
    }
    return const_cast<char*>(pEnd);
}

/**
 * @brief: Parses up to 16 digits using AVX-512 VBMI/MADD.
 */
inline uint64_t parse_int_avx512(const char* p, size_t len) {
    // Fast path for small numbers
    if (len < 8) {
        uint64_t val = 0;
        for (size_t i = 0; i < len; ++i) val = val * 10 + (p[i] - '0');
        return val;
    }

    // --- AVX-512 MADD approach for longer integers ---
    // Loads 16 bytes. If len > 16, a loop would be needed, but MatrixMarket ints are usually row/cols < 10^16.

    // 1. Load data and subtract '0' to get integer values
    __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p)); // Load 16 bytes (SSE load is fine here)
    __m512i vec = _mm512_cvtepu8_epi16(_mm256_castsi128_si256(raw)); // Extend 8-bit to 16-bit integers in ZMM

    // Mask out garbage beyond 'len' (if len < 16)
    if (len < 16) {
        uint16_t mask = (1 << len) - 1;
        vec = _mm512_maskz_mov_epi16(mask, vec);
    }

    vec = _mm512_sub_epi16(vec, _mm512_set1_epi16('0'));

    // 2. Multiply-Add horizontal pairs
    uint64_t val = 0;
    size_t i = 0;
    while (i + 8 <= len) {
        uint64_t chunk;
        std::memcpy(&chunk, p + i, 8);
        chunk ^= 0x3030303030303030ULL;
        chunk = (chunk * 10) + (chunk >> 8);
        chunk = (((chunk & 0x00FF00FF00FF00FFULL) * 100) + ((chunk >> 16) & 0x00FF00FF00FF00FFULL));
        chunk = (((chunk & 0x0000FFFF0000FFFFULL) * 10000) + ((chunk >> 32) & 0x0000FFFF0000FFFFULL));
        chunk &= 0xFFFFFFFF; // Isolate the result

        val = val * 100000000 + chunk;
        i += 8;
    }
    for (; i < len; ++i) val = val * 10 + (p[i] - '0');
    return val;
}

/**
 * @brief: Using AVX-512 FMA for combining parts.
 * Significant speedup over standard library.
 */
inline double parse_double_avx512(const char* p, char** endPtr) {
    // 1. Sign
    bool negative = false;
    if (*p == '-') { negative = true; ++p; }

    // 2. Integer Part
    uint64_t int_part = 0;
    while (*p >= '0' && *p <= '9') {
        int_part = int_part * 10 + (*p - '0'); // Scalar is fast enough for sequential digits here
        ++p;
    }

    // 3. Fraction Part
    double frac_part = 0.0;
    int frac_len = 0;
    if (*p == '.') {
        ++p;
        uint64_t frac_int = 0;
        // Parse fraction as integer first
        while (*p >= '0' && *p <= '9') {
            frac_int = frac_int * 10 + (*p - '0');
            ++p;
            frac_len++;
        }

        // Convert fraction integer to double using FMA
        // Precomputed table is used for division
        if (frac_len > 0 && frac_len < 16) {
            frac_part = (double)frac_int / POW10_TABLE[frac_len];
        }
        else if (frac_len >= 16) {
            frac_part = (double)frac_int / std::pow(10.0, frac_len);
        }
    }

    // 4. Combine using AVX-512 logic
    __m128d v_int = _mm_set_sd((double)int_part);
    __m128d v_frac = _mm_set_sd(frac_part);
    __m128d v_res = _mm_add_sd(v_int, v_frac);

    double result = _mm_cvtsd_f64(v_res);

    if (negative) result = -result;
    if (endPtr) *endPtr = const_cast<char*>(p);
    return result;
}

/**
 * @brief Helper to quickly find the next newline character using std::memchr.
 */
inline char* find_next_newline(char* p, const char* pEnd) {
    void* found = std::memchr(p, '\n', pEnd - p);
    return found ? static_cast<char*>(found) + 1 : const_cast<char*>(pEnd);
}

/**
 * @brief Scans a chunk to count how many valid data entries it contains.
 */
size_t count_entries_in_chunk(char* pStart, char* pEnd) {
    size_t count = 0;
    char* pCurrent = pStart;

    // Prefetch initial page
    _mm_prefetch(pCurrent, _MM_HINT_T0);

    while (pCurrent < pEnd) {
        // PREFETCH: Look 4 cache lines ahead (256 bytes) or next page
        if (((uintptr_t)pCurrent & 0xFF) == 0) {
            _mm_prefetch(pCurrent + 256, _MM_HINT_T0);
        }

        pCurrent = skip_whitespace_avx512(pCurrent, pEnd);
        if (pCurrent >= pEnd) break;

        if (*pCurrent == '%') {
            pCurrent = find_next_newline(pCurrent, pEnd);
            continue;
        }

        count++;
        pCurrent = find_next_newline(pCurrent, pEnd);
    }
    return count;
}

/**
 * @brief SIMD Parser: Parses Row, Col, and Value simultaneously.
 */
void parse_chunk_true_simd(char* pStart, char* pEnd, bool is_real_or_int, MatrixEntry* output_ptr) {
    char* p = pStart;
    size_t idx = 0;

    // --- SIMD Constants ---
    __m512i v_zero_char = _mm512_set1_epi8('0');
    __m512i v_mult_pairs = _mm512_set1_epi16(0x010A);
    __m512i v_mult_quads = _mm512_set1_epi32(0x00010064);

    // --- MAIN SIMD LOOP (Process 64-byte chunks) ---
    while (p + 64 <= pEnd) {
        __m512i chunk = _mm512_loadu_si512(p);
        __mmask64 newline_mask = _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8('\n'));

        uint64_t processed_offset = 0;

        while (newline_mask) {
            uint64_t absolute_newline_idx = _tzcnt_u64(newline_mask);
            int line_len = absolute_newline_idx - processed_offset;
            char* line_ptr = p + processed_offset;

            if (*line_ptr != '%') {
                // --- STEP 1: LOCATE DELIMITERS ---
                __mmask64 load_mask = (1ULL << (line_len + 1)) - 1;
                __m512i line_vec = _mm512_maskz_loadu_epi8(load_mask, line_ptr);

                // Find all spaces
                __mmask64 space_mask = _mm512_cmpeq_epi8_mask(line_vec, _mm512_set1_epi8(' '));

                // Find First Space
                int space1 = _tzcnt_u64(space_mask);
                // Find Second Space (mask out the first one)
                int space2 = _tzcnt_u64(space_mask & ~(1ULL << space1));

                // --- STEP 1.5: FIX POINTERS (Whitespace Skipping) ---

                // Skip leading spaces for Row and calculate digit length
                int off_r = 0;
                while (off_r < space1 && line_ptr[off_r] == ' ') off_r++;
                int len_r = space1 - off_r;

                // Skip leading spaces for Col and calculate digit length
                int off_c = space1 + 1;
                while (off_c < space2 && line_ptr[off_c] == ' ') off_c++;
                int len_c = space2 - off_c;

                // Skip leading spaces for Value and calculate digit length
                int off_v = space2 + 1;
                while (off_v < line_len && line_ptr[off_v] == ' ') off_v++;
                int len_v = line_len - off_v;

                // --- STEP 2: GATHER & RIGHT-ALIGN ---

                // Load the 3 chunks directly using SSE loads
                __m128i r_raw = _mm_loadu_si128((const __m128i*)(line_ptr + off_r));
                __m128i c_raw = _mm_loadu_si128((const __m128i*)(line_ptr + off_c));
                __m128i v_raw = _mm_loadu_si128((const __m128i*)(line_ptr + off_v));

                // Build the 512-bit register
                // Then insert the other two lanes.
                __m512i v_batch = _mm512_castsi128_si512(r_raw);    // Lane 0: Row
                v_batch = _mm512_inserti32x4(v_batch, c_raw, 1);    // Lane 1: Col
                v_batch = _mm512_inserti32x4(v_batch, v_raw, 2);    // Lane 2: Value

                // Load Shuffle Masks
                __m512i v_shuffles = _mm512_castsi128_si512(_mm_load_si128((const __m128i*)RIGHT_ALIGN_LUT[len_r]));
                v_shuffles = _mm512_inserti32x4(v_shuffles, _mm_load_si128((const __m128i*)RIGHT_ALIGN_LUT[len_c]), 1);
                v_shuffles = _mm512_inserti32x4(v_shuffles, _mm_load_si128((const __m128i*)RIGHT_ALIGN_LUT[len_v]), 2);

                // Execute Shuffle: Align digits for all 3 numbers simultaneously
                v_batch = _mm512_shuffle_epi8(v_batch, v_shuffles);

                // --- STEP 3: CONVERT & SUM ---
                // Subtract '0' to convert ASCII to int; clamp negatives to zero
                __m512i v_nums = _mm512_sub_epi8(v_batch, v_zero_char);
                v_nums = _mm512_max_epi8(v_nums, _mm512_setzero_si512());

                // Multiply-Add: Pairwise sum (Digit*10 + Digit)
                __m512i v_pairs = _mm512_maddubs_epi16(v_nums, v_mult_pairs);
                // Multiply-Add: Quadwise sum (Pair*100 + Pair)
                __m512i v_quads = _mm512_madd_epi16(v_pairs, v_mult_quads);

                // Store the partial sums back to memory
                alignas(64) int32_t final_parts[16];
                _mm512_store_si512(final_parts, v_quads);

                // Combine partial chunks into final integers
                auto sum_lane = [&](int lane_idx) -> int {
                    int offset = lane_idx * 4;
                    return final_parts[offset] * 1000000000000LL +
                        final_parts[offset + 1] * 100000000 +
                        final_parts[offset + 2] * 10000 +
                        final_parts[offset + 3];
                    };

                // Finalise Row and Col
                int r = sum_lane(0);
                int c = sum_lane(1);

                double v;
                if (!is_real_or_int) {
                    // Fast Path: Use SIMD result for integer values
                    v = (double)sum_lane(2);
                }
                else {
                    // Slow Path: Fallback to strtod for floating point values
                    v = strtod(line_ptr + off_v, nullptr);
                }

                // Write to output struct
                output_ptr[idx].row = r;
                output_ptr[idx].col = c;
                output_ptr[idx].value = v;
                idx++;
            }

            // Advance processed count past this newline
            processed_offset = absolute_newline_idx + 1;
            // Clear bit for current newline to find the next one in the mask
            newline_mask &= ~(1ULL << absolute_newline_idx);
        }

        // Move main pointer forward by total bytes processed
        p += processed_offset;
    }

    // --- CLEANUP LOOP (Scalar Fallback for the tail) ---
    // Handles the last <64 bytes where the SIMD loop stopped.
    while (p < pEnd) {
        // Find next newline or end of buffer
        char* next_nl = p;
        while (next_nl < pEnd && *next_nl != '\n') next_nl++;

        if (next_nl > p && *p != '%') {
            // Null-terminate temporarily for std functions
            char* end_ptr;
            long r = std::strtol(p, &end_ptr, 10);
            long c = std::strtol(end_ptr, &end_ptr, 10);
            double v = std::strtod(end_ptr, nullptr);

            // Simple valid check: if we parsed numbers
            if (end_ptr > p) {
                output_ptr[idx].row = (int)r;
                output_ptr[idx].col = (int)c;
                output_ptr[idx].value = v;
                idx++;
            }
        }
        p = next_nl + 1;
    }
}

bool parallelParser(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries) {
    rows = 0; cols = 0; entries.clear();

    // 1. Open File
    ScopedFile file(filename);
    if (!file.isValid()) {
        std::cerr << "Error: Open file failed." << std::endl;
        return false;
    }

    // 2. Get File Stats (Size)
    struct stat sb;
    if (fstat(file.fd, &sb) == -1) {
        std::cerr << "Error: fstat failed." << std::endl;
        return false;
    }

    // Check for empty file
    if (sb.st_size == 0) return false;

    // 3. Memory Map File
    ScopedMap mappedFile(file.fd, sb.st_size);
    if (!mappedFile.isValid()) {
        std::cerr << "Error: mmap failed." << std::endl;
        return false;
    }

    char* pCurrent = mappedFile.get();
    const char* const pEnd = pCurrent + sb.st_size;

    // Header Parsing
    bool banner_found = false;
    std::string field;
    while (pCurrent < pEnd) {
        if (*pCurrent == '%') {
            if (pCurrent + 1 < pEnd && *(pCurrent + 1) == '%') {
                const char* line_end = find_next_newline(pCurrent, pEnd);
                std::string line(pCurrent, line_end - pCurrent);
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                std::stringstream ss(line);
                std::string tok, obj, fmt, sym;
                ss >> tok >> obj >> fmt >> field >> sym;
                if (tok == "%%MatrixMarket" && obj == "matrix" && fmt == "coordinate") {
                    std::transform(field.begin(), field.end(), field.begin(), ::tolower);
                    banner_found = true;
                    pCurrent = const_cast<char*>(line_end);
                    break;
                }
            }
        }
        pCurrent = find_next_newline(pCurrent, pEnd);
    }
    if (!banner_found) return false;

    int num_entries = 0;
    while (pCurrent < pEnd) {
        pCurrent = skip_whitespace_avx512(pCurrent, pEnd);
        if (pCurrent >= pEnd) break;
        if (*pCurrent == '%') { pCurrent = find_next_newline(pCurrent, pEnd); continue; }

        char* pNext = find_delimiter_avx512(pCurrent, pEnd);
        rows = (int)parse_int_avx512(pCurrent, pNext - pCurrent);
        pCurrent = skip_whitespace_avx512(pNext, pEnd);
        pNext = find_delimiter_avx512(pCurrent, pEnd);
        cols = (int)parse_int_avx512(pCurrent, pNext - pCurrent);
        pCurrent = skip_whitespace_avx512(pNext, pEnd);
        pNext = find_delimiter_avx512(pCurrent, pEnd);
        num_entries = (int)parse_int_avx512(pCurrent, pNext - pCurrent);
        pCurrent = find_next_newline(pNext, pEnd);
        break;
    }

    // Parallel Parse Setup
    unsigned int num_threads = std::thread::hardware_concurrency();

    std::cout << "Starting parallel parse with " << num_threads << " threads." << std::endl;

    long long chunk_size = (std::max)(1LL, (long long)(pEnd - pCurrent) / num_threads);
    std::vector<char*> starts, ends;
    char* chunk_start = pCurrent;

    for (unsigned int i = 0; i < num_threads; ++i) {
        if (chunk_start >= pEnd) break;

        // 1. Determine rough end point
        char* chunk_end = chunk_start + chunk_size;

        // 2. Adjust to valid file boundaries
        if (i == num_threads - 1) chunk_end = const_cast<char*>(pEnd);
        else chunk_end = find_next_newline(chunk_end, pEnd);

        // 3. Safety Check
        if (chunk_end > pEnd) chunk_end = const_cast<char*>(pEnd);

        starts.push_back(chunk_start);
        ends.push_back(chunk_end);
        chunk_start = chunk_end;
    }

    // PASS 1: Count
    // Launch tasks to count how many valid entries are in each chunk
    std::vector<std::future<size_t>> count_futures;
    for (size_t i = 0; i < starts.size(); ++i) {
        count_futures.push_back(std::async(std::launch::async, count_entries_in_chunk, starts[i], ends[i]));
    }

    // Calculate Offsets based on counts
    std::vector<size_t> offsets;
    size_t total_found = 0;
    for (auto& fut : count_futures) {
        size_t c = fut.get();
        offsets.push_back(total_found);
        total_found += c;
    }

    // Resize if header lied
    if (entries.size() != total_found) entries.resize(total_found);

    // PASS 2: Parse
    // Write the data into the pre-calculated slots
    bool is_real_or_int = (field == "real" || field == "integer");
    std::vector<std::future<void>> parse_futures;
    MatrixEntry* global_data_ptr = entries.data();

    for (size_t i = 0; i < starts.size(); ++i) {
        parse_futures.push_back(
            std::async(std::launch::async, parse_chunk_true_simd,
                starts[i], ends[i], is_real_or_int,
                global_data_ptr + offsets[i])
        );
    }

    for (auto& fut : parse_futures) fut.get();

    return true;
}

bool parallelParserSingleThread(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries) {
    rows = 0; cols = 0; entries.clear();

    // 1. Open File
    ScopedFile file(filename);
    if (!file.isValid()) {
        std::cerr << "Error: Open file failed." << std::endl;
        return false;
    }

    // 2. Get File Stats (Size)
    struct stat sb;
    if (fstat(file.fd, &sb) == -1) {
        std::cerr << "Error: fstat failed." << std::endl;
        return false;
    }

    // Check for empty file
    if (sb.st_size == 0) return false;

    // 3. Memory Map File
    ScopedMap mappedFile(file.fd, sb.st_size);
    if (!mappedFile.isValid()) {
        std::cerr << "Error: mmap failed." << std::endl;
        return false;
    }

    char* pCurrent = mappedFile.get();
    const char* const pEnd = pCurrent + sb.st_size;

    // Header Parsing
    bool banner_found = false;
    std::string field;
    while (pCurrent < pEnd) {
        if (*pCurrent == '%') {
            if (pCurrent + 1 < pEnd && *(pCurrent + 1) == '%') {
                const char* line_end = find_next_newline(pCurrent, pEnd);
                std::string line(pCurrent, line_end - pCurrent);
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
                std::stringstream ss(line);
                std::string tok, obj, fmt, sym;
                ss >> tok >> obj >> fmt >> field >> sym;
                if (tok == "%%MatrixMarket" && obj == "matrix" && fmt == "coordinate") {
                    std::transform(field.begin(), field.end(), field.begin(), ::tolower);
                    banner_found = true;
                    pCurrent = const_cast<char*>(line_end);
                    break;
                }
            }
        }
        pCurrent = find_next_newline(pCurrent, pEnd);
    }
    if (!banner_found) return false;

    int num_entries = 0;
    while (pCurrent < pEnd) {
        pCurrent = skip_whitespace_avx512(pCurrent, pEnd);
        if (pCurrent >= pEnd) break;
        if (*pCurrent == '%') { pCurrent = find_next_newline(pCurrent, pEnd); continue; }

        char* pNext = find_delimiter_avx512(pCurrent, pEnd);
        rows = (int)parse_int_avx512(pCurrent, pNext - pCurrent);
        pCurrent = skip_whitespace_avx512(pNext, pEnd);
        pNext = find_delimiter_avx512(pCurrent, pEnd);
        cols = (int)parse_int_avx512(pCurrent, pNext - pCurrent);
        pCurrent = skip_whitespace_avx512(pNext, pEnd);
        pNext = find_delimiter_avx512(pCurrent, pEnd);
        num_entries = (int)parse_int_avx512(pCurrent, pNext - pCurrent);
        pCurrent = find_next_newline(pNext, pEnd);
        break;
    }

    // Single Threaded Parse Setup
    std::cout << "Starting single-threaded parse." << std::endl;

    entries.resize(num_entries); // Resize vector for direct writting
    bool is_real_or_int = (field == "real" || field == "integer");

    // Call the AVX-512 kernel on the entire file range
    // offsets[0] is effectively 0 here, so just write to entries.data()
    parse_chunk_true_simd(pCurrent, const_cast<char*>(pEnd), is_real_or_int, entries.data());

    return true;
}