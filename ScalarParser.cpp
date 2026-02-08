// ScalarParser.cpp

#include "pch.h"
#include "ScalarParser.h"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <memory>       // For std::unique_ptr
#include <cstdlib>      // For strtol, strtod
// --- Linux mmap Stuff ---
#include <sys/mman.h>   // mmap, munmap
#include <sys/stat.h>   // fstat
#include <fcntl.h>      // open
#include <unistd.h>     // close

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


bool scalarParser(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries) {
    rows = 0;
    cols = 0;
    entries.clear();

    // 1. Open the file
    ScopedFile file(filename);
    if (!file.isValid()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    // 2. Get file size using 'fstat'
    struct stat sb;
    if (fstat(file.fd, &sb) == -1) {
        std::cerr << "Error: Could not determine file size." << std::endl;
        return false;
    }

    if (sb.st_size == 0) {
        std::cerr << "Error: File is empty." << std::endl;
        return false;
    }

    // 3. Map the file into memory
    ScopedMap mappedFile(file.fd, sb.st_size);
    if (!mappedFile.isValid()) {
        std::cerr << "Error: mmap failed." << std::endl;
        return false;
    }

    // --- Begin parsing directly from the memory buffer ---
    char* pCurrent = mappedFile.get();
    const char* const pEnd = pCurrent + sb.st_size;

    // --- Find and Read the Banner Line ---
    bool banner_found = false;
    std::string field, symmetry;
    while (pCurrent < pEnd) {
        if (*pCurrent == '%') {
            if (pCurrent + 1 < pEnd && *(pCurrent + 1) == '%') {
                const char* line_start = pCurrent;
                const char* line_end = line_start;
                while (line_end < pEnd && *line_end != '\n') line_end++;

                std::string banner_line(line_start, line_end - line_start);
                std::stringstream banner_ss(banner_line);
                std::string banner_tok, object, format;
                banner_ss >> banner_tok >> object >> format >> field >> symmetry;

                if (banner_tok != "%%MatrixMarket" || object != "matrix" || format != "coordinate") {
                    std::cerr << "Error: Unsupported file type. Must be '%%MatrixMarket matrix coordinate'." << std::endl;
                    return false;
                }
                std::transform(field.begin(), field.end(), field.begin(), ::tolower);
                std::transform(symmetry.begin(), symmetry.end(), symmetry.begin(), ::tolower);

                pCurrent = const_cast<char*>(line_end);
                banner_found = true;
                break;
            }
        }
        // Skip to the end of the line
        while (pCurrent < pEnd && *pCurrent != '\n') pCurrent++;
        if (pCurrent < pEnd) pCurrent++; // Move past '\n'
    }

    if (!banner_found) {
        std::cerr << "Error: MatrixMarket banner not found." << std::endl;
        return false;
    }

    // --- Read the Dimensions Line ---
    int num_entries = 0;
    while (pCurrent < pEnd) {
        // Skip any leading whitespace or entire blank lines
        while (pCurrent < pEnd && isspace(static_cast<unsigned char>(*pCurrent))) {
            pCurrent++;
        }
        if (pCurrent >= pEnd) break; // Reached end of file

        if (*pCurrent != '%') {
            char* pNext;
            rows = std::strtol(pCurrent, &pNext, 10);
            pCurrent = pNext;
            cols = std::strtol(pCurrent, &pNext, 10);
            pCurrent = pNext;
            num_entries = std::strtol(pCurrent, &pNext, 10);
            pCurrent = pNext;
            break;
        }
        // Skip comment line
        while (pCurrent < pEnd && *pCurrent != '\n') pCurrent++;
    }

    if (num_entries <= 0) {
        std::cerr << "Error: Invalid number of entries (" << num_entries << ") specified in header." << std::endl;
        return false;
    }

    entries.reserve(num_entries);

    // --- Read the Data Entries ---
    bool is_pattern = (field == "pattern");
    bool is_real_or_int = (field == "real" || field == "integer");
    if (!is_pattern && !is_real_or_int) {
        std::cerr << "Error: Unsupported field type: '" << field << "'." << std::endl;
        return false;
    }

    while (entries.size() < static_cast<size_t>(num_entries) && pCurrent < pEnd) {
        // Skip empty lines or leading whitespace
        while (pCurrent < pEnd && isspace(static_cast<unsigned char>(*pCurrent))) {
            pCurrent++;
        }
        if (pCurrent >= pEnd) break;

        char* pNext;
        int r = std::strtol(pCurrent, &pNext, 10);
        if (pCurrent == pNext) break; // Parse failed
        pCurrent = pNext;

        int c = std::strtol(pCurrent, &pNext, 10);
        if (pCurrent == pNext) break; // Parse failed
        pCurrent = pNext;

        double v = 1.0;
        if (is_real_or_int) {
            v = std::strtod(pCurrent, &pNext);
            pCurrent = pNext;
        }
        entries.push_back({ r, c, v });
    }

    return true;
}