// TestSuite.cpp

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "ScalarParser.h"
#include "ParallelParser.h"

int main() {
    // --- Prompt user for the filename ---
    std::string filename;
    std::cout << "Please enter the MatrixMarket filename: ";
    std::getline(std::cin, filename);
    std::string folderPath = "../MatrixMarket_Files/";
    std::string fullPath = folderPath + filename;

    std::cout << "\n--- Attempting to parse file: " << filename << " ---" << std::endl;

    // --- SETUP & EXECUTE ---
    int rows = 0;
    int cols = 0;
    std::vector<MatrixEntry> entries;

    auto start_timeS = std::chrono::steady_clock::now();
    bool successS = scalarParser(fullPath, rows, cols, entries);
    auto end_timeS = std::chrono::steady_clock::now();

    auto start_timePS = std::chrono::steady_clock::now();
    bool successPS = parallelParserSingleThread(fullPath, rows, cols, entries);
    auto end_timePS = std::chrono::steady_clock::now();

    auto start_timePM = std::chrono::steady_clock::now();
    bool successP = parallelParser(fullPath, rows, cols, entries);
    auto end_timePM = std::chrono::steady_clock::now();


    std::chrono::duration<double, std::milli> duration_msS = end_timeS - start_timeS;
    std::chrono::duration<double, std::milli> duration_msPS = end_timePS - start_timePS;
    std::chrono::duration<double, std::milli> duration_msPM = end_timePM - start_timePM;

    // --- DISPLAY RESULTS ---
    if (successP) {
        std::cout << "\nFile parsed successfully!" << std::endl;
        std::cout << "---------------------------------------" << std::endl;
        std::cout << "Matrix Dimensions: " << rows << " x " << cols << std::endl;
        std::cout << "Number of Entries Found: " << entries.size() << std::endl;
        std::cout << "---------------------------------------" << std::endl;

        if (!entries.empty()) {
            // Display a sample of the first 5 and last 5 entries
            size_t num_to_show = 10;
            std::cout << "Showing a sample of the entries:" << std::endl;

            for (size_t i = 0; i < entries.size() && i < (num_to_show / 2); ++i) {
                const auto& e = entries[i];
                std::cout << "  Entry " << i << ": (row=" << e.row << ", col=" << e.col << ", val=" << e.value << ")" << std::endl;
            }

            if (entries.size() > num_to_show) {
                std::cout << "  ..." << std::endl;
                size_t start_idx = entries.size() - (num_to_show / 2);
                for (size_t i = start_idx; i < entries.size(); ++i) {
                    const auto& e = entries[i];
                    std::cout << "  Entry " << i << ": (row=" << e.row << ", col=" << e.col << ", val=" << e.value << ")" << std::endl;
                }
            }
            else if (entries.size() > (num_to_show / 2)) {
                for (size_t i = (num_to_show / 2); i < entries.size(); ++i) {
                    const auto& e = entries[i];
                    std::cout << "  Entry " << i << ": (row=" << e.row << ", col=" << e.col << ", val=" << e.value << ")" << std::endl;
                }
            }
        }

        std::cout << "---------------------------------------" << std::endl;
        std::cout << "Scalar Parsing took: " << duration_msS.count() << " ms" << std::endl;
        std::cout << "Parallel Parsing Single Thread took: " << duration_msPS.count() << " ms" << std::endl;
        std::cout << "Parallel Parsing Multi-Thread took: " << duration_msPM.count() << " ms" << std::endl;

        return 0; // Success
    }
    else {
        std::cerr << "\nFATAL ERROR: The scalarParse function returned false." << std::endl;
        return 1; // Failure
    }
}