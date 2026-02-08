// ScalarParser.h

#pragma once

#include <string>
#include <vector>
#include "MatrixEntry.h"

/**
 * @brief Parses a file in the MatrixMarket coordinate format.
 *
 * This function parses MatrixMarket files scalarly.
 *
 * @param filename The path to the .mtx file to parse.
 * @param rows Output parameter that will be filled with the number of rows.
 * @param cols Output parameter that will be filled with the number of columns.
 * @param entries Output parameter (a vector) that will be filled with the coordinates and their values.
 * @return True if the file was successfully parsed, false otherwise.
 */
bool scalarParser(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries);