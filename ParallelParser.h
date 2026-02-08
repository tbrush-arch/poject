// ParallelParser.h

#pragma once

#include <vector>
#include <string>
#include "ScalarParser.h"

bool parallelParser(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries);
bool parallelParserSingleThread(const std::string& filename, int& rows, int& cols, std::vector<MatrixEntry>& entries);