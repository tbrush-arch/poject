#pragma once

// This is the single, authoritative definition for MatrixEntry.
// All other headers will include this file.
struct MatrixEntry {
    int row;
    int col;
    double value;
};