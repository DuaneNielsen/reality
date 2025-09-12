#pragma once

#include "types.hpp"
#include <vector>
#include <string>

namespace madEscape {

// Internal C++ level I/O functions (unified format)
// These functions use the same format as the C API but are for internal C++ use

/**
 * Write levels to file using unified format
 * @param filepath Path to output file
 * @param levels Vector of CompiledLevel structs to save
 * @return Result::Success on success, error code otherwise
 */
Result writeCompiledLevels(
    const std::string& filepath,
    const std::vector<CompiledLevel>& levels
);

/**
 * Read levels from file using unified format
 * @param filepath Path to input file
 * @param out_levels Vector to store loaded levels
 * @return Result::Success on success, error code otherwise
 */
Result readCompiledLevels(
    const std::string& filepath,
    std::vector<CompiledLevel>& out_levels
);

} // namespace madEscape