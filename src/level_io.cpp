#include "level_io.hpp"
#include <cstdio>
#include <cstring>

namespace madEscape {

// Core parsing function that works on memory buffer
static Result readCompiledLevelsCore(
    const char* data,
    size_t data_size,
    std::vector<CompiledLevel>& out_levels
) {
    // Check minimum size for header
    if (data_size < 10) { // 6 bytes magic + 4 bytes count
        return Result::ErrorInvalidFile;
    }
    
    // Check magic header
    if (std::memcmp(data, "LEVELS", 6) != 0) {
        return Result::ErrorInvalidFile;
    }
    
    // Read number of levels
    uint32_t num_levels;
    std::memcpy(&num_levels, data + 6, sizeof(uint32_t));
    
    // Check total size
    size_t expected_size = 10 + (sizeof(CompiledLevel) * num_levels);
    if (data_size != expected_size) {
        return Result::ErrorInvalidFile;
    }
    
    // Read all levels
    out_levels.resize(num_levels);
    size_t levels_size = sizeof(CompiledLevel) * num_levels;
    std::memcpy(out_levels.data(), data + 10, levels_size);
    
    return Result::Success;
}

Result writeCompiledLevels(
    const std::string& filepath,
    const std::vector<CompiledLevel>& levels
) {
    if (levels.empty()) {
        return Result::ErrorInvalidParameter;
    }
    
    FILE* f = fopen(filepath.c_str(), "wb");
    if (!f) {
        return Result::ErrorFileNotFound;
    }
    
    // Write unified format header
    const char magic[] = "LEVELS";
    size_t magic_written = fwrite(magic, sizeof(char), 6, f);
    if (magic_written != 6) {
        fclose(f);
        return Result::ErrorFileIO;
    }
    
    // Write count
    uint32_t num_levels = static_cast<uint32_t>(levels.size());
    size_t count_written = fwrite(&num_levels, sizeof(uint32_t), 1, f);
    if (count_written != 1) {
        fclose(f);
        return Result::ErrorFileIO;
    }
    
    // Write all levels
    size_t levels_size = sizeof(CompiledLevel) * num_levels;
    size_t levels_written = fwrite(levels.data(), 1, levels_size, f);
    if (levels_written != levels_size) {
        fclose(f);
        return Result::ErrorFileIO;
    }
    
    fclose(f);
    return Result::Success;
}

Result readCompiledLevels(
    const std::string& filepath,
    std::vector<CompiledLevel>& out_levels
) {
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) {
        return Result::ErrorFileNotFound;
    }
    
    // Get file size
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    if (file_size < 0) {
        fclose(f);
        return Result::ErrorFileIO;
    }
    
    // Read entire file into memory
    std::vector<char> buffer(file_size);
    size_t bytes_read = fread(buffer.data(), 1, file_size, f);
    fclose(f);
    
    if (bytes_read != static_cast<size_t>(file_size)) {
        return Result::ErrorFileIO;
    }
    
    // Use core parsing function
    return readCompiledLevelsCore(buffer.data(), file_size, out_levels);
}

Result readCompiledLevelsFromMemory(
    const char* data,
    size_t data_size,
    std::vector<CompiledLevel>& out_levels
) {
    return readCompiledLevelsCore(data, data_size, out_levels);
}

} // namespace madEscape