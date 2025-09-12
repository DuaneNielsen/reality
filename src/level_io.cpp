#include "level_io.hpp"
#include <cstdio>
#include <cstring>

namespace madEscape {

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
    
    // Read magic header
    char magic[7] = {0};
    size_t magic_read = fread(magic, sizeof(char), 6, f);
    
    if (magic_read != 6 || strcmp(magic, "LEVELS") != 0) {
        fclose(f);
        return Result::ErrorInvalidFile;
    }
    
    // Read number of levels
    uint32_t num_levels;
    size_t count_read = fread(&num_levels, sizeof(uint32_t), 1, f);
    if (count_read != 1) {
        fclose(f);
        return Result::ErrorFileIO;
    }
    
    // Read all levels
    out_levels.resize(num_levels);
    size_t levels_size = sizeof(CompiledLevel) * num_levels;
    size_t read = fread(out_levels.data(), 1, levels_size, f);
    if (read != levels_size) {
        fclose(f);
        return Result::ErrorFileIO;
    }
    
    fclose(f);
    return Result::Success;
}

} // namespace madEscape