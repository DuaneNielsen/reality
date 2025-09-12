#include "mgr.hpp"
#include "types.hpp"
#include "level_io.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <ctime>
#include <iomanip>

using namespace madrona::escape_room;
using namespace madEscape;

struct FileInfo {
    std::string filepath;
    std::string extension;
    size_t file_size;
};

FileInfo analyzeFile(const std::string& filepath) {
    FileInfo info;
    info.filepath = filepath;
    
    // Get file extension
    size_t dot_pos = filepath.find_last_of('.');
    if (dot_pos != std::string::npos) {
        info.extension = filepath.substr(dot_pos);
    } else {
        info.extension = "";
    }
    
    // Get file size
    std::error_code ec;
    info.file_size = std::filesystem::file_size(filepath, ec);
    if (ec) {
        info.file_size = 0;
    }
    
    return info;
}

std::string formatTimestamp(uint64_t timestamp) {
    if (timestamp == 0) {
        return "Not set";
    }
    
    std::time_t time = static_cast<std::time_t>(timestamp);
    std::stringstream ss;
    ss << std::put_time(std::gmtime(&time), "%Y-%m-%d %H:%M:%S UTC");
    return ss.str();
}

bool validateReplayFile(const FileInfo& info) {
    std::cout << "Recording File: " << std::filesystem::path(info.filepath).filename().string() << "\n";
    
    // Load metadata using existing manager method
    auto metadata_opt = madEscape::Manager::readReplayMetadata(info.filepath);
    
    if (!metadata_opt.has_value()) {
        std::cout << "✗ Failed to read replay metadata\n";
        return false;
    }
    
    const auto& metadata = metadata_opt.value();
    
    // Validate magic number
    if (metadata.magic == REPLAY_MAGIC) {
        std::cout << "✓ Valid magic number (MESR)\n";
    } else {
        std::cout << "✗ Invalid magic number: 0x" << std::hex << metadata.magic << std::dec << "\n";
        return false;
    }
    
    // Validate version
    if (metadata.version == 1 || metadata.version == 2) {
        std::cout << "✓ Valid version (" << metadata.version << ")\n";
    } else {
        std::cout << "✗ Unsupported version: " << metadata.version << "\n";
        return false;
    }
    
    // Check file structure - calculate expected size
    size_t expected_size = sizeof(madEscape::ReplayMetadata) + sizeof(CompiledLevel);
    if (metadata.num_steps > 0) {
        expected_size += metadata.num_steps * metadata.num_worlds * metadata.num_agents_per_world * metadata.actions_per_step * sizeof(int32_t);
    }
    
    if (info.file_size >= expected_size) {
        std::cout << "✓ File structure intact\n";
    } else {
        std::cout << "✗ File too small: " << info.file_size << " bytes < " << expected_size << " expected\n";
        return false;
    }
    
    // Validate metadata ranges
    bool ranges_valid = true;
    if (metadata.num_worlds == 0 || metadata.num_worlds > consts::limits::maxWorlds) {
        std::cout << "✗ Invalid num_worlds: " << metadata.num_worlds << "\n";
        ranges_valid = false;
    }
    if (metadata.num_agents_per_world == 0 || metadata.num_agents_per_world > consts::limits::maxAgentsPerWorld) {
        std::cout << "✗ Invalid num_agents_per_world: " << metadata.num_agents_per_world << "\n";
        ranges_valid = false;
    }
    if (metadata.actions_per_step != consts::numActionComponents) {
        std::cout << "✗ Invalid actions_per_step: " << metadata.actions_per_step << " (expected " << consts::numActionComponents << ")\n";
        ranges_valid = false;
    }
    
    if (ranges_valid) {
        std::cout << "✓ Metadata fields within valid ranges\n";
    }
    
    std::cout << "\nRecording Metadata:\n";
    std::cout << "  Simulation: " << metadata.sim_name << "\n";
    
    // Handle level name field (only available in version 2)
    if (metadata.version >= 2) {
        std::cout << "  Level: " << metadata.level_name << "\n";
    }
    
    std::cout << "  Created: " << formatTimestamp(metadata.timestamp) << "\n";
    std::cout << "  Worlds: " << metadata.num_worlds << ", Agents per world: " << metadata.num_agents_per_world << "\n";
    std::cout << "  Steps recorded: " << metadata.num_steps << ", Actions per step: " << metadata.actions_per_step << "\n";
    std::cout << "  Random seed: " << metadata.seed << "\n";
    
    // Try to load embedded level
    auto level_opt = madrona::escape_room::ReplayLoader::loadEmbeddedLevel(info.filepath);
    if (level_opt.has_value()) {
        const auto& level = level_opt.value();
        std::cout << "\nEmbedded Level:\n";
        std::cout << "  Name: " << level.level_name << "\n";
        std::cout << "  Dimensions: " << level.width << "x" << level.height << " grid, Scale: " << level.world_scale << "\n";
        std::cout << "  Tiles: " << level.num_tiles << ", Spawns: " << level.num_spawns << "\n";
        std::cout << "  File size: " << info.file_size << " bytes (matches expected)\n";
    } else {
        std::cout << "\n✗ Failed to read embedded level\n";
        return false;
    }
    
    return ranges_valid;
}

bool validateLevelFile(const FileInfo& info) {
    std::cout << "Level File: " << std::filesystem::path(info.filepath).filename().string() << "\n";
    
    // Try to read using unified format
    std::vector<CompiledLevel> levels;
    Result result = readCompiledLevels(info.filepath, levels);
    
    if (result != Result::Success) {
        std::cout << "✗ Failed to read level file (error: " << static_cast<int>(result) << ")\n";
        return false;
    }
    
    std::cout << "✓ Valid level file format\n";
    std::cout << "  Contains " << levels.size() << " level(s)\n";
    
    bool all_levels_valid = true;
    
    // Validate each level
    for (size_t i = 0; i < levels.size(); i++) {
        const auto& level = levels[i];
        std::cout << "\nLevel " << (i+1) << "/" << levels.size() << ":\n";
        std::cout << "  Name: " << level.level_name << "\n";
        std::cout << "  Grid: " << level.width << "x" << level.height << "\n";
        std::cout << "  Scale: " << level.world_scale << "\n";
        std::cout << "  Tiles: " << level.num_tiles << "\n";
        std::cout << "  Spawns: " << level.num_spawns << "\n";
        
        // Validate ranges
        bool level_valid = true;
        if (level.width <= 0 || level.width > consts::limits::maxGridSize) {
            std::cout << "  ✗ Invalid width: " << level.width << "\n";
            level_valid = false;
        }
        if (level.height <= 0 || level.height > consts::limits::maxGridSize) {
            std::cout << "  ✗ Invalid height: " << level.height << "\n";
            level_valid = false;
        }
        if (level.world_scale <= 0.0f || level.world_scale > consts::limits::maxScale) {
            std::cout << "  ✗ Invalid world_scale: " << level.world_scale << "\n";
            level_valid = false;
        }
        if (level.num_tiles < 0 || level.num_tiles > CompiledLevel::MAX_TILES) {
            std::cout << "  ✗ Invalid num_tiles: " << level.num_tiles << "\n";
            level_valid = false;
        }
        if (level.num_spawns <= 0 || level.num_spawns > CompiledLevel::MAX_SPAWNS) {
            std::cout << "  ✗ Invalid num_spawns: " << level.num_spawns << "\n";
            level_valid = false;
        }
        
        // Validate spawn data
        for (int j = 0; j < level.num_spawns; j++) {
            float x = level.spawn_x[j];
            float y = level.spawn_y[j];
            if (x < -consts::limits::maxCoordinate || x > consts::limits::maxCoordinate || 
                y < -consts::limits::maxCoordinate || y > consts::limits::maxCoordinate) {
                std::cout << "  ✗ Invalid spawn " << j << " position: (" << x << ", " << y << ")\n";
                level_valid = false;
            }
        }
        
        if (level_valid) {
            std::cout << "  ✓ Level data valid\n";
        } else {
            all_levels_valid = false;
        }
    }
    
    // Show distribution example
    if (levels.size() > 1) {
        std::cout << "\nLevel Distribution Examples:\n";
        std::cout << "  10 worlds: ";
        for (size_t i = 0; i < std::min(10u, static_cast<uint32_t>(levels.size() * 2)); i++) {
            std::cout << (i % levels.size() + 1) << " ";
        }
        std::cout << "...\n";
        
        std::cout << "  100 worlds: each level used " 
                  << (100 / levels.size()) << "-" << (100 / levels.size() + 1) 
                  << " times\n";
    }
    
    return all_levels_valid;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <file.(rec|lvl)>\n";
        std::cerr << "\nUnified file inspector for Madrona Escape Room files:\n";
        std::cerr << "  .rec files: Recording files with metadata, embedded level, and action data\n";
        std::cerr << "  .lvl files: Compiled level files with level geometry and spawn data\n";
        return 1;
    }
    
    std::string filepath = argv[1];
    FileInfo info = analyzeFile(filepath);
    
    if (info.file_size == 0) {
        std::cerr << "Error: Cannot access file or file is empty: " << filepath << "\n";
        return 1;
    }
    
    bool success = false;
    
    if (info.extension == ".rec") {
        success = validateReplayFile(info);
    } else if (info.extension == ".lvl") {
        success = validateLevelFile(info);
    } else {
        std::cerr << "Error: Unsupported file type '" << info.extension << "'\n";
        std::cerr << "Supported extensions: .rec (recording files), .lvl (level files)\n";
        return 1;
    }
    
    if (success) {
        std::cout << "\n✓ File validation completed successfully\n";
        return 0;
    } else {
        std::cout << "\n✗ File validation failed\n";
        return 1;
    }
}