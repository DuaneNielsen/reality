#include "mgr.hpp"
#include "replay_metadata.hpp"
#include "types.hpp"
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
    size_t expected_size = sizeof(ReplayMetadata) + sizeof(CompiledLevel);
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
    if (metadata.num_worlds == 0 || metadata.num_worlds > consts::capi::maxWorlds) {
        std::cout << "✗ Invalid num_worlds: " << metadata.num_worlds << "\n";
        ranges_valid = false;
    }
    if (metadata.num_agents_per_world == 0 || metadata.num_agents_per_world > consts::capi::maxAgentsPerWorld) {
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
    auto level_opt = ReplayLoader::loadEmbeddedLevel(info.filepath);
    if (level_opt.has_value()) {
        const auto& level = level_opt.value();
        std::cout << "\nEmbedded Level:\n";
        std::cout << "  Name: " << level.level_name << "\n";
        std::cout << "  Dimensions: " << level.width << "x" << level.height << " grid, Scale: " << level.scale << "\n";
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
    
    // Check file size matches CompiledLevel struct
    size_t expected_size = sizeof(CompiledLevel);
    if (info.file_size == expected_size) {
        std::cout << "✓ Valid file size (" << info.file_size << " bytes)\n";
    } else {
        std::cout << "✗ Invalid file size: " << info.file_size << " bytes (expected " << expected_size << ")\n";
        return false;
    }
    
    // Read level data
    std::ifstream file(info.filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "✗ Cannot open file\n";
        return false;
    }
    
    CompiledLevel level;
    file.read(reinterpret_cast<char*>(&level), sizeof(CompiledLevel));
    if (!file.good()) {
        std::cout << "✗ Failed to read level data\n";
        return false;
    }
    
    // Validate level data ranges
    bool ranges_valid = true;
    
    if (level.width <= 0 || level.width > consts::capi::maxGridSize) {
        std::cout << "✗ Invalid width: " << level.width << "\n";
        ranges_valid = false;
    }
    if (level.height <= 0 || level.height > consts::capi::maxGridSize) {
        std::cout << "✗ Invalid height: " << level.height << "\n";
        ranges_valid = false;
    }
    if (level.scale <= 0.0f || level.scale > consts::capi::maxScale) {
        std::cout << "✗ Invalid scale: " << level.scale << "\n";
        ranges_valid = false;
    }
    if (level.num_tiles < 0 || level.num_tiles > CompiledLevel::MAX_TILES) {
        std::cout << "✗ Invalid num_tiles: " << level.num_tiles << "\n";
        ranges_valid = false;
    }
    if (level.num_spawns <= 0 || level.num_spawns > CompiledLevel::MAX_SPAWNS) {
        std::cout << "✗ Invalid num_spawns: " << level.num_spawns << "\n";
        ranges_valid = false;
    }
    
    if (ranges_valid) {
        std::cout << "✓ Level data within valid ranges\n";
    }
    
    // Validate spawn data
    bool spawns_valid = true;
    for (int i = 0; i < level.num_spawns; i++) {
        float x = level.spawn_x[i];
        float y = level.spawn_y[i];
        if (x < -consts::capi::maxCoordinate || x > consts::capi::maxCoordinate || 
            y < -consts::capi::maxCoordinate || y > consts::capi::maxCoordinate) {
            std::cout << "✗ Invalid spawn " << i << " position: (" << x << ", " << y << ")\n";
            spawns_valid = false;
        }
    }
    
    if (spawns_valid) {
        std::cout << "✓ Spawn data validated\n";
    }
    
    std::cout << "\nLevel Details:\n";
    std::cout << "  Name: " << level.level_name << "\n";
    std::cout << "  Dimensions: " << level.width << "x" << level.height << " grid, Scale: " << level.scale << "\n";
    std::cout << "  Tiles: " << level.num_tiles << ", Max entities: " << level.max_entities << "\n";
    
    // Show spawn information
    for (int i = 0; i < level.num_spawns; i++) {
        float facing_deg = level.spawn_facing[i] * consts::math::radiansToDegrees;
        std::cout << "  Spawn " << i << ": (" << level.spawn_x[i] << ", " << level.spawn_y[i] 
                  << ") facing " << facing_deg << "°\n";
    }
    
    // Count actual tiles
    int actual_tiles = 0;
    for (int i = 0; i < level.num_tiles; i++) {
        if (level.tile_types[i] != 0) {  // Not TILE_EMPTY
            actual_tiles++;
        }
    }
    std::cout << "  Tile data: " << actual_tiles << " valid tiles in bounds\n";
    
    return ranges_valid && spawns_valid;
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