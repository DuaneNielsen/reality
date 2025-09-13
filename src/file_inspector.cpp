#include "mgr.hpp"
#include "types.hpp"
#include "level_io.hpp"
#include "../external/optionparser/optionparser.h"
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace madrona::escape_room;
using namespace madEscape;

struct FileInfo {
    std::string filepath;
    std::string extension;
    size_t file_size;
};

enum OptionIndex { UNKNOWN, HELP, JSON };

const option::Descriptor usage[] = {
    {UNKNOWN, 0, "", "", option::Arg::None, "USAGE: file_inspector [options] <file.(rec|lvl)>\n\n"
                                           "Unified file inspector for Madrona Escape Room files:\n"
                                           "  .rec files: Recording files with metadata, embedded level, and action data\n"
                                           "  .lvl files: Compiled level files with level geometry and spawn data\n\n"
                                           "Options:" },
    {HELP, 0, "h", "help", option::Arg::None, "  --help, -h  \tPrint usage and exit." },
    {JSON, 0, "j", "json", option::Arg::None, "  --json, -j  \tOutput results in JSON format." },
    {UNKNOWN, 0, "", "", option::Arg::None, "\nExamples:\n"
                                           "  file_inspector level.lvl\n"
                                           "  file_inspector --json recording.rec\n"
                                           "  file_inspector -j test.lvl > output.json\n" },
    {0,0,0,0,0,0}
};

struct InspectorConfig {
    bool json_output = false;
    std::string filepath;
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

std::string escapeJsonString(const std::string& str) {
    std::stringstream ss;
    ss << "\"";
    for (char c : str) {
        switch (c) {
            case '"':  ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b"; break;
            case '\f': ss << "\\f"; break;
            case '\n': ss << "\\n"; break;
            case '\r': ss << "\\r"; break;
            case '\t': ss << "\\t"; break;
            default:
                if (c >= 0 && c < 32) {
                    ss << "\\u" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(c);
                } else {
                    ss << c;
                }
                break;
        }
    }
    ss << "\"";
    return ss.str();
}

bool validateReplayFileJson(const FileInfo& info) {
    std::cout << "{\n";
    std::cout << "  \"type\": \"recording\",\n";
    std::cout << "  \"filename\": " << escapeJsonString(std::filesystem::path(info.filepath).filename().string()) << ",\n";
    std::cout << "  \"file_size\": " << info.file_size << ",\n";
    
    // Load metadata using existing manager method
    auto metadata_opt = madEscape::Manager::readReplayMetadata(info.filepath);
    
    if (!metadata_opt.has_value()) {
        std::cout << "  \"valid\": false,\n";
        std::cout << "  \"error\": \"Failed to read replay metadata\"\n";
        std::cout << "}\n";
        return false;
    }
    
    const auto& metadata = metadata_opt.value();
    
    bool valid = true;
    std::stringstream errors;
    
    // Validate magic number
    if (metadata.magic != REPLAY_MAGIC) {
        valid = false;
        errors << "Invalid magic number: 0x" << std::hex << metadata.magic << std::dec;
    }
    
    // Validate version - only v3 supported
    if (metadata.version != 3) {
        valid = false;
        if (errors.tellp() > 0) errors << "; ";
        errors << "Unsupported version: " << metadata.version << " (only v3 supported)";
    }
    
    // Check file structure
    size_t expected_size = sizeof(madEscape::ReplayMetadata) + (metadata.num_worlds * sizeof(CompiledLevel));
    if (metadata.num_steps > 0) {
        expected_size += metadata.num_steps * metadata.num_worlds * metadata.num_agents_per_world * metadata.actions_per_step * sizeof(int32_t);
    }
    
    if (info.file_size < expected_size) {
        valid = false;
        if (errors.tellp() > 0) errors << "; ";
        errors << "File too small: " << info.file_size << " bytes < " << expected_size << " expected";
    }
    
    // Validate metadata ranges
    if (metadata.num_worlds == 0 || metadata.num_worlds > consts::limits::maxWorlds) {
        valid = false;
        if (errors.tellp() > 0) errors << "; ";
        errors << "Invalid num_worlds: " << metadata.num_worlds;
    }
    if (metadata.num_agents_per_world == 0 || metadata.num_agents_per_world > consts::limits::maxAgentsPerWorld) {
        valid = false;
        if (errors.tellp() > 0) errors << "; ";
        errors << "Invalid num_agents_per_world: " << metadata.num_agents_per_world;
    }
    if (metadata.actions_per_step != consts::numActionComponents) {
        valid = false;
        if (errors.tellp() > 0) errors << "; ";
        errors << "Invalid actions_per_step: " << metadata.actions_per_step << " (expected " << consts::numActionComponents << ")";
    }
    
    std::cout << "  \"valid\": " << (valid ? "true" : "false") << ",\n";
    if (!valid) {
        std::cout << "  \"errors\": " << escapeJsonString(errors.str()) << ",\n";
    }
    
    // Output metadata
    std::cout << "  \"metadata\": {\n";
    std::cout << "    \"magic\": \"0x" << std::hex << metadata.magic << std::dec << "\",\n";
    std::cout << "    \"version\": " << metadata.version << ",\n";
    std::cout << "    \"sim_name\": " << escapeJsonString(metadata.sim_name) << ",\n";
    std::cout << "    \"level_name\": " << escapeJsonString(metadata.level_name) << ",\n";
    std::cout << "    \"timestamp\": " << metadata.timestamp << ",\n";
    std::cout << "    \"timestamp_formatted\": " << escapeJsonString(formatTimestamp(metadata.timestamp)) << ",\n";
    std::cout << "    \"num_worlds\": " << metadata.num_worlds << ",\n";
    std::cout << "    \"num_agents_per_world\": " << metadata.num_agents_per_world << ",\n";
    std::cout << "    \"num_steps\": " << metadata.num_steps << ",\n";
    std::cout << "    \"actions_per_step\": " << metadata.actions_per_step << ",\n";
    std::cout << "    \"seed\": " << metadata.seed << "\n";
    std::cout << "  },\n";
    
    // Try to load all embedded levels (v3 format)
    auto levels_opt = madrona::escape_room::ReplayLoader::loadAllEmbeddedLevels(info.filepath);
    if (levels_opt.has_value()) {
        const auto& levels = levels_opt.value();
        std::cout << "  \"embedded_levels\": [\n";
        
        for (size_t i = 0; i < levels.size(); i++) {
            const auto& level = levels[i];
            std::cout << "    {\n";
            std::cout << "      \"index\": " << i << ",\n";
            std::cout << "      \"name\": " << escapeJsonString(level.level_name) << ",\n";
            std::cout << "      \"width\": " << level.width << ",\n";
            std::cout << "      \"height\": " << level.height << ",\n";
            std::cout << "      \"world_scale\": " << level.world_scale << ",\n";
            std::cout << "      \"num_tiles\": " << level.num_tiles << ",\n";
            std::cout << "      \"num_spawns\": " << level.num_spawns << "\n";
            std::cout << "    }" << (i < levels.size() - 1 ? "," : "") << "\n";
        }
        
        std::cout << "  ]\n";
    } else {
        std::cout << "  \"embedded_levels\": null,\n";
        std::cout << "  \"embedded_levels_error\": \"Failed to read embedded levels\"\n";
    }
    
    std::cout << "}\n";
    return valid;
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
    
    // Validate version - only v3 supported
    if (metadata.version == 3) {
        std::cout << "✓ Valid version (" << metadata.version << ")\n";
    } else {
        std::cout << "✗ Unsupported version: " << metadata.version << " (only v3 supported)\n";
        return false;
    }
    
    // Check file structure - calculate expected size for v3 format
    // Format: [ReplayMetadata][CompiledLevel1...N][Actions...]
    size_t expected_size = sizeof(madEscape::ReplayMetadata) + (metadata.num_worlds * sizeof(CompiledLevel));
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
    
    // v3 format displays multi-level information
    std::cout << "  Primary level: " << metadata.level_name << " (legacy field)\n";
    std::cout << "  World levels: " << metadata.num_worlds << "\n";
    
    std::cout << "  Created: " << formatTimestamp(metadata.timestamp) << "\n";
    std::cout << "  Worlds: " << metadata.num_worlds << ", Agents per world: " << metadata.num_agents_per_world << "\n";
    std::cout << "  Steps recorded: " << metadata.num_steps << ", Actions per step: " << metadata.actions_per_step << "\n";
    std::cout << "  Random seed: " << metadata.seed << "\n";
    
    // Try to load all embedded levels (v3 format)
    auto levels_opt = madrona::escape_room::ReplayLoader::loadAllEmbeddedLevels(info.filepath);
    if (levels_opt.has_value()) {
        const auto& levels = levels_opt.value();
        std::cout << "\nEmbedded Levels (" << levels.size() << " total):\n";
        
        for (size_t i = 0; i < levels.size(); i++) {
            const auto& level = levels[i];
            std::cout << "  Level " << i << ": " << level.level_name << "\n";
            std::cout << "    Dimensions: " << level.width << "x" << level.height 
                     << " grid, Scale: " << level.world_scale << "\n";
            std::cout << "    Tiles: " << level.num_tiles << ", Spawns: " << level.num_spawns << "\n";
        }
        
        // Show level names for each world
        std::cout << "\nWorld Levels:\n";
        for (uint32_t worldIdx = 0; worldIdx < std::min(metadata.num_worlds, 10u); worldIdx++) {
            if (worldIdx < levels.size()) {
                std::cout << "  World " << worldIdx << " -> " << levels[worldIdx].level_name << "\n";
            }
        }
        
        if (metadata.num_worlds > 10) {
            std::cout << "  ... (showing first 10 worlds only)\n";
        }
        
        std::cout << "\nFile size: " << info.file_size << " bytes (matches expected)\n";
    } else {
        std::cout << "\n✗ Failed to read embedded levels\n";
        return false;
    }
    
    return ranges_valid;
}

bool validateLevelFileJson(const FileInfo& info) {
    std::cout << "{\n";
    std::cout << "  \"type\": \"level\",\n";
    std::cout << "  \"filename\": " << escapeJsonString(std::filesystem::path(info.filepath).filename().string()) << ",\n";
    std::cout << "  \"file_size\": " << info.file_size << ",\n";
    
    // Try to read using unified format
    std::vector<CompiledLevel> levels;
    Result result = readCompiledLevels(info.filepath, levels);
    
    if (result != Result::Success) {
        std::cout << "  \"valid\": false,\n";
        std::cout << "  \"error\": \"Failed to read level file\",\n";
        std::cout << "  \"error_code\": " << static_cast<int>(result) << "\n";
        std::cout << "}\n";
        return false;
    }
    
    bool all_levels_valid = true;
    std::cout << "  \"valid\": true,\n";
    std::cout << "  \"level_count\": " << levels.size() << ",\n";
    std::cout << "  \"levels\": [\n";
    
    // Validate each level
    for (size_t i = 0; i < levels.size(); i++) {
        const auto& level = levels[i];
        bool level_valid = true;
        std::stringstream level_errors;
        
        // Validate ranges
        if (level.width <= 0 || level.width > consts::limits::maxGridSize) {
            level_valid = false;
            level_errors << "Invalid width: " << level.width;
        }
        if (level.height <= 0 || level.height > consts::limits::maxGridSize) {
            level_valid = false;
            if (level_errors.tellp() > 0) level_errors << "; ";
            level_errors << "Invalid height: " << level.height;
        }
        if (level.world_scale <= 0.0f || level.world_scale > consts::limits::maxScale) {
            level_valid = false;
            if (level_errors.tellp() > 0) level_errors << "; ";
            level_errors << "Invalid world_scale: " << level.world_scale;
        }
        if (level.num_tiles < 0 || level.num_tiles > CompiledLevel::MAX_TILES) {
            level_valid = false;
            if (level_errors.tellp() > 0) level_errors << "; ";
            level_errors << "Invalid num_tiles: " << level.num_tiles;
        }
        if (level.num_spawns <= 0 || level.num_spawns > CompiledLevel::MAX_SPAWNS) {
            level_valid = false;
            if (level_errors.tellp() > 0) level_errors << "; ";
            level_errors << "Invalid num_spawns: " << level.num_spawns;
        }
        
        // Validate spawn data
        for (int j = 0; j < level.num_spawns; j++) {
            float x = level.spawn_x[j];
            float y = level.spawn_y[j];
            if (x < -consts::limits::maxCoordinate || x > consts::limits::maxCoordinate || 
                y < -consts::limits::maxCoordinate || y > consts::limits::maxCoordinate) {
                level_valid = false;
                if (level_errors.tellp() > 0) level_errors << "; ";
                level_errors << "Invalid spawn " << j << " position: (" << x << ", " << y << ")";
            }
        }
        
        if (!level_valid) {
            all_levels_valid = false;
        }
        
        std::cout << "    {\n";
        std::cout << "      \"index\": " << i << ",\n";
        std::cout << "      \"name\": " << escapeJsonString(level.level_name) << ",\n";
        std::cout << "      \"width\": " << level.width << ",\n";
        std::cout << "      \"height\": " << level.height << ",\n";
        std::cout << "      \"world_scale\": " << level.world_scale << ",\n";
        std::cout << "      \"num_tiles\": " << level.num_tiles << ",\n";
        std::cout << "      \"num_spawns\": " << level.num_spawns << ",\n";
        std::cout << "      \"valid\": " << (level_valid ? "true" : "false");
        if (!level_valid) {
            std::cout << ",\n      \"errors\": " << escapeJsonString(level_errors.str());
        }
        std::cout << "\n    }" << (i < levels.size() - 1 ? "," : "") << "\n";
    }
    
    std::cout << "  ],\n";
    std::cout << "  \"all_levels_valid\": " << (all_levels_valid ? "true" : "false") << "\n";
    std::cout << "}\n";
    
    return all_levels_valid;
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
    // Skip program name for option parser
    argc -= (argc > 0); 
    argv += (argc > 0);
    
    option::Stats stats(usage, argc, argv);
    option::Option options[stats.options_max], buffer[stats.buffer_max];
    option::Parser parse(usage, argc, argv, options, buffer);
    
    if (parse.error()) {
        return 1;
    }
    
    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return 0;
    }
    
    if (parse.nonOptionsCount() != 1) {
        std::cerr << "Error: Exactly one file path is required\n";
        option::printUsage(std::cerr, usage);
        return 1;
    }
    
    InspectorConfig config;
    config.json_output = options[JSON];
    config.filepath = parse.nonOption(0);
    
    FileInfo info = analyzeFile(config.filepath);
    
    if (info.file_size == 0) {
        if (config.json_output) {
            std::cout << "{\n";
            std::cout << "  \"valid\": false,\n";
            std::cout << "  \"error\": \"Cannot access file or file is empty\",\n";
            std::cout << "  \"filename\": " << escapeJsonString(std::filesystem::path(config.filepath).filename().string()) << "\n";
            std::cout << "}\n";
        } else {
            std::cerr << "Error: Cannot access file or file is empty: " << config.filepath << "\n";
        }
        return 1;
    }
    
    bool success = false;
    
    if (info.extension == ".rec") {
        success = config.json_output ? validateReplayFileJson(info) : validateReplayFile(info);
    } else if (info.extension == ".lvl") {
        success = config.json_output ? validateLevelFileJson(info) : validateLevelFile(info);
    } else {
        if (config.json_output) {
            std::cout << "{\n";
            std::cout << "  \"valid\": false,\n";
            std::cout << "  \"error\": \"Unsupported file type\",\n";
            std::cout << "  \"extension\": " << escapeJsonString(info.extension) << ",\n";
            std::cout << "  \"supported_extensions\": [\".rec\", \".lvl\"],\n";
            std::cout << "  \"filename\": " << escapeJsonString(std::filesystem::path(config.filepath).filename().string()) << "\n";
            std::cout << "}\n";
        } else {
            std::cerr << "Error: Unsupported file type '" << info.extension << "'\n";
            std::cerr << "Supported extensions: .rec (recording files), .lvl (level files)\n";
        }
        return 1;
    }
    
    if (!config.json_output) {
        if (success) {
            std::cout << "\n✓ File validation completed successfully\n";
        } else {
            std::cout << "\n✗ File validation failed\n";
        }
    }
    
    return success ? 0 : 1;
}