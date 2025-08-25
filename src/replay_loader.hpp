#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <fstream>
#include <string>
#include "types.hpp"  // For CompiledLevel
#include "consts.hpp"  // For file format constants

namespace madrona {
namespace escape_room {

// Use ReplayMetadata from types.hpp
using madEscape::ReplayMetadata;
using madEscape::REPLAY_MAGIC;
using madEscape::REPLAY_VERSION;
using madEscape::MAX_SIM_NAME_LENGTH;

// Utility functions for loading replay data
struct ReplayLoader {
    // Load just the metadata from a replay file
    static std::optional<ReplayMetadata> loadMetadata(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return std::nullopt;
        }
        
        ReplayMetadata metadata;
        file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
        
        if (!file.good() || !metadata.isValid()) {
            return std::nullopt;
        }
        
        return metadata;
    }
    
    // Load the embedded level from a replay file
    // Replay format: [ReplayMetadata][CompiledLevel][Actions...]
    static std::optional<madEscape::CompiledLevel> loadEmbeddedLevel(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return std::nullopt;
        }
        
        // Skip metadata
        ReplayMetadata metadata;
        file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
        
        if (!file.good() || !metadata.isValid()) {
            return std::nullopt;
        }
        
        // Read embedded level
        madEscape::CompiledLevel level;
        file.read(reinterpret_cast<char*>(&level), sizeof(madEscape::CompiledLevel));
        
        if (!file.good()) {
            return std::nullopt;
        }
        
        return level;
    }
};

}
}