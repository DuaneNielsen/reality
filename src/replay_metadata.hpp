#pragma once

#include <cstdint>
#include <cstring>
#include <optional>
#include <fstream>
#include <string>
#include "types.hpp"  // For CompiledLevel

namespace madrona {
namespace escape_room {

// Magic number to identify replay files
constexpr uint32_t REPLAY_MAGIC = 0x4D455352;  // "MESR" (Madrona Escape Room)

// Version of the replay format
constexpr uint32_t REPLAY_VERSION = 1;

// Maximum length for sim name string
constexpr uint32_t MAX_SIM_NAME_LENGTH = 64;

struct ReplayMetadata {
    uint32_t magic;                         // Magic number for file identification
    uint32_t version;                       // Format version
    char sim_name[MAX_SIM_NAME_LENGTH];     // Name of the simulation
    uint32_t num_worlds;                    // Number of worlds recorded
    uint32_t num_agents_per_world;          // Number of agents per world
    uint32_t num_steps;                     // Total number of steps recorded
    uint32_t actions_per_step;              // Number of action components (3: move_amount, move_angle, rotate)
    uint64_t timestamp;                     // Unix timestamp when recording started
    uint32_t seed;                          // Random seed used for simulation
    uint32_t reserved[8];                   // Reserved for future use
    
    static ReplayMetadata createDefault() {
        ReplayMetadata meta;
        meta.magic = REPLAY_MAGIC;
        meta.version = REPLAY_VERSION;
        std::strcpy(meta.sim_name, "madrona_escape_room");
        meta.num_worlds = 1;
        meta.num_agents_per_world = 1;
        meta.num_steps = 0;
        meta.actions_per_step = 3;
        meta.timestamp = 0;
        meta.seed = 5;
        std::memset(meta.reserved, 0, sizeof(meta.reserved));
        return meta;
    }
    
    bool isValid() const {
        return magic == REPLAY_MAGIC && version == REPLAY_VERSION;
    }
};

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