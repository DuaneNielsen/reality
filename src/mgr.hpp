#pragma once

#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/heap_array.hpp>

#include <madrona/render/render_mgr.hpp>
#include "types.hpp"  // For CompiledLevel

namespace madEscape {

    // Replay metadata constants and structure (host-only with STL usage)
    static constexpr uint32_t REPLAY_MAGIC = 0x4D455352; // "MESR" in hex
    static constexpr uint32_t REPLAY_VERSION = 4;  // Version 4 adds checksum support
    static constexpr uint32_t MAX_SIM_NAME_LENGTH = 64;

    // [GAME_SPECIFIC] Host-only replay metadata structure with STL usage
    struct ReplayMetadata {
        uint32_t magic;                         // Magic number for file identification
        uint32_t version;                       // Format version
        char sim_name[MAX_SIM_NAME_LENGTH];     // Name of the simulation
        char level_name[MAX_SIM_NAME_LENGTH];   // Name of the level being played (kept for compatibility but deprecated in v3)
        uint32_t num_worlds;                    // Number of worlds recorded
        uint32_t num_agents_per_world;          // Number of agents per world
        uint32_t num_steps;                     // Total number of steps recorded
        uint32_t actions_per_step;              // Number of action components (3: move_amount, move_angle, rotate)
        uint64_t timestamp;                     // Unix timestamp when recording started
        uint32_t seed;                          // Random seed used for simulation
        uint32_t checksum_version;              // Version of checksum algorithm (1 for initial implementation)
        uint32_t enable_checksums;              // Flag to enable/disable checksum recording (0 or 1)
        uint32_t num_episode_checksums;         // Total number of episode checksums stored
        uint32_t reserved[consts::fileFormat::replayMagicLength - 3];  // Reserved for future use
        
        static ReplayMetadata createDefault() {
            ReplayMetadata meta;
            meta.magic = REPLAY_MAGIC;
            meta.version = REPLAY_VERSION;
            std::strncpy(meta.sim_name, "madrona_escape_room", sizeof(meta.sim_name) - 1);
            meta.sim_name[sizeof(meta.sim_name) - 1] = '\0';
            std::strncpy(meta.level_name, "unknown_level", sizeof(meta.level_name) - 1);
            meta.level_name[sizeof(meta.level_name) - 1] = '\0';
            meta.num_worlds = 1;
            meta.num_agents_per_world = 1;
            meta.num_steps = 0;
            meta.actions_per_step = consts::numActionComponents;
            meta.timestamp = 0;
            meta.seed = consts::fileFormat::defaultSeed;
            meta.checksum_version = 1;  // Version 1 of checksum algorithm
            meta.enable_checksums = 1;  // Enable checksums by default
            meta.num_episode_checksums = 0;
            std::memset(meta.reserved, 0, sizeof(meta.reserved));
            return meta;
        }
        
        bool isValid() const {
            return magic == REPLAY_MAGIC && (version == 3 || version == 4);
        }
    };

    // [GAME_SPECIFIC] Episode checksum structure for validation
    struct EpisodeChecksum {
        uint32_t world_idx;         // World index that completed episode
        uint32_t episode_num;       // Episode number (increments per world)
        uint32_t step_num;          // Step number when episode ended
        uint64_t hash;              // Computed hash of episode state

        // Hash algorithm version 1: Combines agent positions, target positions, and level data
        static uint64_t computeHash(uint32_t world_idx, uint32_t step_num,
                                  const float* agent_positions, size_t num_agents,
                                  const float* target_positions, size_t num_targets,
                                  uint32_t level_width, uint32_t level_height);
    };

}

// Replay utility functions (moved from replay_loader.hpp to avoid circular dependencies)
namespace madrona {
namespace escape_room {

// Utility functions for loading replay data
struct ReplayLoader {
    // Load just the metadata from a replay file
    static std::optional<madEscape::ReplayMetadata> loadMetadata(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return std::nullopt;
        }
        
        madEscape::ReplayMetadata metadata;
        file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
        
        if (!file.good() || !metadata.isValid()) {
            return std::nullopt;
        }
        
        return metadata;
    }
    
    // Load the embedded levels from a replay file
    // Replay format v3: [ReplayMetadata][NumUniqueLevels][CompiledLevel1...N][Actions...]
    // Load the first embedded level from a replay file (for backward compatibility)
    static std::optional<madEscape::CompiledLevel> loadEmbeddedLevel(const std::string& filepath) {
        auto levels = loadAllEmbeddedLevels(filepath);
        if (!levels.has_value() || levels->empty()) {
            return std::nullopt;
        }
        return levels->front();
    }
    
    // Load all embedded levels from a replay file (v3 format)
    static std::optional<std::vector<madEscape::CompiledLevel>> loadAllEmbeddedLevels(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            return std::nullopt;
        }
        
        // Read metadata
        madEscape::ReplayMetadata metadata;
        file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
        
        if (!file.good() || !metadata.isValid()) {
            return std::nullopt;
        }
        
        // Read all world levels
        std::vector<madEscape::CompiledLevel> levels;
        levels.reserve(metadata.num_worlds);
        
        for (uint32_t i = 0; i < metadata.num_worlds; i++) {
            madEscape::CompiledLevel level;
            file.read(reinterpret_cast<char*>(&level), sizeof(madEscape::CompiledLevel));
            
            if (!file.good()) {
                return std::nullopt;
            }
            
            levels.push_back(level);
        }
        
        return levels;
    }
};

}
}

namespace madEscape {

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t randSeed; // Seed for random world gen
        bool autoReset; // Immediately generate new world on episode end
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = consts::display::defaultBatchRenderSize;
        uint32_t batchRenderViewHeight = consts::display::defaultBatchRenderSize;
        float customVerticalFov = 0.0f; // Custom vertical FOV in degrees (0 = use default)
        int32_t renderMode = 0; // Render mode: 0=RGBD, 1=Depth (default: 0=RGBD)
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
        bool enableTrajectoryTracking = false; // Print agent trajectories to stdout
        std::vector<std::optional<CompiledLevel>> perWorldCompiledLevels;  // Per-world compiled levels
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    madrona::py::Tensor resetTensor() const;
    madrona::py::Tensor actionTensor() const;
    madrona::py::Tensor rewardTensor() const;
    madrona::py::Tensor doneTensor() const;
    madrona::py::Tensor terminationReasonTensor() const;
    madrona::py::Tensor selfObservationTensor() const;
    madrona::py::Tensor compassTensor() const;
    madrona::py::Tensor lidarTensor() const;
    madrona::py::Tensor stepsTakenTensor() const;
    madrona::py::Tensor progressTensor() const;
    madrona::py::Tensor targetPositionTensor() const;
    madrona::py::Tensor rgbTensor() const;
    madrona::py::Tensor depthTensor() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    void triggerReset(int32_t world_idx);
    void setAction(int32_t world_idx,
                   int32_t move_amount,
                   int32_t move_angle,
                   int32_t rotate);
    void toggleLidarVisualization(uint32_t world_idx = 0);
    void toggleLidarVisualizationGlobal(); // Toggle lidar for all worlds
    
    // Trajectory logging for debugging
    void enableTrajectoryLogging(int32_t world_idx, int32_t agent_idx, std::optional<const char*> filename = std::nullopt);
    void disableTrajectoryLogging();
    void logCurrentTrajectoryState();
    
    // Get agent position for camera tracking
    madrona::math::Vector3 getAgentPosition(int32_t world_idx, int32_t agent_idx = 0) const;
    
    // Get compiled level for camera positioning
    const CompiledLevel* getCompiledLevel(int32_t world_idx = 0) const;

    madrona::render::RenderManager & getRenderManager();

    // Replay functionality
    struct ReplayData {
        madEscape::ReplayMetadata metadata;
        madrona::HeapArray<int32_t> actions;
    };
    
    // Recording functionality
    Result startRecording(const std::string& filepath, bool enable_checksums = true);
    void stopRecording();
    bool isRecording() const;
    void recordActions(const std::vector<int32_t>& frame_actions);
    void captureEpisodeChecksums();  // Capture checksums when episodes end
    void validateEpisodeChecksums(); // Validate checksums during replay

    // Checksum control functionality
    
    // Replay functionality
    bool loadReplay(const std::string& filepath);
    bool hasReplay() const;
    bool replayStep();
    uint32_t getCurrentReplayStep() const;
    uint32_t getTotalReplaySteps() const;
    const ReplayData* getReplayData() const;
    
    // Static method to read replay metadata without creating a Manager
    static std::optional<madEscape::ReplayMetadata> readReplayMetadata(const std::string& filepath);
    
    // Static method to read embedded level from replay file
    static std::optional<CompiledLevel> readEmbeddedLevel(const std::string& filepath);
    
    // Static factory method to create Manager from replay file
    static std::unique_ptr<Manager> fromReplay(
        const std::string& filepath,
        madrona::ExecMode execMode,
        int gpuID,
        bool enableBatchRenderer = false,
        madrona::render::APIBackend *extRenderAPI = nullptr,
        madrona::render::GPUDevice *extRenderDev = nullptr);

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
