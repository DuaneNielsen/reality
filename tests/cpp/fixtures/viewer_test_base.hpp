#pragma once

#include <gtest/gtest.h>
#include "test_base.hpp"
#include "test_level_helper.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>

// Structure to represent a trajectory point from the CSV log
struct TrajectoryPoint {
    uint32_t step;
    int32_t world;
    int32_t agent;
    float x, y, z;
    float rotation;  // in degrees
    int32_t compass;  // compass bucket index (0-127)
    float progress;
    float reward;
    int32_t termination_reason;  // 0=episode_steps_reached, 1=goal_achieved, 2=collision_death
    
    bool operator==(const TrajectoryPoint& other) const {
        const float epsilon = 0.001f;
        return step == other.step &&
               world == other.world &&
               agent == other.agent &&
               std::abs(x - other.x) < epsilon &&
               std::abs(y - other.y) < epsilon &&
               std::abs(z - other.z) < epsilon &&
               std::abs(rotation - other.rotation) < epsilon &&
               compass == other.compass &&
               std::abs(progress - other.progress) < epsilon &&
               std::abs(reward - other.reward) < epsilon &&
               termination_reason == other.termination_reason;
    }
};

// Utility class for comparing trajectories
class TrajectoryComparer {
public:
    static std::vector<TrajectoryPoint> parseTrajectoryFile(const std::string& filename) {
        std::vector<TrajectoryPoint> points;
        std::ifstream file(filename);
        if (!file.is_open()) {
            return points;
        }
        
        std::string line;
        while (std::getline(file, line)) {
            TrajectoryPoint point;
            uint32_t remaining = 0;
            int32_t done = 0;

            // Try newest format with state in brackets and action data (simplified step counter)
            char action_str[64];
            if (sscanf(line.c_str(),
                      "Step %u: World %d Agent %d: [pos=(%f,%f,%f) rot=%f° compass=%d progress=%f] %63s reward=%f done=%d term=%d",
                      &point.step, &point.world, &point.agent,
                      &point.x, &point.y, &point.z,
                      &point.rotation, &point.compass, &point.progress,
                      action_str, &point.reward, &done, &point.termination_reason) == 13) {
                points.push_back(point);
            }
            // Try previous format with Episode step and remaining counter
            else if (sscanf(line.c_str(),
                      "Episode step %u (%u remaining): World %d Agent %d: [pos=(%f,%f,%f) rot=%f° compass=%d progress=%f] %63s reward=%f done=%d term=%d",
                      &point.step, &remaining, &point.world, &point.agent,
                      &point.x, &point.y, &point.z,
                      &point.rotation, &point.compass, &point.progress,
                      action_str, &point.reward, &done, &point.termination_reason) == 14) {
                points.push_back(point);
            }
            // Try old format with action data and termination reason
            else if (sscanf(line.c_str(),
                      "Episode step %u (%u remaining): World %d Agent %d: %63s pos=(%f,%f,%f) rot=%f° compass=%d progress=%f reward=%f done=%d term=%d",
                      &point.step, &remaining, &point.world, &point.agent,
                      action_str, &point.x, &point.y, &point.z,
                      &point.rotation, &point.compass, &point.progress, &point.reward, &done, &point.termination_reason) == 14) {
                points.push_back(point);
            }
            // Try format with action data but no termination reason
            else if (sscanf(line.c_str(),
                      "Episode step %u (%u remaining): World %d Agent %d: %63s pos=(%f,%f,%f) rot=%f° compass=%d progress=%f reward=%f done=%d",
                      &point.step, &remaining, &point.world, &point.agent,
                      action_str, &point.x, &point.y, &point.z,
                      &point.rotation, &point.compass, &point.progress, &point.reward, &done) == 13) {
                point.termination_reason = -1;  // Default value for old format
                points.push_back(point);
            }
            // Try legacy format with termination reason (no action)
            else if (sscanf(line.c_str(),
                      "Episode step %u (%u remaining): World %d Agent %d: pos=(%f,%f,%f) rot=%f° compass=%d progress=%f reward=%f done=%d term=%d",
                      &point.step, &remaining, &point.world, &point.agent,
                      &point.x, &point.y, &point.z,
                      &point.rotation, &point.compass, &point.progress, &point.reward, &done, &point.termination_reason) == 13) {
                points.push_back(point);
            }
            // Try format with compass and reward but no termination reason
            else if (sscanf(line.c_str(),
                      "Episode step %u (%u remaining): World %d Agent %d: pos=(%f,%f,%f) rot=%f° compass=%d progress=%f reward=%f done=%d",
                      &point.step, &remaining, &point.world, &point.agent,
                      &point.x, &point.y, &point.z,
                      &point.rotation, &point.compass, &point.progress, &point.reward, &done) == 12) {
                point.termination_reason = -1;  // Default value for old format
                points.push_back(point);
            }
            // Try format with compass but no reward (backwards compatibility)
            else if (sscanf(line.c_str(),
                      "Episode step %u (%u remaining): World %d Agent %d: pos=(%f,%f,%f) rot=%f° compass=%d progress=%f done=%d",
                      &point.step, &remaining, &point.world, &point.agent,
                      &point.x, &point.y, &point.z,
                      &point.rotation, &point.compass, &point.progress, &done) == 11) {
                point.reward = 0.0f;  // Default reward value for old format
                point.termination_reason = -1;  // Default termination reason for old format
                points.push_back(point);
            }
            // Try format without compass
            else if (sscanf(line.c_str(),
                      "Episode step %u (%u remaining): World %d Agent %d: pos=(%f,%f,%f) rot=%f° progress=%f done=%d",
                      &point.step, &remaining, &point.world, &point.agent,
                      &point.x, &point.y, &point.z,
                      &point.rotation, &point.progress, &done) == 10) {
                point.compass = -1;  // Default compass value for old format
                point.reward = 0.0f;  // Default reward value for old format
                point.termination_reason = -1;  // Default termination reason for old format
                points.push_back(point);
            }
            // Then try old format
            else if (sscanf(line.c_str(),
                      "Step %u: World %d Agent %d: pos=(%f,%f,%f) rot=%f° progress=%f",
                      &point.step, &point.world, &point.agent,
                      &point.x, &point.y, &point.z,
                      &point.rotation, &point.progress) == 8) {
                point.compass = -1;  // Default compass value for old format
                point.reward = 0.0f;  // Default reward value for old format
                point.termination_reason = -1;  // Default termination reason for old format
                points.push_back(point);
            }
        }
        
        return points;
    }
    
    static bool compareTrajectories(const std::string& file1, 
                                   const std::string& file2, 
                                   float epsilon = 0.001f) {
        auto traj1 = parseTrajectoryFile(file1);
        auto traj2 = parseTrajectoryFile(file2);
        
        if (traj1.size() != traj2.size()) {
            return false;
        }
        
        for (size_t i = 0; i < traj1.size(); i++) {
            if (!(traj1[i] == traj2[i])) {
                return false;
            }
        }
        
        return true;
    }
};

class LevelComparer {
public:
    static bool compareLevels(const MER_CompiledLevel& level1, 
                             const MER_CompiledLevel& level2) {
        if (level1.width != level2.width || 
            level1.height != level2.height ||
            level1.num_tiles != level2.num_tiles) {
            return false;
        }
        
        // Compare tile data - check all tile arrays including new transform data
        for (int32_t i = 0; i < level1.num_tiles; i++) {
            if (level1.object_ids[i] != level2.object_ids[i] ||
                level1.tile_x[i] != level2.tile_x[i] ||
                level1.tile_y[i] != level2.tile_y[i] ||
                level1.tile_z[i] != level2.tile_z[i] ||
                level1.tile_scale_x[i] != level2.tile_scale_x[i] ||
                level1.tile_scale_y[i] != level2.tile_scale_y[i] ||
                level1.tile_scale_z[i] != level2.tile_scale_z[i] ||
                level1.tile_rotation[i].w != level2.tile_rotation[i].w ||
                level1.tile_rotation[i].x != level2.tile_rotation[i].x ||
                level1.tile_rotation[i].y != level2.tile_rotation[i].y ||
                level1.tile_rotation[i].z != level2.tile_rotation[i].z ||
                level1.tile_persistent[i] != level2.tile_persistent[i] ||
                level1.tile_render_only[i] != level2.tile_render_only[i] ||
                level1.tile_entity_type[i] != level2.tile_entity_type[i] ||
                level1.tile_response_type[i] != level2.tile_response_type[i] ||
                level1.tile_rand_scale_x[i] != level2.tile_rand_scale_x[i] ||
                level1.tile_rand_scale_y[i] != level2.tile_rand_scale_y[i] ||
                level1.tile_rand_scale_z[i] != level2.tile_rand_scale_z[i]) {
                return false;
            }
        }
        
        return true;
    }
    
    static MER_CompiledLevel loadLevelFromFile(const std::string& filename) {
        MER_CompiledLevel level = {};
        std::ifstream file(filename, std::ios::binary);
        
        if (file.is_open()) {
            file.read(reinterpret_cast<char*>(&level), sizeof(MER_CompiledLevel));
        }
        
        return level;
    }
    
    // Helper to get the default level using DefaultLevelProvider
    static MER_CompiledLevel getDefaultLevel() {
        return DefaultLevelProvider::GetDefaultLevelC();
    }
};

// Base test fixture for viewer tests
class ViewerTestBase : public MadronaTestBase {
protected:
    void SetUp() override {
        MadronaTestBase::SetUp();
        // Additional viewer-specific setup if needed
    }
    
    void TearDown() override {
        // Clean up any test files created
        // DISABLED: Keep test files for inspection
        // cleanupTestFiles();
        MadronaTestBase::TearDown();
    }
    
    // Helper to build command line arguments for viewer
    std::vector<const char*> buildViewerArgs(const std::vector<std::string>& args) {
        arg_strings_.clear();
        arg_pointers_.clear();
        
        // Store strings to keep them alive
        arg_strings_ = args;
        
        // Build pointers
        for (const auto& arg : arg_strings_) {
            arg_pointers_.push_back(arg.c_str());
        }
        
        return arg_pointers_;
    }
    
    // Helper to create a simple test level file
    void createTestLevelFile(const std::string& filename, int32_t width = 16, int32_t height = 16) {
        MER_CompiledLevel level = {};
        level.width = width;
        level.height = height;
        level.world_scale = 1.0f;
        level.num_tiles = width * height;
        // Calculate max_entities using same formula as level_compiler.py:
        // entity_count (tiles that need physics bodies) + 6 persistent + 30 buffer
        // For test levels, all tiles are entities (walls/cubes)
        level.max_entities = level.num_tiles + 6 + 30;
        
        // Fill with simple tile pattern
        for (int32_t i = 0; i < level.num_tiles && i < 1024; i++) {
            level.object_ids[i] = (i % 2) + 1;  // Cycle between CUBE (1) and WALL (2)
            level.tile_x[i] = (i % width) * level.world_scale;
            level.tile_y[i] = (i / width) * level.world_scale;
            level.tile_persistent[i] = false;  // Default: non-persistent
            level.tile_render_only[i] = false;  // Default: physics entities
            level.tile_entity_type[i] = (i % 2) + 1;  // Match object_ids (1=Cube, 2=Wall)
            // ResponseType values: 0=Dynamic, 1=Kinematic, 2=Static
            level.tile_response_type[i] = 2;  // Static (walls and cubes are static in tests)
            
            // Initialize transform data with defaults matching old hardcoded logic
            level.tile_z[i] = 0.0f;  // Default ground level
            level.tile_scale_x[i] = level.world_scale;
            level.tile_scale_y[i] = level.world_scale;
            
            // Apply object-specific defaults to match old behavior
            if (level.object_ids[i] == 2) {  // WALL (AssetIDs::WALL)
                level.tile_scale_z[i] = 2.0f;  // wallHeight constant from consts.hpp
            } else if (level.object_ids[i] == 1) {  // CUBE (AssetIDs::CUBE)
                float s = 1.5f * level.world_scale / 2.0f;  // cubeHeightRatio * scale / cubeScaleFactor
                level.tile_scale_x[i] = s;
                level.tile_scale_y[i] = s;
                level.tile_scale_z[i] = s;
                level.tile_z[i] = s;  // Cubes float above ground
            } else {
                level.tile_scale_z[i] = level.world_scale;
            }
            
            // Identity rotation quaternion
            level.tile_rotation[i] = madEscape::Quat::id();  // Identity quaternion
        }
        
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            file.write(reinterpret_cast<const char*>(&level), sizeof(MER_CompiledLevel));
        }
        
        test_files_.push_back(filename);
    }
    
    // Helper to create a test recording file with embedded level
    void createTestRecordingFile(const std::string& filename, 
                                uint32_t num_worlds = 1,
                                uint32_t num_steps = 100,
                                uint32_t seed = 42) {
        // Create a simple recording file structure
        // This would need to match the actual recording format used by Manager
        std::ofstream file(filename, std::ios::binary);
        if (file.is_open()) {
            // Write metadata
            file.write(reinterpret_cast<const char*>(&num_worlds), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&num_steps), sizeof(uint32_t));
            file.write(reinterpret_cast<const char*>(&seed), sizeof(uint32_t));
            
            // Write embedded level
            MER_CompiledLevel level = {};
            level.width = 16;
            level.height = 16;
            level.world_scale = 1.0f;
            level.num_tiles = 256;
            // Calculate max_entities using same formula as level_compiler.py
            level.max_entities = level.num_tiles + 6 + 30;
            // Initialize tile data
            for (int32_t i = 0; i < level.num_tiles && i < 1024; i++) {
                level.object_ids[i] = 0;
                level.tile_x[i] = (i % 16) * level.world_scale;
                level.tile_y[i] = (i / 16) * level.world_scale;
                level.tile_persistent[i] = false;  // Default: non-persistent
                level.tile_render_only[i] = false;  // Default: physics entities
                level.tile_entity_type[i] = 0;  // Default EntityType::None
                level.tile_response_type[i] = 2;  // Static (default for test entities)
            }
            file.write(reinterpret_cast<const char*>(&level), sizeof(MER_CompiledLevel));
            
            // Write some dummy action data
            for (uint32_t step = 0; step < num_steps; step++) {
                for (uint32_t world = 0; world < num_worlds; world++) {
                    int32_t actions[3] = {0, 0, 2}; // move_amount, move_angle, rotate
                    file.write(reinterpret_cast<const char*>(actions), sizeof(actions));
                }
            }
        }
        
        test_files_.push_back(filename);
    }
    
    // Helper to create a test trajectory file
    void createTestTrajectoryFile(const std::string& filename,
                                  uint32_t num_steps = 10,
                                  int32_t world = 0,
                                  int32_t agent = 0) {
        std::ofstream file(filename);
        if (file.is_open()) {
            for (uint32_t step = 0; step < num_steps; step++) {
                file << "Step " << std::setw(4) << step 
                     << ": World " << world << " Agent " << agent 
                     << ": pos=(" << std::fixed << std::setprecision(2) 
                     << step * 1.0f << "," << 0.0f << "," << 0.0f << ") "
                     << "rot=" << std::fixed << std::setprecision(1) << 0.0f << "° "
                     << "progress=" << std::fixed << std::setprecision(2) << step * 0.1f << "\n";
            }
        }
        
        test_files_.push_back(filename);
    }
    
    // Check if a file exists
    bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
    
    // Get file size in bytes
    size_t getFileSize(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return 0;
        }
        return file.tellg();
    }
    
protected:
    // Clean up test files
    void cleanupTestFiles() {
        for (const auto& file : test_files_) {
            std::remove(file.c_str());
        }
        test_files_.clear();
    }
    
private:
    std::vector<std::string> arg_strings_;
    std::vector<const char*> arg_pointers_;
    std::vector<std::string> test_files_;
};

// Mock window manager for headless testing
class MockWindowManager {
public:
    MockWindowManager() = default;
    
    // Stub out window creation
    void* makeWindow(const std::string& title, int width, int height) {
        return reinterpret_cast<void*>(1); // Non-null dummy pointer
    }
    
    // Provide fake GPU handle
    void* initGPU(uint32_t gpu_id, void* window) {
        return reinterpret_cast<void*>(1); // Non-null dummy pointer
    }
    
    // Simulate keyboard input
    void simulateKeyPress(int key) {
        last_key_pressed_ = key;
    }
    
    int getLastKeyPressed() const {
        return last_key_pressed_;
    }
    
private:
    int last_key_pressed_ = -1;
};