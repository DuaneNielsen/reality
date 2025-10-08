#include <gtest/gtest.h>
#include "types.hpp"
#include "mgr.hpp"
#include "test_level_helper.hpp"
#include "level_io.hpp"
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <string>
#include <sstream>

using namespace madEscape;
using namespace madrona::escape_room;

// Test fixture for file inspector functionality - simplified to avoid linker conflicts
class FileInspectorTest : public ::testing::Test {
protected:
    std::string testDataDir;
    std::string testJsonLevel;
    std::string testTextLevel;
    std::string testLevelFile;
    std::string testRecordingFile;
    
    void SetUp() override {
        
        // Create test data directory
        testDataDir = "tests/cpp/test_data/file_inspector";
        std::filesystem::create_directories(testDataDir);
        
        // Define test file paths
        testJsonLevel = testDataDir + "/test_level.json";
        testTextLevel = testDataDir + "/test_level.txt";
        testLevelFile = testDataDir + "/test_level.lvl";
        testRecordingFile = testDataDir + "/test_recording.rec";
        
        createTestJsonLevel();
        createTestTextLevel();
        createTestLevelFile();
        createTestRecordingFile();
    }
    
    void TearDown() override {
        // Clean up test files
        std::filesystem::remove_all(testDataDir);
    }
    
    void createTestJsonLevel() {
        std::ofstream file(testJsonLevel);
        file << R"({
    "name": "test_json_level",
    "ascii": "########\n#S.....#\n#......#\n#......#\n########",
    "scale": 2.5,
    "agent_facing": [0.0],
    "_comment": "Test JSON level for file inspector tests"
})";
        file.close();
    }
    
    void createTestTextLevel() {
        std::ofstream file(testTextLevel);
        file << R"(########
#S.....#
#......#
#......#
########)";
        file.close();
    }
    
    void createTestLevelFile() {
        // Start with the default level as a base
        CompiledLevel level = DefaultLevelProvider::GetDefaultLevel();
        
        // Overwrite with test-specific values
        std::strcpy(level.level_name, "test_compiled_level");
        level.width = 8;
        level.height = 5;
        level.world_scale = 2.5f;
        // level.done_on_collide = false;  // This field was renamed/moved
        level.num_tiles = 22;
        level.max_entities = 58;
        
        // Set proper number of spawns and update spawn data for test
        level.num_spawns = 1;
        level.spawn_x[0] = -6.25f;
        level.spawn_y[0] = 2.5f;
        level.spawn_facing[0] = 0.0f; // 0 degrees
        
        // Write to file using unified format
        std::vector<CompiledLevel> levels = {level};
        Result result = writeCompiledLevels(testLevelFile, levels);
        // Note: Cannot use exceptions in this project - test will fail if file write fails
    }
    
    void createTestRecordingFile() {
        // Create a test level with specific properties
        CompiledLevel level = DefaultLevelProvider::GetDefaultLevel();
        std::strcpy(level.level_name, "embedded_test_level");
        level.width = 5;
        level.height = 4;
        level.world_scale = 1.5f;
        level.num_spawns = 1;
        level.spawn_x[0] = 0.0f;
        level.spawn_y[0] = 0.0f;
        level.spawn_facing[0] = 1.57f; // 90 degrees
        
        // Use Manager API to create proper recording file
        {
            Manager::Config cfg;
            cfg.execMode = madrona::ExecMode::CPU;
            cfg.gpuID = -1;
            cfg.numWorlds = 1;
            cfg.randSeed = 42;
            cfg.autoReset = true;
            cfg.enableBatchRenderer = false;
            cfg.perWorldCompiledLevels = {level};
            
            Manager mgr(cfg);
            auto result = mgr.startRecording(testRecordingFile);
            if (result != Result::Success) {
                // Test will fail if recording setup fails
                return;
            }
            
            // Step through 3 simulation steps
            for (int i = 0; i < 3; i++) {
                mgr.step();
            }
            mgr.stopRecording();
        }
    }
    
    // Helper to run file inspector and capture output
    std::string runFileInspector(const std::string& filepath) {
        std::string command = "./build/file_inspector " + filepath + " 2>&1";
        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) return "";
        
        std::stringstream result;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result << buffer;
        }
        
        int exitCode = pclose(pipe);
        
        // Add exit code info to result for validation
        result << "\nEXIT_CODE:" << WEXITSTATUS(exitCode);
        
        return result.str();
    }
    
    bool fileExists(const std::string& path) {
        return std::filesystem::exists(path);
    }
};

// Test file inspector with default level from build directory
TEST_F(FileInspectorTest, InspectorHandlesDefaultLevel) {
    std::string defaultLevelPath = "./build/default_level.lvl";
    
    // Check if default_level.lvl exists in build directory
    ASSERT_TRUE(fileExists(defaultLevelPath)) << "default_level.lvl not found in build directory";
    
    // Run file inspector on default level
    std::string output = runFileInspector(defaultLevelPath);
    
    // Check for expected output (updated for 2-level unified format)
    EXPECT_TRUE(output.find("Level File:") != std::string::npos);
    EXPECT_TRUE(output.find("default_level") != std::string::npos);
    EXPECT_TRUE(output.find("✓ Valid level file format") != std::string::npos);
    EXPECT_TRUE(output.find("Contains 2 level(s)") != std::string::npos);
    EXPECT_TRUE(output.find("Name: default_full_obstacles") != std::string::npos);
    EXPECT_TRUE(output.find("Name: default_cubes_only") != std::string::npos);
    EXPECT_TRUE(output.find("Grid: 16x16") != std::string::npos);
    EXPECT_TRUE(output.find("Scale: 1") != std::string::npos);
    EXPECT_TRUE(output.find("Tiles: 74") != std::string::npos);
    EXPECT_TRUE(output.find("Tiles: 66") != std::string::npos);
    EXPECT_TRUE(output.find("Spawns: 1") != std::string::npos);
    EXPECT_TRUE(output.find("✓ Level data valid") != std::string::npos);
    EXPECT_TRUE(output.find("✓ File validation completed successfully") != std::string::npos);
    EXPECT_TRUE(output.find("EXIT_CODE:0") != std::string::npos);
}

// Test file inspector with pre-created .lvl file
TEST_F(FileInspectorTest, InspectorHandlesLevelFile) {
    ASSERT_TRUE(fileExists(testLevelFile));
    
    std::string output = runFileInspector(testLevelFile);
    
    // Check for expected output (updated for unified format)
    EXPECT_TRUE(output.find("Level File:") != std::string::npos);
    EXPECT_TRUE(output.find("test_level") != std::string::npos);
    EXPECT_TRUE(output.find("✓ Valid level file format") != std::string::npos);
    EXPECT_TRUE(output.find("Contains 1 level(s)") != std::string::npos);
    EXPECT_TRUE(output.find("Name: test_compiled_level") != std::string::npos);
    EXPECT_TRUE(output.find("Grid: 8x5") != std::string::npos);
    EXPECT_TRUE(output.find("Scale: 2.5") != std::string::npos);
    EXPECT_TRUE(output.find("Tiles: 22") != std::string::npos);
    EXPECT_TRUE(output.find("Spawns: 1") != std::string::npos);
    EXPECT_TRUE(output.find("✓ Level data valid") != std::string::npos);
    EXPECT_TRUE(output.find("✓ File validation completed successfully") != std::string::npos);
    EXPECT_TRUE(output.find("EXIT_CODE:0") != std::string::npos);
}

// Test file inspector with recording file
TEST_F(FileInspectorTest, InspectorHandlesRecordingFile) {
    ASSERT_TRUE(fileExists(testRecordingFile));
    
    std::string output = runFileInspector(testRecordingFile);
    
    // Check for expected recording output (v5 format)
    EXPECT_TRUE(output.find("Recording File:") != std::string::npos);
    EXPECT_TRUE(output.find("✓ Valid magic number (MESR)") != std::string::npos);
    EXPECT_TRUE(output.find("✓ Valid version (v5 with sensor config)") != std::string::npos);
    EXPECT_TRUE(output.find("✓ File structure intact (v5 format with checksums and sensor config)") != std::string::npos);
    EXPECT_TRUE(output.find("✓ Metadata fields within valid ranges") != std::string::npos);
    
    // Check metadata content
    EXPECT_TRUE(output.find("Simulation: madrona_escape_room") != std::string::npos);
    EXPECT_TRUE(output.find("Worlds: 1, Agents per world: 1") != std::string::npos);
    EXPECT_TRUE(output.find("Steps recorded: 3") != std::string::npos);
    EXPECT_TRUE(output.find("Actions per step: 3") != std::string::npos);
    
    // Check embedded level content (v5 format shows levels)
    EXPECT_TRUE(output.find("Embedded Levels") != std::string::npos);
    EXPECT_TRUE(output.find("embedded_test_level") != std::string::npos);
    EXPECT_TRUE(output.find("5x4 grid") != std::string::npos);
    EXPECT_TRUE(output.find("Scale: 1.5") != std::string::npos);
    
    EXPECT_TRUE(output.find("✓ File validation completed successfully") != std::string::npos);
    EXPECT_TRUE(output.find("EXIT_CODE:0") != std::string::npos);
}

// Test file inspector error handling
TEST_F(FileInspectorTest, InspectorHandlesErrors) {
    // Test with non-existent file
    std::string output = runFileInspector("nonexistent.rec");
    EXPECT_TRUE(output.find("Error: Cannot access file or file is empty") != std::string::npos);
    EXPECT_TRUE(output.find("EXIT_CODE:1") != std::string::npos);
    
    // Test with unsupported extension
    std::string invalidFile = testDataDir + "/test.xyz";
    std::ofstream file(invalidFile);
    file << "dummy content";
    file.close();
    
    output = runFileInspector(invalidFile);
    EXPECT_TRUE(output.find("Error: Unsupported file type") != std::string::npos);
    EXPECT_TRUE(output.find("EXIT_CODE:1") != std::string::npos);
}

// Test file inspector with corrupted level file
TEST_F(FileInspectorTest, InspectorHandlesCorruptedLevelFile) {
    std::string corruptedFile = testDataDir + "/corrupted.lvl";
    
    // Create a file with wrong size
    std::ofstream file(corruptedFile, std::ios::binary);
    char dummy[100] = {0}; // Much smaller than sizeof(CompiledLevel)
    file.write(dummy, sizeof(dummy));
    file.close();
    
    std::string output = runFileInspector(corruptedFile);
    EXPECT_TRUE(output.find("✗ Failed to read level file") != std::string::npos);
    EXPECT_TRUE(output.find("✗ File validation failed") != std::string::npos);
    EXPECT_TRUE(output.find("EXIT_CODE:1") != std::string::npos);
}

// Test file inspector command line argument handling
TEST_F(FileInspectorTest, InspectorHandlesNoArguments) {
    std::string command = "./build/file_inspector 2>&1";
    FILE* pipe = popen(command.c_str(), "r");
    ASSERT_NE(pipe, nullptr);
    
    std::stringstream result;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result << buffer;
    }
    
    int exitCode = pclose(pipe);
    std::string output = result.str();
    
    EXPECT_TRUE(output.find("Usage:") != std::string::npos);
    EXPECT_TRUE(output.find("file.(rec|lvl)") != std::string::npos);
    EXPECT_TRUE(output.find("Recording files with metadata") != std::string::npos);
    EXPECT_TRUE(output.find("Compiled level files") != std::string::npos);
    EXPECT_EQ(WEXITSTATUS(exitCode), 1);
}

// Test that file inspector correctly identifies file types
TEST_F(FileInspectorTest, InspectorIdentifiesFileTypes) {
    // Test with .lvl file
    ASSERT_TRUE(fileExists(testLevelFile));
    std::string lvlOutput = runFileInspector(testLevelFile);
    EXPECT_TRUE(lvlOutput.find("Level File:") != std::string::npos);
    EXPECT_FALSE(lvlOutput.find("Recording File:") != std::string::npos);
    
    // Test with .rec file
    ASSERT_TRUE(fileExists(testRecordingFile));
    std::string recOutput = runFileInspector(testRecordingFile);
    EXPECT_TRUE(recOutput.find("Recording File:") != std::string::npos);
    EXPECT_FALSE(recOutput.find("Level File:") != std::string::npos);
}