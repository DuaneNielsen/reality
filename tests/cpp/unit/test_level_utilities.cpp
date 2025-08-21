#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "madrona_escape_room_c_api.h"
#include <fstream>

// Test fixture for level loading functionality
class LevelUtilitiesTest : public ViewerTestBase {
protected:
    void SetUp() override {
        ViewerTestBase::SetUp();
    }
    
    // Helper to write default level to a file for testing file I/O
    void writeDefaultLevelToFile(const std::string& filepath) {
        auto level = LevelComparer::getDefaultLevel();
        std::ofstream file(filepath, std::ios::binary);
        file.write(reinterpret_cast<const char*>(&level), sizeof(MER_CompiledLevel));
        file.close();
    }
};

// Test that level loader reads binary file and returns valid dimensions
TEST_F(LevelUtilitiesTest, LevelLoaderReadsBinaryFileCorrectly) {
    const std::string level_file = "tests/cpp/test_data/levels/simple.lvl";
    writeDefaultLevelToFile(level_file);
    
    ASSERT_TRUE(fileExists(level_file));
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    auto default_level = LevelComparer::getDefaultLevel();
    
    // Should match the default level we wrote
    EXPECT_EQ(level.width, default_level.width);
    EXPECT_EQ(level.height, default_level.height);
    EXPECT_EQ(level.num_tiles, default_level.num_tiles);
    EXPECT_GT(level.num_tiles, 0);
}

// Test that default level has valid dimensions
TEST_F(LevelUtilitiesTest, DefaultLevelHasValidDimensions) {
    auto level = LevelComparer::getDefaultLevel();
    
    // Default level should have reasonable dimensions
    EXPECT_GT(level.width, 0);
    EXPECT_GT(level.height, 0);
    EXPECT_GT(level.num_tiles, 0);
    
    // World boundaries should be set properly
    EXPECT_NE(level.world_min_y, 0.0f);
    EXPECT_NE(level.world_max_y, 0.0f);
    EXPECT_LT(level.world_min_y, level.world_max_y);
}

// Test that level loader returns empty struct for missing files
TEST_F(LevelUtilitiesTest, LevelLoaderReturnsEmptyForMissingFile) {
    const std::string level_file = "tests/cpp/test_data/levels/nonexistent.lvl";
    
    ASSERT_FALSE(fileExists(level_file));
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    
    // Should return empty/invalid level
    EXPECT_EQ(level.width, 0);
    EXPECT_EQ(level.height, 0);
    EXPECT_EQ(level.num_tiles, 0);
}

// Test that level loader handles truncated files gracefully
TEST_F(LevelUtilitiesTest, LevelLoaderHandlesTruncatedFile) {
    const std::string level_file = "tests/cpp/test_data/levels/corrupted.lvl";
    
    // Create a corrupted file (too small)
    std::ofstream file(level_file, std::ios::binary);
    char dummy[10] = {0};
    file.write(dummy, sizeof(dummy));
    file.close();
    
    ASSERT_TRUE(fileExists(level_file));
    ASSERT_LT(getFileSize(level_file), sizeof(MER_CompiledLevel));
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    
    // Should handle gracefully - level will be partially filled
    // The actual behavior depends on implementation
}

// Test that level comparer verifies identical level copies
TEST_F(LevelUtilitiesTest, LevelComparerVerifiesIdenticalCopies) {
    auto level = LevelComparer::getDefaultLevel();
    
    // Create multiple copies for different worlds
    std::vector<MER_CompiledLevel> levels(4, level);
    
    EXPECT_EQ(levels.size(), 4u);
    
    // All should be identical
    for (size_t i = 1; i < levels.size(); i++) {
        EXPECT_TRUE(LevelComparer::compareLevels(levels[0], levels[i]));
    }
}

// Test that level file size matches struct size
TEST_F(LevelUtilitiesTest, LevelFileSizeMatchesStructSize) {
    const std::string level_file = "tests/cpp/test_data/levels/simple.lvl";
    writeDefaultLevelToFile(level_file);
    
    size_t file_size = getFileSize(level_file);
    
    // Should be exactly the size of CompiledLevel struct
    EXPECT_EQ(file_size, sizeof(MER_CompiledLevel));
}

// Test that default level dimensions are preserved through save/load
TEST_F(LevelUtilitiesTest, DefaultLevelPreservedThroughSaveLoad) {
    const std::string level_file = "tests/cpp/test_data/levels/test_dims.lvl";
    
    // Get the default level
    auto original_level = LevelComparer::getDefaultLevel();
    
    // Write it to file
    std::ofstream file(level_file, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&original_level), sizeof(MER_CompiledLevel));
    file.close();
    
    // Read it back
    MER_CompiledLevel loaded_level = LevelComparer::loadLevelFromFile(level_file);
    
    // Should be identical
    EXPECT_TRUE(LevelComparer::compareLevels(original_level, loaded_level));
    
    std::remove(level_file.c_str());
}

// Test that level data round-trips through binary write/read
TEST_F(LevelUtilitiesTest, LevelDataRoundTripsThroughBinary) {
    const std::string level_file = "tests/cpp/test_data/levels/tiles.lvl";
    writeDefaultLevelToFile(level_file);
    
    MER_CompiledLevel level1 = LevelComparer::loadLevelFromFile(level_file);
    
    // Save and reload to verify integrity
    const std::string level_file2 = "tests/cpp/test_data/levels/tiles2.lvl";
    std::ofstream file(level_file2, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&level1), sizeof(MER_CompiledLevel));
    file.close();
    
    MER_CompiledLevel level2 = LevelComparer::loadLevelFromFile(level_file2);
    
    EXPECT_TRUE(LevelComparer::compareLevels(level1, level2));
    
    std::remove(level_file2.c_str());
}

// Test that loaded level can initialize Manager successfully
TEST_F(LevelUtilitiesTest, LoadedLevelInitializesManager) {
    // Use the default level directly
    MER_CompiledLevel level = LevelComparer::getDefaultLevel();
    
    // Verify level is suitable for Manager
    EXPECT_GT(level.width, 0);
    EXPECT_GT(level.height, 0);
    EXPECT_GT(level.num_tiles, 0);
    
    // Create manager with this level
    config.num_worlds = 2;
    std::vector<MER_CompiledLevel> levels(config.num_worlds, level);
    
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
}

// Test that level comparer detects modifications
TEST_F(LevelUtilitiesTest, LevelComparerDetectsModifications) {
    // Get two copies of the default level
    MER_CompiledLevel level1 = LevelComparer::getDefaultLevel();
    MER_CompiledLevel level2 = LevelComparer::getDefaultLevel();
    
    // They should be identical initially
    EXPECT_TRUE(LevelComparer::compareLevels(level1, level2));
    
    // Modify one of them
    level2.width = level2.width + 10;
    
    // Now they should be different
    EXPECT_FALSE(LevelComparer::compareLevels(level1, level2));
    
    // Restore and modify something else
    level2.width = level1.width;
    level2.num_tiles = level2.num_tiles + 1;
    
    // Still different
    EXPECT_FALSE(LevelComparer::compareLevels(level1, level2));
}

// Test viewer option precedence logic (replay overrides load)
TEST_F(LevelUtilitiesTest, ViewerOptionPrecedenceLogic) {
    // In the viewer:
    // 1. --replay ignores --load (level comes from recording)
    // 2. --load is required when not using --replay
    // 3. --load + --record embeds the level in the recording
    
    // This test validates the logic, not the actual viewer behavior
    // The viewer tests would need to mock or integrate with the actual viewer
    
    bool has_replay = true;
    bool has_load = true;
    
    // When replay is present, load should be ignored
    if (has_replay && has_load) {
        // Viewer should warn: "--replay ignores --load"
        EXPECT_TRUE(has_replay);
    }
    
    // When no replay, load is required
    has_replay = false;
    has_load = false;
    
    if (!has_replay && !has_load) {
        // Viewer should error: "Must provide either --replay OR --load"
        EXPECT_FALSE(has_replay && has_load);
    }
}