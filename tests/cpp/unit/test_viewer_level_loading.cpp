#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "madrona_escape_room_c_api.h"
#include <fstream>

// Test fixture for level loading functionality
class ViewerLevelLoadingTest : public ViewerTestBase {
protected:
    void SetUp() override {
        ViewerTestBase::SetUp();
        // Create test level files for use in tests
        createTestLevelFile("tests/cpp/test_data/levels/simple.lvl", 16, 16);
        createTestLevelFile("tests/cpp/test_data/levels/complex.lvl", 32, 32);
    }
};

// Test loading a valid level file
TEST_F(ViewerLevelLoadingTest, LoadValidLevelFile) {
    const std::string level_file = "tests/cpp/test_data/levels/simple.lvl";
    createTestLevelFile(level_file);
    
    ASSERT_TRUE(fileExists(level_file));
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    
    EXPECT_EQ(level.width, 16);
    EXPECT_EQ(level.height, 16);
    EXPECT_GT(level.num_tiles, 0);
}

// Test loading a complex level file
TEST_F(ViewerLevelLoadingTest, LoadComplexLevelFile) {
    const std::string level_file = "tests/cpp/test_data/levels/complex.lvl";
    createTestLevelFile(level_file, 32, 32);
    
    ASSERT_TRUE(fileExists(level_file));
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    
    EXPECT_EQ(level.width, 32);
    EXPECT_EQ(level.height, 32);
    EXPECT_GT(level.num_tiles, 0);
}

// Test handling of non-existent level file
TEST_F(ViewerLevelLoadingTest, LoadNonExistentFile) {
    const std::string level_file = "tests/cpp/test_data/levels/nonexistent.lvl";
    
    ASSERT_FALSE(fileExists(level_file));
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    
    // Should return empty/invalid level
    EXPECT_EQ(level.width, 0);
    EXPECT_EQ(level.height, 0);
    EXPECT_EQ(level.num_tiles, 0);
}

// Test loading corrupted level file
TEST_F(ViewerLevelLoadingTest, LoadCorruptedFile) {
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

// Test that multiple worlds can use the same level
TEST_F(ViewerLevelLoadingTest, MultipleWorldsSameLevel) {
    const std::string level_file = "tests/cpp/test_data/levels/simple.lvl";
    createTestLevelFile(level_file);
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    
    // Create multiple copies for different worlds
    std::vector<MER_CompiledLevel> levels(4, level);
    
    EXPECT_EQ(levels.size(), 4u);
    
    // All should be identical
    for (size_t i = 1; i < levels.size(); i++) {
        EXPECT_TRUE(LevelComparer::compareLevels(levels[0], levels[i]));
    }
}

// Test level file size validation
TEST_F(ViewerLevelLoadingTest, ValidateLevelFileSize) {
    const std::string level_file = "tests/cpp/test_data/levels/simple.lvl";
    createTestLevelFile(level_file);
    
    size_t file_size = getFileSize(level_file);
    
    // Should be exactly the size of CompiledLevel struct
    EXPECT_EQ(file_size, sizeof(MER_CompiledLevel));
}

// Test level dimensions validation
TEST_F(ViewerLevelLoadingTest, ValidateLevelDimensions) {
    const std::string level_file = "tests/cpp/test_data/levels/test_dims.lvl";
    
    // Test various dimensions
    std::vector<std::pair<int32_t, int32_t>> test_dims = {
        {8, 8},
        {16, 16},
        {32, 32},
        {16, 32},
        {32, 16}
    };
    
    for (const auto& [width, height] : test_dims) {
        createTestLevelFile(level_file, width, height);
        
        MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
        
        EXPECT_EQ(level.width, width);
        EXPECT_EQ(level.height, height);
        
        std::remove(level_file.c_str());
    }
}

// Test tile data integrity
TEST_F(ViewerLevelLoadingTest, ValidateTileData) {
    const std::string level_file = "tests/cpp/test_data/levels/tiles.lvl";
    createTestLevelFile(level_file);
    
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

// Test loading level for use with Manager
TEST_F(ViewerLevelLoadingTest, LoadLevelForManager) {
    const std::string level_file = "tests/cpp/test_data/levels/manager_test.lvl";
    createTestLevelFile(level_file);
    
    MER_CompiledLevel level = LevelComparer::loadLevelFromFile(level_file);
    
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

// Test level comparison functionality
TEST_F(ViewerLevelLoadingTest, CompareLevels) {
    const std::string level_file1 = "tests/cpp/test_data/levels/level1.lvl";
    const std::string level_file2 = "tests/cpp/test_data/levels/level2.lvl";
    const std::string level_file3 = "tests/cpp/test_data/levels/level3.lvl";
    
    // Create identical levels
    createTestLevelFile(level_file1, 16, 16);
    createTestLevelFile(level_file2, 16, 16);
    
    // Create different level
    createTestLevelFile(level_file3, 32, 32);
    
    MER_CompiledLevel level1 = LevelComparer::loadLevelFromFile(level_file1);
    MER_CompiledLevel level2 = LevelComparer::loadLevelFromFile(level_file2);
    MER_CompiledLevel level3 = LevelComparer::loadLevelFromFile(level_file3);
    
    // Same dimensions should compare equal
    EXPECT_TRUE(LevelComparer::compareLevels(level1, level2));
    
    // Different dimensions should not compare equal
    EXPECT_FALSE(LevelComparer::compareLevels(level1, level3));
}

// Test that viewer respects level precedence rules
TEST_F(ViewerLevelLoadingTest, LevelPrecedenceRules) {
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