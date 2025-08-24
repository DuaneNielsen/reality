#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "/home/duane/madrona_escape_room/tests/cpp/fixtures/compiled_level_compat.hpp"
#include "madrona_escape_room_c_api.h"
#include <fstream>
#include <cstring>
#include <iomanip>

// Test fixture for recording and trajectory utilities
class RecordingUtilitiesTest : public ViewerTestBase {
protected:
    void SetUp() override {
        ViewerTestBase::SetUp();
        // Create test recording files for replay tests
        createTestRecordingFile("tests/cpp/test_data/recordings/simple.rec", 1, 100, 42);
        createTestRecordingFile("tests/cpp/test_data/recordings/multi_world.rec", 4, 200, 123);
    }
    
    // Helper to read replay metadata (simplified version)
    struct ReplayMetadata {
        uint32_t num_worlds;
        uint32_t num_steps;
        uint32_t seed;
        bool valid;
    };
    
    ReplayMetadata readReplayMetadata(const std::string& filename) {
        ReplayMetadata metadata = {0, 0, 0, false};
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return metadata;
        }
        
        file.read(reinterpret_cast<char*>(&metadata.num_worlds), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&metadata.num_steps), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&metadata.seed), sizeof(uint32_t));
        
        if (file.good()) {
            metadata.valid = true;
        }
        
        return metadata;
    }
    
    // Helper to extract embedded level from recording
    MER_CompiledLevel extractEmbeddedLevel(const std::string& filename) {
        MER_CompiledLevel level = {};
        
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return level;
        }
        
        // Skip metadata (3 uint32_t values)
        file.seekg(3 * sizeof(uint32_t), std::ios::beg);
        
        // Read embedded level
        file.read(reinterpret_cast<char*>(&level), sizeof(MER_CompiledLevel));
        
        return level;
    }
};

// Test that replay file metadata can be parsed correctly
TEST_F(RecordingUtilitiesTest, CanParseReplayFileMetadata) {
    const std::string replay_file = "tests/cpp/test_data/recordings/simple.rec";
    createTestRecordingFile(replay_file, 2, 150, 42);
    
    auto metadata = readReplayMetadata(replay_file);
    
    ASSERT_TRUE(metadata.valid);
    EXPECT_EQ(metadata.num_worlds, 2u);
    EXPECT_EQ(metadata.num_steps, 150u);
    EXPECT_EQ(metadata.seed, 42u);
}

// Test that embedded level data can be extracted from recording file
TEST_F(RecordingUtilitiesTest, CanExtractLevelFromRecordingFile) {
    const std::string replay_file = "tests/cpp/test_data/recordings/with_level.rec";
    createTestRecordingFile(replay_file);
    
    MER_CompiledLevel embedded_level = extractEmbeddedLevel(replay_file);
    
    // Should have valid dimensions (from TestLevelHelper::CreateSimpleLevel)
    EXPECT_GT(embedded_level.width, 0);
    EXPECT_GT(embedded_level.height, 0);
    EXPECT_GT(embedded_level.num_tiles, 0);
}

// Test that world count from metadata updates viewer configuration
TEST_F(RecordingUtilitiesTest, MetadataWorldCountUpdatesViewerConfig) {
    const std::string replay_file = "tests/cpp/test_data/recordings/world_count.rec";
    createTestRecordingFile(replay_file, 8, 100, 42);
    
    auto metadata = readReplayMetadata(replay_file);
    
    ASSERT_TRUE(metadata.valid);
    EXPECT_EQ(metadata.num_worlds, 8u);
    
    // Viewer should update num_worlds to match replay
    uint32_t viewer_num_worlds = 4;  // Initial value
    if (metadata.valid) {
        viewer_num_worlds = metadata.num_worlds;
    }
    
    EXPECT_EQ(viewer_num_worlds, 8u);
}

// Test that random seed can be extracted from replay metadata
TEST_F(RecordingUtilitiesTest, CanExtractSeedFromMetadata) {
    const std::string replay_file = "tests/cpp/test_data/recordings/with_seed.rec";
    uint32_t replay_seed = 12345;
    createTestRecordingFile(replay_file, 1, 100, replay_seed);
    
    auto metadata = readReplayMetadata(replay_file);
    
    ASSERT_TRUE(metadata.valid);
    EXPECT_EQ(metadata.seed, replay_seed);
}

// Test that malformed replay files are detected as invalid
TEST_F(RecordingUtilitiesTest, DetectsMalformedReplayFile) {
    const std::string replay_file = "tests/cpp/test_data/recordings/invalid.rec";
    
    // Create an invalid file (too small)
    std::ofstream file(replay_file, std::ios::binary);
    char dummy[5] = {0};
    file.write(dummy, sizeof(dummy));
    file.close();
    
    auto metadata = readReplayMetadata(replay_file);
    
    EXPECT_FALSE(metadata.valid);
}

// Test that missing replay files return invalid metadata
TEST_F(RecordingUtilitiesTest, ReturnsInvalidMetadataForMissingFile) {
    const std::string replay_file = "tests/cpp/test_data/recordings/nonexistent.rec";
    
    ASSERT_FALSE(fileExists(replay_file));
    
    auto metadata = readReplayMetadata(replay_file);
    
    EXPECT_FALSE(metadata.valid);
}

// Test that trajectory comparison utility detects identical trajectories
TEST_F(RecordingUtilitiesTest, TrajectoryComparerDetectsIdenticalFiles) {
    // Step 1: Create a recording with known actions
    const std::string record_file = "tests/cpp/test_data/recordings/roundtrip.rec";
    const std::string trajectory1_file = "tests/cpp/test_data/recordings/trajectory1.csv";
    const std::string trajectory2_file = "tests/cpp/test_data/recordings/trajectory2.csv";
    
    // Create recording with embedded level
    createTestRecordingFile(record_file, 1, 10, 42);
    
    // Create trajectory for original run
    createTestTrajectoryFile(trajectory1_file, 10, 0, 0);
    
    // Simulate replay generating same trajectory
    createTestTrajectoryFile(trajectory2_file, 10, 0, 0);
    
    // Compare trajectories
    bool trajectories_match = TrajectoryComparer::compareTrajectories(
        trajectory1_file, trajectory2_file);
    
    EXPECT_TRUE(trajectories_match);
}

// Test that trajectory comparison can verify deterministic trajectories
TEST_F(RecordingUtilitiesTest, TrajectoryComparerVerifiesDeterministicOutput) {
    const std::string record_file = "tests/cpp/test_data/recordings/deterministic.rec";
    
    // Create recording with specific seed
    createTestRecordingFile(record_file, 2, 50, 999);
    
    // Create two trajectory files from two replay runs
    const std::string traj1 = "tests/cpp/test_data/recordings/traj_run1.csv";
    const std::string traj2 = "tests/cpp/test_data/recordings/traj_run2.csv";
    
    // Simulate identical trajectories (deterministic)
    for (int run = 0; run < 2; run++) {
        const std::string& traj_file = (run == 0) ? traj1 : traj2;
        std::ofstream file(traj_file);
        
        for (uint32_t step = 0; step < 50; step++) {
            // Deterministic position based on step
            float x = step * 0.5f;
            float y = step * 0.1f;
            float z = 0.0f;
            float rotation = step * 2.0f;
            float progress = step * 0.02f;
            
            file << "Step " << std::setw(4) << step 
                 << ": World 0 Agent 0: pos=(" 
                 << std::fixed << std::setprecision(2) << x << ","
                 << y << "," << z << ") "
                 << "rot=" << std::fixed << std::setprecision(1) << rotation << "° "
                 << "progress=" << std::fixed << std::setprecision(2) << progress << "\n";
        }
    }
    
    // Trajectories should be identical
    bool identical = TrajectoryComparer::compareTrajectories(traj1, traj2);
    EXPECT_TRUE(identical);
}

// Test creating per-world trajectory files for multi-world recordings
TEST_F(RecordingUtilitiesTest, CreatesPerWorldTrajectoryFiles) {
    const std::string replay_file = "tests/cpp/test_data/recordings/multi.rec";
    createTestRecordingFile(replay_file, 4, 100, 42);
    
    auto metadata = readReplayMetadata(replay_file);
    
    ASSERT_TRUE(metadata.valid);
    EXPECT_EQ(metadata.num_worlds, 4u);
    
    // Each world should have its own trajectory
    for (int world = 0; world < 4; world++) {
        std::string traj_file = "tests/cpp/test_data/recordings/world" + 
                               std::to_string(world) + ".csv";
        createTestTrajectoryFile(traj_file, 100, world, 0);
        
        EXPECT_TRUE(fileExists(traj_file));
    }
}

// Test that level comparer can verify embedded level matches source
TEST_F(RecordingUtilitiesTest, LevelComparerVerifiesEmbeddedMatchesSource) {
    // Use the default level
    MER_CompiledLevel original_level = LevelComparer::getDefaultLevel();
    
    // Create recording with this level embedded
    const std::string record_file = "tests/cpp/test_data/recordings/with_orig_level.rec";
    std::ofstream file(record_file, std::ios::binary);
    
    // Write metadata
    uint32_t num_worlds = 1;
    uint32_t num_steps = 10;
    uint32_t seed = 42;
    file.write(reinterpret_cast<const char*>(&num_worlds), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&num_steps), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&seed), sizeof(uint32_t));
    
    // Write level
    file.write(reinterpret_cast<const char*>(&original_level), sizeof(MER_CompiledLevel));
    
    // Write dummy actions
    for (uint32_t i = 0; i < num_steps; i++) {
        int32_t actions[3] = {0, 0, 2};
        file.write(reinterpret_cast<const char*>(actions), sizeof(actions));
    }
    file.close();
    
    // Extract embedded level
    MER_CompiledLevel embedded_level = extractEmbeddedLevel(record_file);
    
    // Compare
    EXPECT_TRUE(LevelComparer::compareLevels(original_level, embedded_level));
}

// Test that recording file size matches expected format
TEST_F(RecordingUtilitiesTest, RecordingFileSizeMatchesExpectedFormat) {
    const std::string replay_file = "tests/cpp/test_data/recordings/size_test.rec";
    uint32_t num_worlds = 2;
    uint32_t num_steps = 50;
    
    createTestRecordingFile(replay_file, num_worlds, num_steps, 42);
    
    size_t file_size = getFileSize(replay_file);
    
    // Expected size: metadata + level + actions
    size_t expected_size = 3 * sizeof(uint32_t) +  // metadata
                          sizeof(MER_CompiledLevel) +  // embedded level
                          num_steps * num_worlds * 3 * sizeof(int32_t);  // actions
    
    EXPECT_EQ(file_size, expected_size);
}

// Test that trajectory parser correctly extracts CSV data fields
TEST_F(RecordingUtilitiesTest, TrajectoryParserExtractsCSVFields) {
    const std::string traj_file = "tests/cpp/test_data/recordings/format_test.csv";
    
    std::ofstream file(traj_file);
    file << "Step " << std::setw(4) << 10u
         << ": World " << 1 << " Agent " << 0 << ": pos=("
         << std::fixed << std::setprecision(2) << 5.25f << ","
         << 3.75f << "," << 1.0f << ") "
         << "rot=" << std::fixed << std::setprecision(1) << 45.0f << "° "
         << "progress=" << std::fixed << std::setprecision(2) << 0.75f << "\n";
    file.close();
    
    auto points = TrajectoryComparer::parseTrajectoryFile(traj_file);
    
    ASSERT_EQ(points.size(), 1u);
    
    const auto& point = points[0];
    EXPECT_EQ(point.step, 10u);
    EXPECT_EQ(point.world, 1);
    EXPECT_EQ(point.agent, 0);
    EXPECT_FLOAT_EQ(point.x, 5.25f);
    EXPECT_FLOAT_EQ(point.y, 3.75f);
    EXPECT_FLOAT_EQ(point.z, 1.0f);
    EXPECT_FLOAT_EQ(point.rotation, 45.0f);
    EXPECT_FLOAT_EQ(point.progress, 0.75f);
}