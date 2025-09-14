#include <gtest/gtest.h>
#include <filesystem>
#include <memory>

#include "mgr.hpp"
#include "test_level_helper.hpp"

using namespace madEscape;
using namespace madrona;

class ManagerFromReplayTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_replay_file = std::filesystem::temp_directory_path() / "test_replay.rec";
    }
    
    void TearDown() override {
        // Clean up test files
        if (std::filesystem::exists(test_replay_file)) {
            std::filesystem::remove(test_replay_file);
        }
    }
    
    std::filesystem::path test_replay_file;
};

TEST_F(ManagerFromReplayTest, CreatesWithEmbeddedLevels) {
    // Record with specific level
    {
        auto level = DefaultLevelProvider::GetDefaultLevel();
        Manager::Config cfg;
        cfg.execMode = ExecMode::CPU;
        cfg.gpuID = -1;
        cfg.numWorlds = 2;
        cfg.randSeed = 42;
        cfg.autoReset = true;
        cfg.enableBatchRenderer = false;
        cfg.perWorldCompiledLevels = {level, level};
        
        Manager mgr(cfg);
        auto result = mgr.startRecording(test_replay_file.string());
        ASSERT_EQ(result, Result::Success);
        
        for (int i = 0; i < 10; i++) {
            mgr.step();
        }
        mgr.stopRecording();
    }
    
    // Create from replay - should use embedded level
    auto replay_mgr = Manager::fromReplay(test_replay_file.string(), ExecMode::CPU, -1);
    ASSERT_NE(replay_mgr, nullptr);
    
    // Verify it works
    for (int i = 0; i < 10; i++) {
        bool finished = replay_mgr->replayStep();
        replay_mgr->step();
        if (finished) {
            break;
        }
    }
}

TEST_F(ManagerFromReplayTest, FailsOnNonExistentFile) {
    std::string non_existent_file = "/tmp/does_not_exist.rec";
    auto replay_mgr = Manager::fromReplay(non_existent_file, ExecMode::CPU, -1);
    ASSERT_EQ(replay_mgr, nullptr);
}

TEST_F(ManagerFromReplayTest, PreservesReplayMetadata) {
    // Record with specific configuration
    {
        auto level = DefaultLevelProvider::GetDefaultLevel();
        Manager::Config cfg;
        cfg.execMode = ExecMode::CPU;
        cfg.gpuID = -1;
        cfg.numWorlds = 3;
        cfg.randSeed = 12345;
        cfg.autoReset = true;
        cfg.enableBatchRenderer = false;
        cfg.perWorldCompiledLevels = {level, level, level};
        
        Manager mgr(cfg);
        auto result = mgr.startRecording(test_replay_file.string());
        ASSERT_EQ(result, Result::Success);
        
        for (int i = 0; i < 5; i++) {
            mgr.step();
        }
        mgr.stopRecording();
    }
    
    // Read metadata to verify
    auto metadata = Manager::readReplayMetadata(test_replay_file.string());
    ASSERT_TRUE(metadata.has_value());
    EXPECT_EQ(metadata->num_worlds, 3u);
    EXPECT_EQ(metadata->seed, 12345u);
    EXPECT_EQ(metadata->num_steps, 5u);
    
    // Create from replay and verify configuration
    auto replay_mgr = Manager::fromReplay(test_replay_file.string(), ExecMode::CPU, -1);
    ASSERT_NE(replay_mgr, nullptr);
    
    // Verify replay can be stepped through
    uint32_t step_count = 0;
    for (int i = 0; i < 5; i++) {
        bool finished = replay_mgr->replayStep();
        replay_mgr->step();
        step_count++;
        if (finished) {
            break;
        }
    }
    EXPECT_EQ(step_count, 5u);
}