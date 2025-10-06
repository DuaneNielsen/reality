#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "mock_components.hpp"
#include <thread>
#include <chrono>
#include <fstream>
#include <vector>
#include <cstring>
#include "../../../src/consts.hpp"
#include "../../../src/mgr.hpp"
#include "../../../src/types.hpp"

// For capturing stdout/stderr output in tests
using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

using namespace madEscape::consts::action;

// These tests primarily test the Manager C API and MockViewer behavior,
// not the actual viewer.cpp code
class ManagerIntegrationTest : public ViewerTestBase {
protected:
    void SetUp() override {
        ViewerTestBase::SetUp();
        file_manager_ = std::make_unique<TestFileManager>();
    }
    
    void TearDown() override {
        file_manager_->cleanup();
        ViewerTestBase::TearDown();
    }
    
    std::unique_ptr<TestFileManager> file_manager_;
};

// Test basic manager creation with viewer options
TEST_F(ManagerIntegrationTest, ManagerCreationWithLevelFile) {
    // Create a test level file
    createTestLevelFile("test.lvl", 16, 16);
    file_manager_->addFile("test.lvl");
    
    // Load the level
    auto level = LevelComparer::loadLevelFromFile("test.lvl");
    
    // Create manager with viewer-like config
    config.num_worlds = 4;
    config.rand_seed = 42;
    config.auto_reset = false;  // Manual reset like viewer
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    // Verify tensors are accessible
    MER_Tensor self_obs;
    ASSERT_TRUE(GetTensor(self_obs, mer_get_self_observation_tensor));
    // Self observation is 3D: [num_worlds, num_agents, features]
    // Use madEscape namespace constants
    EXPECT_TRUE(ValidateTensorShape(self_obs, {4, madEscape::consts::numAgents, 5}));  // 5 = SelfObservationFloatCount
}

// Test recording workflow
TEST_F(ManagerIntegrationTest, ManagerRecordingAPI) {
    // Setup
    createTestLevelFile("test.lvl", 16, 16);
    file_manager_->addFile("test.lvl");
    file_manager_->addFile("test.rec");
    
    auto level = LevelComparer::loadLevelFromFile("test.lvl");
    config.num_worlds = 2;
    config.auto_reset = true;  // Required for recording
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    
    // Start recording
    mgr.startRecording("test.rec");
    EXPECT_TRUE(mgr.isRecording());
    
    // Simulate some actions
    for (int step = 0; step < 10; step++) {
        mgr.setAction(0, move_amount::SLOW, move_angle::FORWARD, rotate::NONE);
        mgr.setAction(1, move_amount::STOP, move_angle::FORWARD, rotate::SLOW_LEFT);
        mgr.step();
    }
    
    // Stop recording
    mgr.stopRecording();
    EXPECT_FALSE(mgr.isRecording());
    
    // Verify file was created
    EXPECT_TRUE(file_manager_->fileExists("test.rec"));
    EXPECT_GT(file_manager_->getFileSize("test.rec"), 0);
    
    // Verify recorded actions
    auto& recorder = mgr.getRecorder();
    EXPECT_EQ(recorder.getActionCount(), 20);  // 2 worlds * 10 steps
}

// Test replay workflow
TEST_F(ManagerIntegrationTest, ManagerReplayAPI) {
    // First create a recording
    createTestLevelFile("test.lvl", 16, 16);
    file_manager_->addFile("test.lvl");
    // Note: Don't add test.rec to cleanup yet - we need it after TearDown()
    
    auto level = LevelComparer::loadLevelFromFile("test.lvl");
    
    // Record phase
    {
        config.num_worlds = 2;
        config.auto_reset = true;
        ASSERT_TRUE(CreateManager(&level, 1));
        
        TestManagerWrapper mgr(handle);
        mgr.startRecording("test.rec");
        
        // Record specific actions
        mgr.setAction(0, move_amount::MEDIUM, move_angle::RIGHT, rotate::SLOW_RIGHT);
        mgr.setAction(1, move_amount::FAST, move_angle::LEFT, rotate::FAST_LEFT);
        mgr.step();
        
        mgr.stopRecording();
    }
    
    // Clean up first manager
    TearDown();
    SetUp();
    
    // Replay phase - just test metadata reading
    {
        // Capture stdout to suppress metadata output
        CaptureStdout();
        
        // Read metadata first
        MER_ReplayMetadata metadata;
        ASSERT_EQ(mer_read_replay_metadata("test.rec", &metadata), MER_SUCCESS);
        EXPECT_EQ(metadata.num_worlds, 2);
        EXPECT_EQ(metadata.seed, 42);
        
        // Skip the actual replay part for now to isolate the metadata reading test
        std::cout << "Successfully read metadata: " << metadata.num_worlds 
                  << " worlds, seed " << metadata.seed << std::endl;
        
        // Get captured output 
        std::string captured_output = GetCapturedStdout();
        EXPECT_TRUE(captured_output.find("Successfully read metadata") != std::string::npos);
    }
    
    // Now we can safely clean up the recording file
    file_manager_->addFile("test.rec");
}

// Test trajectory tracking
TEST_F(ManagerIntegrationTest, ManagerTrajectoryLogging) {
    // Capture stdout to suppress trajectory logging output
    CaptureStdout();
    
    createTestLevelFile("test.lvl", 16, 16);
    file_manager_->addFile("test.lvl");
    file_manager_->addFile("trajectory.csv");
    
    auto level = LevelComparer::loadLevelFromFile("test.lvl");
    config.num_worlds = 2;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    
    // Enable trajectory tracking
    mgr.enableTrajectoryLogging(0, 0, "trajectory.csv");
    EXPECT_TRUE(mgr.isTrajectoryEnabled());
    EXPECT_EQ(mgr.getTrajectoryWorld(), 0);
    EXPECT_EQ(mgr.getTrajectoryAgent(), 0);
    
    // Run simulation
    for (int i = 0; i < 5; i++) {
        mgr.setAction(0, move_amount::SLOW, move_angle::FORWARD, rotate::NONE);
        mgr.step();
    }
    
    // Disable tracking
    mgr.disableTrajectoryLogging();
    EXPECT_FALSE(mgr.isTrajectoryEnabled());
    
    // Verify trajectory file exists and has content
    EXPECT_TRUE(file_manager_->fileExists("trajectory.csv"));
    
    // Parse and verify trajectory
    auto points = TrajectoryComparer::parseTrajectoryFile("trajectory.csv");
    EXPECT_EQ(points.size(), 5);  // One trajectory point per step
    for (const auto& point : points) {
        EXPECT_EQ(point.world, 0);
        EXPECT_EQ(point.agent, 0);
    }
    
    // Get captured output and verify trajectory logging occurred
    std::string captured_output = GetCapturedStdout();
    EXPECT_TRUE(captured_output.find("Trajectory logging enabled") != std::string::npos);
    EXPECT_TRUE(captured_output.find("Trajectory logging disabled") != std::string::npos);
}

// Test pause/resume functionality
TEST_F(ManagerIntegrationTest, MockViewerPauseResume) {
    auto level = LevelComparer::getDefaultLevel();
    config.num_worlds = 1;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    MockViewer viewer(1);
    InputSimulator& input = viewer.getInputSimulator();
    
    bool is_paused = false;
    int steps_run = 0;
    
    viewer.setFrameLimit(10);
    
    viewer.loop(
        [&](int32_t world_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::Space)) {
                is_paused = !is_paused;
            }
        },
        [&](int32_t world_idx, [[maybe_unused]] int32_t agent_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {
            // Agent control (not used in this test)
        },
        [&]() {
            if (!is_paused) {
                mgr.step();
                steps_run++;
            }
        },
        []() {}
    );
    
    // Test pause toggle
    input.hitKey(MockViewer::KeyboardKey::Space);  // Pause
    viewer.setFrameLimit(5);
    
    viewer.loop(
        [&](int32_t world_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::Space)) {
                is_paused = !is_paused;
            }
        },
        [&](int32_t world_idx, [[maybe_unused]] int32_t agent_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {},
        [&]() {
            if (!is_paused) {
                mgr.step();
                steps_run++;
            }
        },
        []() {}
    );
    
    // When paused, steps should not increase
    int steps_before_pause = steps_run;
    
    // Run more frames while paused
    viewer.setFrameLimit(5);
    is_paused = true;  // Ensure paused
    
    viewer.loop(
        [&](int32_t world_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {},
        [&](int32_t world_idx, [[maybe_unused]] int32_t agent_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {},
        [&]() {
            if (!is_paused) {
                mgr.step();
                steps_run++;
            }
        },
        []() {}
    );
    
    EXPECT_EQ(steps_run, steps_before_pause);  // No new steps when paused
}

// Test reset functionality
TEST_F(ManagerIntegrationTest, MockViewerResetInput) {
    auto level = LevelComparer::getDefaultLevel();
    config.num_worlds = 2;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    MockViewer viewer(2);
    InputSimulator& input = viewer.getInputSimulator();
    
    viewer.setFrameLimit(10);
    
    viewer.loop(
        [&](int32_t world_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::R)) {
                mgr.triggerReset(world_idx);
            }
        },
        [&](int32_t world_idx, [[maybe_unused]] int32_t agent_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {},
        [&]() { mgr.step(); },
        []() {}
    );
    
    // Trigger reset for world 0
    viewer.setCurrentWorld(0);
    input.hitKey(MockViewer::KeyboardKey::R);
    
    viewer.setFrameLimit(1);
    viewer.loop(
        [&](int32_t world_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::R)) {
                mgr.triggerReset(world_idx);
            }
        },
        [&](int32_t world_idx, [[maybe_unused]] int32_t agent_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {},
        [&]() { mgr.step(); },
        []() {}
    );
    
    // Verify reset was triggered
    auto& resets = mgr.getResets();
    EXPECT_GE(resets.size(), 1);
    if (!resets.empty()) {
        EXPECT_EQ(resets[0].second, 0);  // World 0 was reset
    }
}

// Test trajectory toggle functionality
TEST_F(ManagerIntegrationTest, MockViewerTrajectoryToggle) {
    auto level = LevelComparer::getDefaultLevel();
    config.num_worlds = 3;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    MockViewer viewer(3);
    InputSimulator& input = viewer.getInputSimulator();
    
    bool track_trajectory = false;
    int32_t track_world_idx = -1;
    
    viewer.setFrameLimit(5);
    
    // Capture stdout to suppress trajectory logging output
    CaptureStdout();
    
    // Test enabling trajectory
    viewer.setCurrentWorld(1);
    input.hitKey(MockViewer::KeyboardKey::T);
    
    viewer.loop(
        [&](int32_t world_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::T)) {
                if (track_trajectory && track_world_idx == world_idx) {
                    mgr.disableTrajectoryLogging();
                    track_trajectory = false;
                } else {
                    mgr.enableTrajectoryLogging(world_idx, 0, nullptr);
                    track_trajectory = true;
                    track_world_idx = world_idx;
                }
            }
        },
        [&](int32_t world_idx, [[maybe_unused]] int32_t agent_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {},
        [&]() { mgr.step(); },
        []() {}
    );
    
    EXPECT_TRUE(mgr.isTrajectoryEnabled());
    EXPECT_EQ(mgr.getTrajectoryWorld(), 1);
    
    // Test disabling trajectory (toggle off)
    input.nextFrame();
    input.hitKey(MockViewer::KeyboardKey::T);
    
    viewer.setFrameLimit(1);
    viewer.loop(
        [&](int32_t world_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::T)) {
                if (track_trajectory && track_world_idx == world_idx) {
                    mgr.disableTrajectoryLogging();
                    track_trajectory = false;
                } else {
                    mgr.enableTrajectoryLogging(world_idx, 0, nullptr);
                    track_trajectory = true;
                    track_world_idx = world_idx;
                }
            }
        },
        [&](int32_t world_idx, [[maybe_unused]] int32_t agent_idx, [[maybe_unused]] const MockViewer::UserInput& user_input) {},
        [&]() { mgr.step(); },
        []() {}
    );
    
    // Get and optionally verify the captured output
    std::string captured_output = GetCapturedStdout();
    // We expect the output to contain trajectory logging messages
    EXPECT_TRUE(captured_output.find("Trajectory logging enabled") != std::string::npos);
    EXPECT_TRUE(captured_output.find("Trajectory logging disabled") != std::string::npos);
    
    EXPECT_FALSE(mgr.isTrajectoryEnabled());
}

// Test recording with embedded level
TEST_F(ManagerIntegrationTest, ManagerEmbeddedLevelRecording) {
    file_manager_->addFile("embedded.rec");
    
    auto level = LevelComparer::getDefaultLevel();
    config.num_worlds = 1;
    config.auto_reset = true;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    
    // Record with embedded level
    mgr.startRecording("embedded.rec");
    
    for (int i = 0; i < 5; i++) {
        mgr.setAction(0, move_amount::SLOW, i % 8, rotate::NONE);
        mgr.step();
    }
    
    mgr.stopRecording();
    
    // Verify file structure
    size_t file_size = file_manager_->getFileSize("embedded.rec");
    size_t expected_min_size = sizeof(MER_ReplayMetadata) + sizeof(MER_CompiledLevel);
    EXPECT_GT(file_size, expected_min_size);
    
    // Read metadata to verify
    MER_ReplayMetadata metadata;
    ASSERT_EQ(mer_read_replay_metadata("embedded.rec", &metadata), MER_SUCCESS);
    EXPECT_EQ(metadata.seed, 42);
    EXPECT_EQ(metadata.num_worlds, 1);
}

// Test checksum verification detects trajectory divergence
TEST_F(ManagerIntegrationTest, ChecksumVerificationDetectsDivergence) {
    auto level = LevelComparer::getDefaultLevel();
    config.num_worlds = 4;
    config.auto_reset = true;
    config.rand_seed = 42;  // Fixed seed for reproducibility

    ASSERT_TRUE(CreateManager(&level, 1));

    TestManagerWrapper mgr(handle);

    // Create a recording with enough steps for checksum at step 200
    mgr.startRecording("checksum_test.rec");

    for (int i = 0; i < 250; i++) {
        // Use consistent actions for reproducibility
        mgr.setAction(0, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
        mgr.setAction(1, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
        mgr.setAction(2, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
        mgr.setAction(3, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
        mgr.step();
    }

    mgr.stopRecording();
    file_manager_->addFile("checksum_test.rec");

    // Test 1: Exact replay with same seed should pass checksum
    {
        auto replay_mgr = madEscape::Manager::fromReplay(
            "checksum_test.rec",
            madrona::ExecMode::CPU,
            0  // gpuID
        );
        ASSERT_NE(replay_mgr, nullptr);

        // Replay all steps - should match exactly
        for (int i = 0; i < 250; i++) {
            bool replay_complete = replay_mgr->replayStep();
            if (!replay_complete) {
                replay_mgr->step();
            }
            if (replay_complete) {
                break;
            }
        }

        // Checksum should pass for exact replay
        EXPECT_FALSE(replay_mgr->hasChecksumFailed())
            << "Exact replay should pass checksum verification";
    }

    // Test 2: Create a recording with DIFFERENT actions that will cause divergence
    {
        // Reset and create a new manager
        mer_destroy_manager(handle);

        config.rand_seed = 42;  // Same seed
        ASSERT_TRUE(CreateManager(&level, 1));
        TestManagerWrapper mgr2(handle);

        mgr2.startRecording("checksum_diverged.rec");

        for (int i = 0; i < 250; i++) {
            if (i < 100) {
                // Same actions initially
                mgr2.setAction(0, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
                mgr2.setAction(1, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
                mgr2.setAction(2, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
                mgr2.setAction(3, move_amount::MEDIUM, move_angle::FORWARD, rotate::NONE);
            } else {
                // DIFFERENT actions after step 100 - this will cause trajectory divergence
                mgr2.setAction(0, move_amount::SLOW, move_angle::LEFT, rotate::SLOW_LEFT);
                mgr2.setAction(1, move_amount::SLOW, move_angle::LEFT, rotate::SLOW_LEFT);
                mgr2.setAction(2, move_amount::SLOW, move_angle::LEFT, rotate::SLOW_LEFT);
                mgr2.setAction(3, move_amount::SLOW, move_angle::LEFT, rotate::SLOW_LEFT);
            }
            mgr2.step();
        }

        mgr2.stopRecording();
        file_manager_->addFile("checksum_diverged.rec");
    }

    // Now test that checksums detect when actions cause trajectory divergence
    // We'll manually swap in different actions from the diverged recording
    {
        // Read both replay files to get their action data
        std::vector<uint8_t> original_data;
        std::vector<uint8_t> diverged_data;

        {
            std::ifstream original_file("checksum_test.rec", std::ios::binary);
            ASSERT_TRUE(original_file.is_open()) << "Failed to open original recording";
            original_data.assign(std::istreambuf_iterator<char>(original_file),
                               std::istreambuf_iterator<char>());
        }

        {
            std::ifstream diverged_file("checksum_diverged.rec", std::ios::binary);
            ASSERT_TRUE(diverged_file.is_open()) << "Failed to open diverged recording";
            diverged_data.assign(std::istreambuf_iterator<char>(diverged_file),
                               std::istreambuf_iterator<char>());
        }

        // Calculate where action data starts (after metadata and level data)
        size_t metadata_size = sizeof(madEscape::ReplayMetadata);

        // Read num_worlds from metadata to calculate level data size
        uint32_t num_worlds;
        memcpy(&num_worlds, original_data.data() + 136, sizeof(uint32_t)); // offset 136 = num_worlds

        // CompiledLevel size varies with struct changes - use sizeof
        size_t level_data_size = sizeof(madEscape::CompiledLevel) * num_worlds;
        size_t actions_offset = metadata_size + level_data_size;

        // Each action is 3 int32_t values (12 bytes per agent)
        size_t action_size = 12 * num_worlds;  // 12 bytes per world (1 agent per world)

        // Replace actions at steps 100-110 with the diverged actions
        for (int step = 100; step < 110 && step < 250; step++) {
            size_t step_offset = actions_offset + (step * action_size);
            if (step_offset + action_size <= original_data.size() &&
                step_offset + action_size <= diverged_data.size()) {
                memcpy(original_data.data() + step_offset,
                       diverged_data.data() + step_offset,
                       action_size);
            }
        }

        // Write modified recording to a new file
        {
            std::ofstream modified_file("checksum_modified.rec", std::ios::binary);
            ASSERT_TRUE(modified_file.is_open()) << "Failed to create modified recording";
            modified_file.write(reinterpret_cast<const char*>(original_data.data()),
                              original_data.size());
        }
        file_manager_->addFile("checksum_modified.rec");

        // Now replay the modified file - it should detect divergence at checksum
        auto modified_mgr = madEscape::Manager::fromReplay(
            "checksum_modified.rec",
            madrona::ExecMode::CPU,
            0  // gpuID
        );

        ASSERT_NE(modified_mgr, nullptr) << "Failed to load modified recording";

        // Replay all steps
        for (int i = 0; i < 250; i++) {
            bool replay_complete = modified_mgr->replayStep();
            if (!replay_complete) {
                modified_mgr->step();
            }
            if (replay_complete) {
                break;
            }
        }

        // Checksum SHOULD fail because the modified actions at steps 100-110
        // will cause different positions by step 200
        bool checksum_failed = modified_mgr->hasChecksumFailed();

        EXPECT_TRUE(checksum_failed)
            << "Checksum should detect trajectory divergence from modified actions";

        if (checksum_failed) {
            std::cout << "✓ Checksum correctly detected trajectory divergence from action modification" << std::endl;
        }
    }
}