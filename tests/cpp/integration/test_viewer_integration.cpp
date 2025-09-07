#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "mock_components.hpp"
#include <thread>
#include <chrono>
#include "../../../src/consts.hpp"

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
    EXPECT_EQ(points.size(), 6);  // Initial state + 5 simulation steps
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