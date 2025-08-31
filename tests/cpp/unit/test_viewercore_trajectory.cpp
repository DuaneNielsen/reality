#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "mock_components.hpp"
#include "mgr.hpp"
#include "viewer_core.hpp"
#include "test_level_helper.hpp"
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <limits>
#include <cmath>

// For capturing stdout/stderr output in tests
using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

using namespace madEscape;

// Test fixture for ViewerCore trajectory verification
class ViewerCoreTrajectoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clean up any previous test files
        cleanupTestFiles();
    }
    
    void TearDown() override {
        cleanupTestFiles();
    }
    
    void cleanupTestFiles() {
        // Remove test recording and trajectory files
        std::vector<std::string> files_to_remove = {
            "test_viewercore_recording.rec",
            "trajectory_record.csv",
            "trajectory_replay.csv",
            "test_state_recording.rec"
        };
        
        for (const auto& file : files_to_remove) {
            if (std::filesystem::exists(file)) {
                std::filesystem::remove(file);
            }
        }
    }
    
    // Helper to load the test level with clear path for agent
    CompiledLevel loadTestLevel() {
        // Start with the embedded default level
        CompiledLevel level = DefaultLevelProvider::GetDefaultLevel();
        
        // Clear objects in front of the agent to prevent collisions
        // Agent spawns at spawn_x[0], spawn_y[0] and moves forward (positive Y)
        float agent_x = level.spawn_x[0];
        float agent_y = level.spawn_y[0];
        float clear_radius = 10.0f; // Clear +-10 units in X around the agent's path
        
        // Clear tiles that might be in the agent's forward path
        for (int32_t i = 0; i < level.num_tiles && i < 1024; i++) {
            // Check if tile is in the agent's potential path (forward movement area)
            if (level.tile_y[i] > agent_y - 2.0f) { // Tiles ahead of spawn
                if (std::abs(level.tile_x[i] - agent_x) < clear_radius) {
                    // Clear this tile by setting it to empty (object_id = 0)
                    level.object_ids[i] = 0;
                    level.tile_entity_type[i] = 0;
                }
            }
        }
        
        // Clear any per-tile randomization for deterministic behavior
        for (int32_t i = 0; i < level.num_tiles && i < 1024; i++) {
            level.tile_rand_x[i] = 0.0f;
            level.tile_rand_y[i] = 0.0f;
            level.tile_rand_z[i] = 0.0f;
            level.tile_rand_rot_z[i] = 0.0f;
            level.tile_rand_scale_x[i] = 0.0f;
            level.tile_rand_scale_y[i] = 0.0f;
            level.tile_rand_scale_z[i] = 0.0f;
        }
        
        return level;
    }
};

TEST_F(ViewerCoreTrajectoryTest, DeterministicReplayWithTrajectory) {
    // Capture stdout to suppress trajectory logging output
    CaptureStdout();
    
    // Phase 1: Setup and Recording
    // =============================
    
    // 1. Load Level
    CompiledLevel test_level = loadTestLevel();
    std::vector<std::optional<CompiledLevel>> per_world_levels;
    per_world_levels.push_back(test_level);
    
    // 2. Create Manager Instance for recording
    Manager::Config mgr_config_record;
    mgr_config_record.execMode = madrona::ExecMode::CPU;
    mgr_config_record.gpuID = 0;
    mgr_config_record.numWorlds = 1;
    mgr_config_record.randSeed = 42;
    mgr_config_record.autoReset = true;
    mgr_config_record.enableBatchRenderer = false;
    mgr_config_record.batchRenderViewWidth = 64;
    mgr_config_record.batchRenderViewHeight = 64;
    mgr_config_record.extRenderAPI = nullptr;
    mgr_config_record.extRenderDev = nullptr;
    mgr_config_record.enableTrajectoryTracking = false;
    mgr_config_record.perWorldCompiledLevels = per_world_levels;
    
    Manager mgr_record(mgr_config_record);
    
    // 3. Create ViewerCore in Recording Mode
    ViewerCore::Config record_config;
    record_config.num_worlds = 1;
    record_config.rand_seed = 42;
    record_config.auto_reset = true;
    record_config.load_path = "";  // Level already loaded in Manager
    record_config.record_path = "test_viewercore_recording.rec";
    record_config.replay_path = "";
    
    ViewerCore core_record(record_config, &mgr_record);
    
    // 4. Start Recording with Trajectory Tracking
    core_record.startRecording("test_viewercore_recording.rec");
    
    // Enable trajectory tracking to "trajectory_record.csv"
    mgr_record.enableTrajectoryLogging(0, 0, "trajectory_record.csv");
    
    // Verify we're in paused recording state
    auto state = core_record.getFrameState();
    EXPECT_TRUE(state.is_paused) << "Recording should start paused";
    EXPECT_TRUE(state.is_recording) << "Should be in recording mode";
    
    // 5. Unpause and Simulate Movement
    // Send SPACE key to unpause
    ViewerCore::InputEvent unpause_event;
    unpause_event.type = ViewerCore::InputEvent::KeyHit;
    unpause_event.key = ViewerCore::InputEvent::Space;
    core_record.handleInput(0, unpause_event);
    
    // Verify unpaused
    state = core_record.getFrameState();
    EXPECT_FALSE(state.is_paused) << "Should be unpaused after SPACE";
    
    // Hold W key for forward movement for 100 frames
    ViewerCore::InputEvent w_press;
    w_press.type = ViewerCore::InputEvent::KeyPress;
    w_press.key = ViewerCore::InputEvent::W;
    
    for (int frame = 0; frame < 100; frame++) {
        // Send W key press event
        core_record.handleInput(0, w_press);
        
        // Update frame actions for world 0, agent 0
        core_record.updateFrameActions(0, 0);
        
        // Step the simulation
        core_record.stepSimulation();
    }
    
    // 6. Stop Recording
    core_record.stopRecording();
    
    // Verify recording file was created
    ASSERT_TRUE(std::filesystem::exists("test_viewercore_recording.rec")) 
        << "Recording file should be created";
    
    // Verify trajectory file was created
    ASSERT_TRUE(std::filesystem::exists("trajectory_record.csv")) 
        << "Recording trajectory file should be created";
    
    // Small delay to ensure files are fully written
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    
    // Phase 2: Replay with Trajectory Tracking
    // =========================================
    
    // 1. Extract the embedded level from the replay file
    auto embedded_level = madrona::escape_room::ReplayLoader::loadEmbeddedLevel("test_viewercore_recording.rec");
    ASSERT_TRUE(embedded_level.has_value()) << "Failed to load embedded level from replay file";
    
    // 2. Create Manager with the extracted level
    std::vector<std::optional<CompiledLevel>> replay_levels;
    replay_levels.push_back(embedded_level.value());
    
    Manager::Config mgr_config_replay;
    mgr_config_replay.execMode = madrona::ExecMode::CPU;
    mgr_config_replay.gpuID = 0;
    mgr_config_replay.numWorlds = 1;
    mgr_config_replay.randSeed = 42;  // Same seed
    mgr_config_replay.autoReset = true;
    mgr_config_replay.enableBatchRenderer = false;
    mgr_config_replay.batchRenderViewWidth = 64;
    mgr_config_replay.batchRenderViewHeight = 64;
    mgr_config_replay.extRenderAPI = nullptr;
    mgr_config_replay.extRenderDev = nullptr;
    mgr_config_replay.enableTrajectoryTracking = false;
    mgr_config_replay.perWorldCompiledLevels = replay_levels;
    
    Manager mgr_replay(mgr_config_replay);
    
    // 2. Create ViewerCore in Replay Mode
    ViewerCore::Config replay_config;
    replay_config.num_worlds = 1;
    replay_config.rand_seed = 42;
    replay_config.auto_reset = true;
    replay_config.load_path = "";
    replay_config.record_path = "";
    replay_config.replay_path = "test_viewercore_recording.rec";
    
    ViewerCore core_replay(replay_config, &mgr_replay);
    
    // 3. Load Replay and Enable Tracking
    core_replay.loadReplay("test_viewercore_recording.rec");
    
    // Enable trajectory tracking to different file
    mgr_replay.enableTrajectoryLogging(0, 0, "trajectory_replay.csv");
    
    // Verify replay state (should NOT be paused)
    state = core_replay.getFrameState();
    EXPECT_FALSE(state.is_paused) << "Replay should not start paused";
    EXPECT_TRUE(state.has_replay) << "Should have replay loaded";
    
    // 4. Step Through Replay
    for (int frame = 0; frame < 100; frame++) {
        core_replay.stepSimulation();
        
        // Check if replay finished early
        state = core_replay.getFrameState();
        if (state.should_exit) {
            break;
        }
    }
    
    // 5. Verify Replay Completion
    state = core_replay.getFrameState();
    // Note: should_exit might not be set if we don't replay all frames
    // The important thing is that we replayed the same number of frames
    
    // Verify replay trajectory file was created
    ASSERT_TRUE(std::filesystem::exists("trajectory_replay.csv")) 
        << "Replay trajectory file should be created";
    
    // Small delay to ensure files are fully written
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    
    // Phase 3: Trajectory Verification
    // =================================
    
    // 1. Load Both CSV Files
    auto record_trajectory = TrajectoryComparer::parseTrajectoryFile("trajectory_record.csv");
    auto replay_trajectory = TrajectoryComparer::parseTrajectoryFile("trajectory_replay.csv");
    
    // 2. Compare Trajectories
    // Both should have 100 points (one per frame)
    EXPECT_EQ(record_trajectory.size(), 100) 
        << "Recording trajectory should have 100 points";
    EXPECT_EQ(replay_trajectory.size(), 100) 
        << "Replay trajectory should have 100 points";
    
    // Verify trajectories match
    bool trajectories_match = TrajectoryComparer::compareTrajectories(
        "trajectory_record.csv", 
        "trajectory_replay.csv", 
        0.001f  // Small epsilon for floating point comparison
    );
    
    EXPECT_TRUE(trajectories_match) 
        << "Recording and replay trajectories should be identical (deterministic replay)";
    
    // 3. Verify Movement Characteristics
    if (!record_trajectory.empty()) {
        // Get first and last points
        const auto& first_point = record_trajectory.front();
        const auto& last_point = record_trajectory.back();
        
        // Agent should have moved forward (Y should increase when moving forward)
        // Note: Coordinate system may vary, adjust as needed
        EXPECT_GT(last_point.y, first_point.y) 
            << "Agent should have moved forward (Y increased)";
        
        // Rotation should stay relatively constant (no rotation input given)
        float rotation_change = std::abs(last_point.rotation - first_point.rotation);
        EXPECT_LT(rotation_change, 1.0f) 
            << "Rotation should remain nearly constant";
        
        // Progress should increase
        EXPECT_GT(last_point.progress, first_point.progress) 
            << "Progress should increase during movement";
        
        // Verify monotonic progress increase
        float prev_progress = -std::numeric_limits<float>::max();  // Start with minimum possible value
        for (const auto& point : record_trajectory) {
            EXPECT_GE(point.progress, prev_progress) 
                << "Progress should increase monotonically";
            prev_progress = point.progress;
        }
    }
    
    // Additional verification: Check that replay matches recording exactly
    if (record_trajectory.size() == replay_trajectory.size()) {
        for (size_t i = 0; i < record_trajectory.size(); i++) {
            const auto& rec_point = record_trajectory[i];
            const auto& rep_point = replay_trajectory[i];
            
            EXPECT_EQ(rec_point.step, rep_point.step) 
                << "Step numbers should match at index " << i;
            EXPECT_EQ(rec_point.world, rep_point.world) 
                << "World indices should match at index " << i;
            EXPECT_EQ(rec_point.agent, rep_point.agent) 
                << "Agent indices should match at index " << i;
            
            // Use NEAR for floating point comparison
            EXPECT_NEAR(rec_point.x, rep_point.x, 0.001f) 
                << "X positions should match at step " << i;
            EXPECT_NEAR(rec_point.y, rep_point.y, 0.001f) 
                << "Y positions should match at step " << i;
            EXPECT_NEAR(rec_point.z, rep_point.z, 0.001f) 
                << "Z positions should match at step " << i;
            EXPECT_NEAR(rec_point.rotation, rep_point.rotation, 0.001f) 
                << "Rotations should match at step " << i;
            EXPECT_NEAR(rec_point.progress, rep_point.progress, 0.001f) 
                << "Progress should match at step " << i;
        }
    }
    
    // Get captured output and verify trajectory logging occurred
    std::string captured_output = GetCapturedStdout();
    EXPECT_TRUE(captured_output.find("Trajectory logging enabled") != std::string::npos);
}

// Test ViewerCore state machine transitions
TEST_F(ViewerCoreTrajectoryTest, StateMachineTransitions) {
    // Capture stdout to suppress any logging output
    CaptureStdout();
    
    // Create minimal manager for testing state transitions
    CompiledLevel test_level = loadTestLevel();
    std::vector<std::optional<CompiledLevel>> per_world_levels;
    per_world_levels.push_back(test_level);
    
    Manager::Config mgr_config;
    mgr_config.execMode = madrona::ExecMode::CPU;
    mgr_config.gpuID = 0;
    mgr_config.numWorlds = 1;
    mgr_config.randSeed = 42;
    mgr_config.autoReset = true;
    mgr_config.enableBatchRenderer = false;
    mgr_config.batchRenderViewWidth = 64;
    mgr_config.batchRenderViewHeight = 64;
    mgr_config.extRenderAPI = nullptr;
    mgr_config.extRenderDev = nullptr;
    mgr_config.enableTrajectoryTracking = false;
    mgr_config.perWorldCompiledLevels = per_world_levels;
    
    Manager mgr(mgr_config);
    
    ViewerCore::Config config;
    config.num_worlds = 1;
    config.rand_seed = 42;
    config.auto_reset = true;
    config.load_path = "";
    config.record_path = "";  // Don't set record_path initially to start in Idle
    config.replay_path = "";
    
    ViewerCore core(config, &mgr);
    
    // Test recording state transitions
    const auto& state_machine = core.getStateMachine();
    
    // Initial state should be Idle
    EXPECT_EQ(state_machine.getState(), RecordReplayStateMachine::Idle);
    
    // Start recording - should be paused
    core.startRecording("test_state_recording.rec");
    EXPECT_EQ(state_machine.getState(), RecordReplayStateMachine::RecordingPaused);
    EXPECT_TRUE(state_machine.isPaused());
    EXPECT_TRUE(state_machine.isRecording());
    
    // Unpause with SPACE key
    ViewerCore::InputEvent space_event;
    space_event.type = ViewerCore::InputEvent::KeyHit;
    space_event.key = ViewerCore::InputEvent::Space;
    core.handleInput(0, space_event);
    
    // Check state after unpause
    EXPECT_EQ(state_machine.getState(), RecordReplayStateMachine::Recording);
    EXPECT_FALSE(state_machine.isPaused());
    EXPECT_TRUE(state_machine.isRecording());
    
    auto frame_state = core.getFrameState();
    EXPECT_FALSE(frame_state.is_paused);
    EXPECT_TRUE(frame_state.is_recording);
    
    // Stop recording
    core.stopRecording();
    EXPECT_EQ(state_machine.getState(), RecordReplayStateMachine::Idle);
    EXPECT_FALSE(state_machine.isRecording());
    EXPECT_FALSE(state_machine.isPaused());
    
    // Get captured output - this test may have minimal logging
    std::string captured_output = GetCapturedStdout();
    // No specific logging assertions needed for state machine test
}

// Test action computation from input
TEST_F(ViewerCoreTrajectoryTest, ActionComputationFromInput) {
    // Create minimal manager
    CompiledLevel test_level = loadTestLevel();
    std::vector<std::optional<CompiledLevel>> per_world_levels;
    per_world_levels.push_back(test_level);
    per_world_levels.push_back(test_level);
    
    Manager::Config mgr_config;
    mgr_config.execMode = madrona::ExecMode::CPU;
    mgr_config.gpuID = 0;
    mgr_config.numWorlds = 2;  // Test with 2 worlds
    mgr_config.randSeed = 42;
    mgr_config.autoReset = true;
    mgr_config.enableBatchRenderer = false;
    mgr_config.batchRenderViewWidth = 64;
    mgr_config.batchRenderViewHeight = 64;
    mgr_config.extRenderAPI = nullptr;
    mgr_config.extRenderDev = nullptr;
    mgr_config.enableTrajectoryTracking = false;
    mgr_config.perWorldCompiledLevels = per_world_levels;
    
    Manager mgr(mgr_config);
    
    ViewerCore::Config config;
    config.num_worlds = 2;
    config.rand_seed = 42;
    config.auto_reset = true;
    config.load_path = "";
    config.record_path = "";
    config.replay_path = "";
    
    ViewerCore core(config, &mgr);
    
    // Test W key -> forward movement
    ViewerCore::InputEvent w_press;
    w_press.type = ViewerCore::InputEvent::KeyPress;
    w_press.key = ViewerCore::InputEvent::W;
    
    core.handleInput(0, w_press);
    core.updateFrameActions(0, 0);
    
    auto frame_state = core.getFrameState();
    // Actions for world 0 should show forward movement
    // move_amount > 0, move_angle = 0 (forward)
    EXPECT_GT(frame_state.frame_actions[0], 0) << "Move amount should be > 0 when W pressed";
    EXPECT_EQ(frame_state.frame_actions[1], 0) << "Move angle should be 0 (forward)";
    
    // Test A+W -> diagonal movement
    ViewerCore::InputEvent a_press;
    a_press.type = ViewerCore::InputEvent::KeyPress;
    a_press.key = ViewerCore::InputEvent::A;
    
    core.handleInput(0, a_press);
    core.updateFrameActions(0, 0);
    
    frame_state = core.getFrameState();
    EXPECT_GT(frame_state.frame_actions[0], 0) << "Move amount should be > 0 when moving diagonally";
    EXPECT_EQ(frame_state.frame_actions[1], 7) << "Move angle should be 7 (forward-left)";
    
    // Test Q key -> rotation left
    ViewerCore::InputEvent q_press;
    q_press.type = ViewerCore::InputEvent::KeyPress;
    q_press.key = ViewerCore::InputEvent::Q;
    
    // Release W and A first
    ViewerCore::InputEvent w_release;
    w_release.type = ViewerCore::InputEvent::KeyRelease;
    w_release.key = ViewerCore::InputEvent::W;
    core.handleInput(1, w_release);
    
    ViewerCore::InputEvent a_release;
    a_release.type = ViewerCore::InputEvent::KeyRelease;
    a_release.key = ViewerCore::InputEvent::A;
    core.handleInput(1, a_release);
    
    core.handleInput(1, q_press);  // Test on world 1
    core.updateFrameActions(1, 0);
    
    frame_state = core.getFrameState();
    // Actions for world 1 (index 3,4,5 since each world has 3 action dims)
    EXPECT_EQ(frame_state.frame_actions[3], 0) << "Move amount should be 0 when only rotating";
    EXPECT_LT(frame_state.frame_actions[5], 2) << "Q key (left) uses non-standard encoding: lower values for left turn";
    
    // Test E key -> rotation right
    ViewerCore::InputEvent q_release;
    q_release.type = ViewerCore::InputEvent::KeyRelease;
    q_release.key = ViewerCore::InputEvent::Q;
    core.handleInput(1, q_release);
    
    ViewerCore::InputEvent e_press;
    e_press.type = ViewerCore::InputEvent::KeyPress;
    e_press.key = ViewerCore::InputEvent::E;
    
    core.handleInput(1, e_press);
    core.updateFrameActions(1, 0);
    
    frame_state = core.getFrameState();
    EXPECT_GT(frame_state.frame_actions[5], 2) << "E key (right) uses non-standard encoding: higher values for right turn";
}

// Test trajectory tracking toggle
TEST_F(ViewerCoreTrajectoryTest, TrajectoryTrackingToggle) {
    // Capture stdout to suppress trajectory logging output
    CaptureStdout();
    
    // Create minimal manager
    CompiledLevel test_level = loadTestLevel();
    std::vector<std::optional<CompiledLevel>> per_world_levels;
    per_world_levels.push_back(test_level);
    
    Manager::Config mgr_config;
    mgr_config.execMode = madrona::ExecMode::CPU;
    mgr_config.gpuID = 0;
    mgr_config.numWorlds = 1;
    mgr_config.randSeed = 42;
    mgr_config.autoReset = true;
    mgr_config.enableBatchRenderer = false;
    mgr_config.batchRenderViewWidth = 64;
    mgr_config.batchRenderViewHeight = 64;
    mgr_config.extRenderAPI = nullptr;
    mgr_config.extRenderDev = nullptr;
    mgr_config.enableTrajectoryTracking = false;
    mgr_config.perWorldCompiledLevels = per_world_levels;
    
    Manager mgr(mgr_config);
    
    ViewerCore::Config config;
    config.num_worlds = 1;
    config.rand_seed = 42;
    config.auto_reset = true;
    config.load_path = "";
    config.record_path = "";
    config.replay_path = "";
    
    ViewerCore core(config, &mgr);
    
    // Initially should not be tracking
    EXPECT_FALSE(core.isTrackingTrajectory(0));
    
    // Toggle on
    core.toggleTrajectoryTracking(0);
    EXPECT_TRUE(core.isTrackingTrajectory(0));
    
    // Toggle off
    core.toggleTrajectoryTracking(0);
    EXPECT_FALSE(core.isTrackingTrajectory(0));
    
    // Get captured output and verify trajectory logging occurred
    std::string captured_output = GetCapturedStdout();
    EXPECT_TRUE(captured_output.find("Trajectory logging enabled") != std::string::npos);
    EXPECT_TRUE(captured_output.find("Trajectory logging disabled") != std::string::npos);
}

// Test that trajectory points match the number of recorded frames
TEST_F(ViewerCoreTrajectoryTest, TrajectoryPointsMatchRecordedFrames) {
    // Capture stdout to suppress trajectory logging output
    CaptureStdout();
    
    // This test verifies that the number of trajectory points written
    // matches the number of frames recorded in the action file.
    // This addresses the bug where 150 frames are saved but only 75 trajectory points.
    
    const int NUM_FRAMES_TO_RECORD = 150;  // Test with 150 frames specifically
    
    // Clean up any previous test files
    std::vector<std::string> files_to_clean = {
        "test_frame_count.rec",
        "trajectory_frame_count.csv"
    };
    for (const auto& file : files_to_clean) {
        if (std::filesystem::exists(file)) {
            std::filesystem::remove(file);
        }
    }
    
    // Load test level
    CompiledLevel test_level = loadTestLevel();
    std::vector<std::optional<CompiledLevel>> per_world_levels;
    per_world_levels.push_back(test_level);
    
    // Create Manager for recording
    Manager::Config mgr_config;
    mgr_config.execMode = madrona::ExecMode::CPU;
    mgr_config.gpuID = 0;
    mgr_config.numWorlds = 1;
    mgr_config.randSeed = 12345;  // Different seed for variety
    mgr_config.autoReset = true;
    mgr_config.enableBatchRenderer = false;
    mgr_config.batchRenderViewWidth = 64;
    mgr_config.batchRenderViewHeight = 64;
    mgr_config.extRenderAPI = nullptr;
    mgr_config.extRenderDev = nullptr;
    mgr_config.enableTrajectoryTracking = false;
    mgr_config.perWorldCompiledLevels = per_world_levels;
    
    Manager mgr(mgr_config);
    
    // Create ViewerCore in recording mode
    ViewerCore::Config record_config;
    record_config.num_worlds = 1;
    record_config.rand_seed = 12345;
    record_config.auto_reset = true;
    record_config.load_path = "";
    record_config.record_path = "test_frame_count.rec";
    record_config.replay_path = "";
    
    ViewerCore core(record_config, &mgr);
    
    // Start recording with trajectory tracking
    core.startRecording("test_frame_count.rec");
    mgr.enableTrajectoryLogging(0, 0, "trajectory_frame_count.csv");
    
    // Unpause recording
    ViewerCore::InputEvent unpause_event;
    unpause_event.type = ViewerCore::InputEvent::KeyHit;
    unpause_event.key = ViewerCore::InputEvent::Space;
    core.handleInput(0, unpause_event);
    
    // Verify unpaused
    auto state = core.getFrameState();
    ASSERT_FALSE(state.is_paused) << "Should be unpaused after SPACE";
    
    // Record exactly NUM_FRAMES_TO_RECORD frames with varied movement
    ViewerCore::InputEvent w_press;
    w_press.type = ViewerCore::InputEvent::KeyPress;
    w_press.key = ViewerCore::InputEvent::W;
    
    ViewerCore::InputEvent d_press;
    d_press.type = ViewerCore::InputEvent::KeyPress;
    d_press.key = ViewerCore::InputEvent::D;
    
    for (int frame = 0; frame < NUM_FRAMES_TO_RECORD; frame++) {
        // Alternate between W and D every 30 frames for variety
        if ((frame / 30) % 2 == 0) {
            core.handleInput(0, w_press);
        } else {
            core.handleInput(0, d_press);
        }
        
        core.updateFrameActions(0, 0);
        core.stepSimulation();
    }
    
    // Stop recording
    core.stopRecording();
    
    // Ensure files are fully written
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Verify both files exist
    ASSERT_TRUE(std::filesystem::exists("test_frame_count.rec")) 
        << "Recording file should exist";
    ASSERT_TRUE(std::filesystem::exists("trajectory_frame_count.csv")) 
        << "Trajectory file should exist";
    
    // Parse the trajectory file and count points
    auto trajectory_points = TrajectoryComparer::parseTrajectoryFile("trajectory_frame_count.csv");
    
    // Count the number of frames in the recording file
    // We can infer this from the replay metadata or by replaying and counting
    auto metadata_opt = Manager::readReplayMetadata("test_frame_count.rec");
    ASSERT_TRUE(metadata_opt.has_value()) << "Should be able to read replay metadata";
    
    // Debug output to understand the discrepancy
    std::cout << "DEBUG: Requested frames: " << NUM_FRAMES_TO_RECORD << std::endl;
    std::cout << "DEBUG: Trajectory points in recording: " << trajectory_points.size() << std::endl;
    
    // The critical assertion: trajectory points should match recorded frames
    EXPECT_EQ(trajectory_points.size(), NUM_FRAMES_TO_RECORD) 
        << "Number of trajectory points (" << trajectory_points.size() 
        << ") should match number of recorded frames (" << NUM_FRAMES_TO_RECORD << ")";
    
    // Additional verification: Check trajectory points have sequential step numbers
    if (!trajectory_points.empty()) {
        for (size_t i = 0; i < trajectory_points.size(); i++) {
            EXPECT_EQ(trajectory_points[i].step, static_cast<int>(i)) 
                << "Trajectory step number should be sequential at index " << i;
        }
        
        // Verify we have exactly the expected range
        EXPECT_EQ(trajectory_points.front().step, 0) 
            << "First trajectory point should be step 0";
        EXPECT_EQ(trajectory_points.back().step, NUM_FRAMES_TO_RECORD - 1) 
            << "Last trajectory point should be step " << (NUM_FRAMES_TO_RECORD - 1);
    }
    
    // Now replay the recording and verify trajectory during replay also matches
    // Extract embedded level for replay
    auto embedded_level = madrona::escape_room::ReplayLoader::loadEmbeddedLevel("test_frame_count.rec");
    ASSERT_TRUE(embedded_level.has_value()) << "Failed to load embedded level from replay";
    
    std::vector<std::optional<CompiledLevel>> replay_levels;
    replay_levels.push_back(embedded_level.value());
    
    // Create new Manager for replay
    Manager::Config replay_mgr_config = mgr_config;
    replay_mgr_config.perWorldCompiledLevels = replay_levels;
    
    Manager mgr_replay(replay_mgr_config);
    
    // Create ViewerCore in replay mode
    ViewerCore::Config replay_config;
    replay_config.num_worlds = 1;
    replay_config.rand_seed = 12345;
    replay_config.auto_reset = true;
    replay_config.load_path = "";
    replay_config.record_path = "";
    replay_config.replay_path = "test_frame_count.rec";
    
    ViewerCore core_replay(replay_config, &mgr_replay);
    
    // Load replay and enable trajectory tracking
    core_replay.loadReplay("test_frame_count.rec");
    mgr_replay.enableTrajectoryLogging(0, 0, "trajectory_replay_count.csv");
    
    // Step through ALL frames in the replay
    int replay_frame_count = 0;
    while (true) {
        core_replay.stepSimulation();
        replay_frame_count++;
        
        auto replay_state = core_replay.getFrameState();
        if (replay_state.should_exit || replay_frame_count >= NUM_FRAMES_TO_RECORD + 10) {
            // Add safety limit to prevent infinite loop
            break;
        }
    }
    
    // Ensure replay trajectory is written
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // Check replay trajectory also has correct number of points
    if (std::filesystem::exists("trajectory_replay_count.csv")) {
        auto replay_trajectory = TrajectoryComparer::parseTrajectoryFile("trajectory_replay_count.csv");
        
        std::cout << "DEBUG: Replay frame count: " << replay_frame_count << std::endl;
        std::cout << "DEBUG: Replay trajectory points: " << replay_trajectory.size() << std::endl;
        
        EXPECT_EQ(replay_trajectory.size(), trajectory_points.size())
            << "Replay trajectory should have same number of points as recording";
        
        // Clean up replay trajectory file
        std::filesystem::remove("trajectory_replay_count.csv");
    }
    
    // Clean up test files
    for (const auto& file : files_to_clean) {
        if (std::filesystem::exists(file)) {
            std::filesystem::remove(file);
        }
    }
    
    // Get captured output and verify trajectory logging occurred
    std::string captured_output = GetCapturedStdout();
    EXPECT_TRUE(captured_output.find("Trajectory logging enabled") != std::string::npos);
}

// Test to diagnose frame count mismatch between recording and replay
TEST_F(ViewerCoreTrajectoryTest, DiagnoseFrameCountMismatch) {
    // Capture stdout to suppress trajectory logging output
    CaptureStdout();
    
    // This test helps diagnose why replay might run more frames than recording
    
    const int NUM_FRAMES = 75;  // Use smaller number for easier debugging
    
    // Clean up test files
    std::vector<std::string> files_to_clean = {
        "diagnose_frames.rec",
        "diagnose_trajectory.csv"
    };
    for (const auto& file : files_to_clean) {
        if (std::filesystem::exists(file)) {
            std::filesystem::remove(file);
        }
    }
    
    // Setup
    CompiledLevel test_level = loadTestLevel();
    std::vector<std::optional<CompiledLevel>> per_world_levels;
    per_world_levels.push_back(test_level);
    
    Manager::Config mgr_config;
    mgr_config.execMode = madrona::ExecMode::CPU;
    mgr_config.gpuID = 0;
    mgr_config.numWorlds = 1;
    mgr_config.randSeed = 999;
    mgr_config.autoReset = true;
    mgr_config.enableBatchRenderer = false;
    mgr_config.batchRenderViewWidth = 64;
    mgr_config.batchRenderViewHeight = 64;
    mgr_config.extRenderAPI = nullptr;
    mgr_config.extRenderDev = nullptr;
    mgr_config.enableTrajectoryTracking = false;
    mgr_config.perWorldCompiledLevels = per_world_levels;
    
    Manager mgr(mgr_config);
    
    ViewerCore::Config config;
    config.num_worlds = 1;
    config.rand_seed = 999;
    config.auto_reset = true;
    config.load_path = "";
    config.record_path = "diagnose_frames.rec";
    config.replay_path = "";
    
    ViewerCore core(config, &mgr);
    
    // Start recording with trajectory
    core.startRecording("diagnose_frames.rec");
    mgr.enableTrajectoryLogging(0, 0, "diagnose_trajectory.csv");
    
    // Unpause
    ViewerCore::InputEvent unpause;
    unpause.type = ViewerCore::InputEvent::KeyHit;
    unpause.key = ViewerCore::InputEvent::Space;
    core.handleInput(0, unpause);
    
    // Record exactly NUM_FRAMES
    std::cout << "\n=== RECORDING PHASE ===" << std::endl;
    for (int i = 0; i < NUM_FRAMES; i++) {
        if (i % 25 == 0) {
            std::cout << "Recording frame " << i << std::endl;
        }
        core.updateFrameActions(0, 0);
        core.stepSimulation();
    }
    std::cout << "Requested " << NUM_FRAMES << " frames to be recorded" << std::endl;
    
    // Stop recording
    core.stopRecording();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Parse trajectory
    auto record_trajectory = TrajectoryComparer::parseTrajectoryFile("diagnose_trajectory.csv");
    std::cout << "Trajectory points saved: " << record_trajectory.size() << std::endl;
    
    // Check metadata
    auto metadata = Manager::readReplayMetadata("diagnose_frames.rec");
    if (metadata.has_value()) {
        std::cout << "Replay metadata reports: " << metadata->num_steps << " steps" << std::endl;
    }
    
    // Now replay
    std::cout << "\n=== REPLAY PHASE ===" << std::endl;
    
    auto embedded_level = madrona::escape_room::ReplayLoader::loadEmbeddedLevel("diagnose_frames.rec");
    ASSERT_TRUE(embedded_level.has_value());
    
    std::vector<std::optional<CompiledLevel>> replay_levels;
    replay_levels.push_back(embedded_level.value());
    
    Manager::Config replay_config = mgr_config;
    replay_config.perWorldCompiledLevels = replay_levels;
    Manager mgr_replay(replay_config);
    
    ViewerCore::Config core_replay_config;
    core_replay_config.num_worlds = 1;
    core_replay_config.rand_seed = 999;
    core_replay_config.auto_reset = true;
    core_replay_config.load_path = "";
    core_replay_config.record_path = "";
    core_replay_config.replay_path = "diagnose_frames.rec";
    
    ViewerCore core_replay(core_replay_config, &mgr_replay);
    core_replay.loadReplay("diagnose_frames.rec");
    mgr_replay.enableTrajectoryLogging(0, 0, "diagnose_replay_trajectory.csv");
    
    // Step through replay and count frames
    int replay_frames = 0;
    while (true) {
        if (replay_frames % 25 == 0) {
            std::cout << "Replaying frame " << replay_frames << std::endl;
        }
        
        core_replay.stepSimulation();
        replay_frames++;
        
        auto state = core_replay.getFrameState();
        if (state.should_exit) {
            std::cout << "Replay finished at frame " << replay_frames << std::endl;
            break;
        }
        
        // Safety limit
        if (replay_frames > NUM_FRAMES + 20) {
            std::cout << "Safety limit reached at frame " << replay_frames << std::endl;
            break;
        }
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Parse replay trajectory
    if (std::filesystem::exists("diagnose_replay_trajectory.csv")) {
        auto replay_trajectory = TrajectoryComparer::parseTrajectoryFile("diagnose_replay_trajectory.csv");
        std::cout << "Replay trajectory points: " << replay_trajectory.size() << std::endl;
        
        // Print first and last few step numbers to understand the pattern
        if (replay_trajectory.size() > 0) {
            std::cout << "First 5 replay steps: ";
            for (size_t i = 0; i < std::min(size_t(5), replay_trajectory.size()); i++) {
                std::cout << replay_trajectory[i].step << " ";
            }
            std::cout << std::endl;
            
            if (replay_trajectory.size() > 5) {
                std::cout << "Last 5 replay steps: ";
                for (size_t i = replay_trajectory.size() - 5; i < replay_trajectory.size(); i++) {
                    std::cout << replay_trajectory[i].step << " ";
                }
                std::cout << std::endl;
            }
        }
        
        std::filesystem::remove("diagnose_replay_trajectory.csv");
    }
    
    std::cout << "\n=== SUMMARY ===" << std::endl;
    std::cout << "Frames requested to record: " << NUM_FRAMES << std::endl;
    std::cout << "Trajectory points in recording: " << record_trajectory.size() << std::endl;
    std::cout << "Frames replayed: " << replay_frames << std::endl;
    
    // The assertion we want to pass
    EXPECT_EQ(record_trajectory.size(), NUM_FRAMES) 
        << "Recording should have exactly the requested number of trajectory points";
    EXPECT_EQ(replay_frames, NUM_FRAMES) 
        << "Replay should run exactly the same number of frames as recorded";
    
    // Cleanup
    for (const auto& file : files_to_clean) {
        if (std::filesystem::exists(file)) {
            std::filesystem::remove(file);
        }
    }
    
    // Get captured output and verify trajectory logging occurred
    std::string captured_output = GetCapturedStdout();
    EXPECT_TRUE(captured_output.find("Trajectory logging enabled") != std::string::npos);
}