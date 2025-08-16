#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "mock_components.hpp"
#include "test_levels.hpp"
#include <thread>
#include <chrono>

// These tests simulate viewer-like workflows but primarily test Manager API
// Only a few tests actually exercise viewer-specific behavior
class SimulatedViewerWorkflowTest : public ViewerTestBase {
protected:
    void SetUp() override {
        ViewerTestBase::SetUp();
        file_manager_ = std::make_unique<TestFileManager>();
    }
    
    void TearDown() override {
        file_manager_->cleanup();
        ViewerTestBase::TearDown();
    }
    
    // Helper to simulate full viewer session
    void runViewerSession(MockViewer& viewer, TestManagerWrapper& mgr,
                         InputSimulator& input, int num_frames,
                         bool& is_paused, bool& is_recording) {
        viewer.setFrameLimit(num_frames);
        
        std::vector<int32_t> frame_actions;
        if (is_recording) {
            frame_actions.resize(config.num_worlds * 3);
            // Initialize with defaults
            for (uint32_t i = 0; i < config.num_worlds; i++) {
                frame_actions[i * 3] = 0;      // move_amount
                frame_actions[i * 3 + 1] = 0;  // move_angle
                frame_actions[i * 3 + 2] = 2;  // rotate
            }
        }
        
        viewer.loop(
            [&](int32_t world_idx, const MockViewer::UserInput& user_input) {
                // Handle world-level controls
                if (user_input.keyHit(MockViewer::KeyboardKey::R)) {
                    mgr.triggerReset(world_idx);
                }
                
                if (user_input.keyHit(MockViewer::KeyboardKey::T)) {
                    if (mgr.isTrajectoryEnabled()) {
                        mgr.disableTrajectoryLogging();
                    } else {
                        mgr.enableTrajectoryLogging(world_idx, 0, nullptr);
                    }
                }
                
                if (user_input.keyHit(MockViewer::KeyboardKey::Space)) {
                    is_paused = !is_paused;
                }
            },
            [&](int32_t world_idx, int32_t, const MockViewer::UserInput& user_input) {
                // Handle agent controls
                int32_t x = 0, y = 0, r = 2;
                bool shift_pressed = user_input.keyPressed(MockViewer::KeyboardKey::Shift);
                
                if (user_input.keyPressed(MockViewer::KeyboardKey::W)) y += 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::S)) y -= 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::D)) x += 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::A)) x -= 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::Q)) 
                    r += shift_pressed ? 2 : 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::E)) 
                    r -= shift_pressed ? 2 : 1;
                
                int32_t move_amount = (x == 0 && y == 0) ? 0 : (shift_pressed ? 3 : 1);
                int32_t move_angle = 0;
                
                // Calculate move angle
                if (x == 0 && y == 1) move_angle = 0;
                else if (x == 1 && y == 1) move_angle = 1;
                else if (x == 1 && y == 0) move_angle = 2;
                else if (x == 1 && y == -1) move_angle = 3;
                else if (x == 0 && y == -1) move_angle = 4;
                else if (x == -1 && y == -1) move_angle = 5;
                else if (x == -1 && y == 0) move_angle = 6;
                else if (x == -1 && y == 1) move_angle = 7;
                
                mgr.setAction(world_idx, move_amount, move_angle, r);
                
                if (is_recording) {
                    uint32_t base_idx = world_idx * 3;
                    frame_actions[base_idx] = move_amount;
                    frame_actions[base_idx + 1] = move_angle;
                    frame_actions[base_idx + 2] = r;
                }
            },
            [&]() {
                if (!is_paused) {
                    if (mgr.hasReplay()) {
                        bool finished = mgr.replayStep();
                        if (finished) {
                            viewer.stopLoop();
                        }
                    }
                    
                    // Record frame actions if recording
                    // Note: In real viewer, this would call mgr.recordActions(frame_actions)
                    // But our test wrapper handles this in setAction
                    
                    mgr.step();
                }
            },
            []() {}
        );
    }
    
    std::unique_ptr<TestFileManager> file_manager_;
};

// Complete recording workflow
TEST_F(SimulatedViewerWorkflowTest, MockViewerRecordingSession) {
    // 1. Create level
    createTestLevelFile("game.lvl", 16, 16);
    file_manager_->addFile("game.lvl");
    file_manager_->addFile("session.rec");
    
    auto level = LevelComparer::loadLevelFromFile("game.lvl");
    
    // 2. Setup manager for recording
    config.num_worlds = 2;
    config.auto_reset = true;  // Required for recording
    config.rand_seed = 12345;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    MockViewer viewer(2);
    InputSimulator& input = viewer.getInputSimulator();
    
    // 3. Start recording (paused initially)
    bool is_paused = true;
    bool is_recording = true;
    mgr.startRecording("session.rec", config.rand_seed);
    
    // 4. Unpause and play
    input.hitKey(MockViewer::KeyboardKey::Space);
    input.nextFrame();
    is_paused = false;
    
    // 5. Simulate gameplay
    // Move forward for 5 frames
    input.simulateMovement(0, 1);
    runViewerSession(viewer, mgr, input, 5, is_paused, is_recording);
    
    // Turn right and move
    input.simulateMovement(1, 0);
    input.simulateRotation(1);
    runViewerSession(viewer, mgr, input, 5, is_paused, is_recording);
    
    // Reset world 0
    viewer.setCurrentWorld(0);
    input.releaseAll();
    input.hitKey(MockViewer::KeyboardKey::R);
    input.nextFrame();
    runViewerSession(viewer, mgr, input, 1, is_paused, is_recording);
    
    // Continue playing
    input.simulateMovement(-1, 1, true);  // Move forward-left with shift
    runViewerSession(viewer, mgr, input, 5, is_paused, is_recording);
    
    // 6. Stop recording
    mgr.stopRecording();
    
    // 7. Verify recording file
    EXPECT_TRUE(file_manager_->fileExists("session.rec"));
    size_t file_size = file_manager_->getFileSize("session.rec");
    EXPECT_GT(file_size, sizeof(MER_ReplayMetadata) + sizeof(MER_CompiledLevel));
    
    // 8. Verify recorded actions
    auto& recorder = mgr.getRecorder();
    EXPECT_GT(recorder.getActionCount(), 0);
    
    // 9. Verify metadata
    MER_ReplayMetadata metadata;
    ASSERT_EQ(mer_read_replay_metadata("session.rec", &metadata), MER_SUCCESS);
    EXPECT_EQ(metadata.num_worlds, 2);
    EXPECT_EQ(metadata.seed, 12345);
}

// Full replay verification workflow
TEST_F(SimulatedViewerWorkflowTest, ManagerReplayDeterminism) {
    // Phase 1: Record a session
    createTestLevelFile("original.lvl", 8, 8);
    file_manager_->addFile("original.lvl");
    // Note: Don't add demo.rec to cleanup yet - we need it after TearDown()
    file_manager_->addFile("trajectory_record.csv");
    file_manager_->addFile("trajectory_replay.csv");
    
    auto level = LevelComparer::loadLevelFromFile("original.lvl");
    
    std::vector<ActionRecorder::RecordedAction> original_actions;
    
    // Record phase
    {
        config.num_worlds = 1;
        config.auto_reset = true;
        config.rand_seed = 999;
        
        ASSERT_TRUE(CreateManager(&level, 1));
        
        TestManagerWrapper mgr(handle);
        mgr.startRecording("demo.rec", config.rand_seed);
        mgr.enableTrajectoryLogging(0, 0, "trajectory_record.csv");
        
        // Record specific sequence
        for (int i = 0; i < 10; i++) {
            int32_t move_amount = (i % 4);
            int32_t move_angle = (i % 8);
            int32_t rotate = 2 + (i % 3) - 1;
            
            mgr.setAction(0, move_amount, move_angle, rotate);
            original_actions.push_back({static_cast<uint32_t>(i), 0, 
                                       move_amount, move_angle, rotate});
            mgr.step();
        }
        
        mgr.stopRecording();
        mgr.disableTrajectoryLogging();
    }
    
    // Clean up first manager
    TearDown();
    SetUp();
    
    // Phase 2: Replay and verify
    {
        // Read metadata
        MER_ReplayMetadata metadata;
        ASSERT_EQ(mer_read_replay_metadata("demo.rec", &metadata), MER_SUCCESS);
        
        // Create manager for replay
        config.num_worlds = metadata.num_worlds;
        config.auto_reset = true;
        config.rand_seed = metadata.seed;
        
        ASSERT_TRUE(CreateManager(nullptr, 0));  // Level embedded in recording
        
        TestManagerWrapper mgr(handle);
        MockViewer viewer(1);
        InputSimulator& input = viewer.getInputSimulator();
        
        mgr.loadReplay("demo.rec");
        mgr.enableTrajectoryLogging(0, 0, "trajectory_replay.csv");
        
        EXPECT_TRUE(mgr.hasReplay());
        
        // Run replay
        bool is_paused = false;
        bool is_recording = false;
        
        viewer.setFrameLimit(10);
        viewer.loop(
            [](int32_t, const MockViewer::UserInput&) {},
            [](int32_t, int32_t, const MockViewer::UserInput&) {},
            [&]() {
                if (mgr.hasReplay()) {
                    mgr.replayStep();
                }
                mgr.step();
            },
            []() {}
        );
        
        mgr.disableTrajectoryLogging();
    }
    
    // Phase 3: Compare trajectories
    auto original_trajectory = TrajectoryComparer::parseTrajectoryFile("trajectory_record.csv");
    auto replay_trajectory = TrajectoryComparer::parseTrajectoryFile("trajectory_replay.csv");
    
    // Should have same number of points
    EXPECT_EQ(original_trajectory.size(), replay_trajectory.size());
    
    // Trajectories should match (deterministic simulation)
    if (original_trajectory.size() == replay_trajectory.size()) {
        for (size_t i = 0; i < original_trajectory.size(); i++) {
            EXPECT_NEAR(original_trajectory[i].x, replay_trajectory[i].x, 0.01f);
            EXPECT_NEAR(original_trajectory[i].y, replay_trajectory[i].y, 0.01f);
            EXPECT_NEAR(original_trajectory[i].z, replay_trajectory[i].z, 0.01f);
            EXPECT_NEAR(original_trajectory[i].rotation, replay_trajectory[i].rotation, 0.1f);
        }
    }
    
    // Now we can safely clean up the recording file
    file_manager_->addFile("demo.rec");
}

// Live simulation with tracking workflow
TEST_F(SimulatedViewerWorkflowTest, MockViewerTrajectoryWorkflow) {
    createTestLevelFile("world.lvl", 16, 16);
    file_manager_->addFile("world.lvl");
    file_manager_->addFile("live_trajectory.csv");
    
    auto level = LevelComparer::loadLevelFromFile("world.lvl");
    
    config.num_worlds = 4;
    config.auto_reset = false;  // Manual reset
    config.rand_seed = 777;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    MockViewer viewer(4);
    InputSimulator& input = viewer.getInputSimulator();
    
    bool is_paused = false;
    bool is_recording = false;
    
    // Start with no tracking
    EXPECT_FALSE(mgr.isTrajectoryEnabled());
    
    // Run for a bit
    input.simulateMovement(1, 1);  // Move forward-right
    runViewerSession(viewer, mgr, input, 5, is_paused, is_recording);
    
    // Enable tracking for world 2
    viewer.setCurrentWorld(2);
    input.releaseAll();
    input.hitKey(MockViewer::KeyboardKey::T);
    input.nextFrame();
    
    viewer.setFrameLimit(1);
    viewer.loop(
        [&](int32_t world_idx, const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::T)) {
                mgr.enableTrajectoryLogging(world_idx, 0, "live_trajectory.csv");
            }
        },
        [](int32_t, int32_t, const MockViewer::UserInput&) {},
        [&]() { mgr.step(); },
        []() {}
    );
    
    EXPECT_TRUE(mgr.isTrajectoryEnabled());
    EXPECT_EQ(mgr.getTrajectoryWorld(), 2);
    
    // Continue simulation with tracking
    input.simulateMovement(0, 1, true);  // Move forward fast
    runViewerSession(viewer, mgr, input, 10, is_paused, is_recording);
    
    // Reset world 2
    input.releaseAll();
    input.hitKey(MockViewer::KeyboardKey::R);
    input.nextFrame();
    
    viewer.setFrameLimit(1);
    viewer.loop(
        [&](int32_t world_idx, const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::R)) {
                mgr.triggerReset(world_idx);
            }
        },
        [](int32_t, int32_t, const MockViewer::UserInput&) {},
        [&]() { mgr.step(); },
        []() {}
    );
    
    // Continue after reset
    input.simulateMovement(-1, 0);  // Move left
    runViewerSession(viewer, mgr, input, 5, is_paused, is_recording);
    
    // Disable tracking
    input.releaseAll();
    input.hitKey(MockViewer::KeyboardKey::T);
    input.nextFrame();
    
    viewer.setFrameLimit(1);
    viewer.loop(
        [&](int32_t world_idx, const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::T)) {
                mgr.disableTrajectoryLogging();
            }
        },
        [](int32_t, int32_t, const MockViewer::UserInput&) {},
        [&]() { mgr.step(); },
        []() {}
    );
    
    EXPECT_FALSE(mgr.isTrajectoryEnabled());
    
    // Verify trajectory file
    EXPECT_TRUE(file_manager_->fileExists("live_trajectory.csv"));
    auto trajectory = TrajectoryComparer::parseTrajectoryFile("live_trajectory.csv");
    EXPECT_GT(trajectory.size(), 0);
    
    // All points should be from world 2
    for (const auto& point : trajectory) {
        EXPECT_EQ(point.world, 2);
        EXPECT_EQ(point.agent, 0);
    }
    
    // Verify reset occurred
    auto& resets = mgr.getResets();
    EXPECT_GE(resets.size(), 1);
    bool found_world2_reset = false;
    for (const auto& [step, world] : resets) {
        if (world == 2) {
            found_world2_reset = true;
            break;
        }
    }
    EXPECT_TRUE(found_world2_reset);
}

// Complex multi-world scenario
TEST_F(SimulatedViewerWorkflowTest, ManagerMultiWorldRecording) {
    createTestLevelFile("complex.lvl", 32, 32);
    file_manager_->addFile("complex.lvl");
    file_manager_->addFile("complex.rec");
    
    auto level = LevelComparer::loadLevelFromFile("complex.lvl");
    
    config.num_worlds = 8;
    config.auto_reset = true;
    config.rand_seed = 2468;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    MockViewer viewer(8);
    InputSimulator& input = viewer.getInputSimulator();
    
    // Start recording
    mgr.startRecording("complex.rec", config.rand_seed);
    
    bool is_paused = false;
    bool is_recording = true;
    
    // Different actions for different worlds
    for (int world = 0; world < 8; world++) {
        viewer.setCurrentWorld(world);
        
        // Each world gets different movement pattern
        switch (world % 4) {
            case 0:  // Circle pattern
                input.simulateMovement(1, 0);
                input.simulateRotation(1);
                break;
            case 1:  // Zigzag
                input.simulateMovement((world % 2) ? 1 : -1, 1);
                break;
            case 2:  // Back and forth
                input.simulateMovement(0, (world % 2) ? 1 : -1);
                break;
            case 3:  // Diagonal
                input.simulateMovement(1, 1, true);
                break;
        }
        
        runViewerSession(viewer, mgr, input, 2, is_paused, is_recording);
    }
    
    // Reset some worlds
    for (int world : {1, 3, 5}) {
        viewer.setCurrentWorld(world);
        input.releaseAll();
        input.hitKey(MockViewer::KeyboardKey::R);
        input.nextFrame();
        
        viewer.setFrameLimit(1);
        viewer.loop(
            [&](int32_t world_idx, const MockViewer::UserInput& user_input) {
                if (user_input.keyHit(MockViewer::KeyboardKey::R)) {
                    mgr.triggerReset(world_idx);
                }
            },
            [](int32_t, int32_t, const MockViewer::UserInput&) {},
            [&]() { mgr.step(); },
            []() {}
        );
    }
    
    // Continue simulation
    input.simulateMovement(0, 1);
    runViewerSession(viewer, mgr, input, 10, is_paused, is_recording);
    
    // Stop recording
    mgr.stopRecording();
    
    // Verify complex recording
    EXPECT_TRUE(file_manager_->fileExists("complex.rec"));
    
    MER_ReplayMetadata metadata;
    ASSERT_EQ(mer_read_replay_metadata("complex.rec", &metadata), MER_SUCCESS);
    EXPECT_EQ(metadata.num_worlds, 8);
    
    // Verify multiple resets occurred
    auto& resets = mgr.getResets();
    EXPECT_GE(resets.size(), 3);
    
    // Verify actions were recorded for all worlds
    auto& recorder = mgr.getRecorder();
    EXPECT_GT(recorder.getActionCount(), 8 * 10);  // At least 10 actions per world
}

// Pause/resume during recording
TEST_F(SimulatedViewerWorkflowTest, MockViewerPauseDuringRecording) {
    createTestLevelFile("pause_test.lvl", 16, 16);
    file_manager_->addFile("pause_test.lvl");
    file_manager_->addFile("pause_test.rec");
    
    auto level = LevelComparer::loadLevelFromFile("pause_test.lvl");
    
    config.num_worlds = 2;
    config.auto_reset = true;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    TestManagerWrapper mgr(handle);
    MockViewer viewer(2);
    InputSimulator& input = viewer.getInputSimulator();
    
    // Start recording (paused)
    bool is_paused = true;
    bool is_recording = true;
    mgr.startRecording("pause_test.rec", 111);
    
    int steps_while_paused = 0;
    int steps_while_running = 0;
    
    // Run while paused (nothing should be recorded)
    viewer.setFrameLimit(5);
    viewer.loop(
        [](int32_t, const MockViewer::UserInput&) {},
        [&](int32_t world_idx, int32_t, const MockViewer::UserInput&) {
            mgr.setAction(world_idx, MER_MOVE_SLOW, MER_MOVE_FORWARD, MER_ROTATE_NONE);
        },
        [&]() {
            if (!is_paused) {
                mgr.step();
                steps_while_running++;
            } else {
                steps_while_paused++;
            }
        },
        []() {}
    );
    
    EXPECT_EQ(steps_while_running, 0);
    EXPECT_GT(steps_while_paused, 0);
    
    // Unpause
    is_paused = false;
    steps_while_paused = 0;
    
    // Run while unpaused
    input.simulateMovement(1, 1);
    runViewerSession(viewer, mgr, input, 10, is_paused, is_recording);
    
    // Pause again
    is_paused = true;
    
    // Try to move while paused (should not record)
    input.simulateMovement(-1, -1, true);
    viewer.setFrameLimit(5);
    viewer.loop(
        [](int32_t, const MockViewer::UserInput&) {},
        [&](int32_t world_idx, int32_t, const MockViewer::UserInput&) {
            mgr.setAction(world_idx, MER_MOVE_FAST, MER_MOVE_BACKWARD_LEFT, MER_ROTATE_NONE);
        },
        [&]() {
            if (!is_paused) {
                mgr.step();
            }
        },
        []() {}
    );
    
    // Resume and finish
    is_paused = false;
    input.simulateMovement(0, 1);
    runViewerSession(viewer, mgr, input, 5, is_paused, is_recording);
    
    mgr.stopRecording();
    
    // Verify recording exists
    EXPECT_TRUE(file_manager_->fileExists("pause_test.rec"));
    
    // Actions should only be from unpaused periods
    auto& recorder = mgr.getRecorder();
    // We ran for 10 + 5 = 15 steps while unpaused, with 2 worlds
    EXPECT_GE(recorder.getActionCount(), 15 * 2);
}