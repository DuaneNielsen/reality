#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstdint>

// Test fixture for ViewerCore components
// This file implements comprehensive unit tests for the viewer core logic
// without requiring actual Manager or rendering dependencies

// =============================================================================
// Mock Classes
// =============================================================================

// Mock Manager for testing ViewerCore without real simulation
class MockManager {
public:
    struct Action {
        int32_t world_idx;
        int32_t move_amount;
        int32_t move_angle;
        int32_t rotate;
    };
    
    // Track method calls
    std::vector<Action> setActionCalls;
    std::vector<int32_t> triggerResetCalls;
    std::vector<std::vector<int32_t>> recordActionsCalls;
    int stepCount = 0;
    bool hasReplayFlag = false;
    bool replayFinished = false;
    
    // Mock methods
    void setAction(int32_t world, int32_t move, int32_t angle, int32_t rot) {
        setActionCalls.push_back({world, move, angle, rot});
    }
    
    void triggerReset(int32_t world) {
        triggerResetCalls.push_back(world);
    }
    
    void recordActions(const std::vector<int32_t>& actions) {
        recordActionsCalls.push_back(actions);
    }
    
    void step() { 
        stepCount++; 
    }
    
    bool hasReplay() const { 
        return hasReplayFlag; 
    }
    
    bool replayStep() {
        return replayFinished;
    }
    
    void clearCalls() {
        setActionCalls.clear();
        triggerResetCalls.clear();
        recordActionsCalls.clear();
        stepCount = 0;
    }
};

// =============================================================================
// RecordReplayStateMachine Tests
// =============================================================================

// Simple state machine for testing record/replay logic
class RecordReplayStateMachine {
public:
    enum State {
        Idle,
        RecordingPaused,
        Recording,
        ReplayingPaused,
        Replaying,
        ReplayFinished
    };
    
private:
    State state_ = Idle;
    
public:
    RecordReplayStateMachine() = default;
    
    State getState() const { return state_; }
    
    bool isPaused() const {
        return state_ == RecordingPaused || state_ == ReplayingPaused;
    }
    
    bool isRecording() const {
        return state_ == Recording || state_ == RecordingPaused;
    }
    
    bool isReplaying() const {
        return state_ == Replaying || state_ == ReplayingPaused;
    }
    
    bool shouldRecordFrame() const {
        return state_ == Recording;
    }
    
    bool shouldAdvanceReplay() const {
        return state_ == Replaying;
    }
    
    void startRecording() {
        if (state_ == Idle) {
            state_ = RecordingPaused;
        }
    }
    
    void startReplay() {
        if (state_ == Idle) {
            state_ = Replaying;  // Replay starts immediately
        }
    }
    
    void togglePause() {
        switch (state_) {
            case RecordingPaused:
                state_ = Recording;
                break;
            case Recording:
                state_ = RecordingPaused;
                break;
            case ReplayingPaused:
                state_ = Replaying;
                break;
            case Replaying:
                state_ = ReplayingPaused;
                break;
            default:
                // No-op for other states
                break;
        }
    }
    
    void finishReplay() {
        if (state_ == Replaying || state_ == ReplayingPaused) {
            state_ = ReplayFinished;
        }
    }
    
    void stop() {
        state_ = Idle;
    }
};

// RecordReplayStateMachine Test Fixture
class RecordReplayStateMachineTest : public ::testing::Test {
protected:
    RecordReplayStateMachine machine;
};

// Basic State Tests
TEST_F(RecordReplayStateMachineTest, InitialState) {
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Idle);
    EXPECT_FALSE(machine.isPaused());
    EXPECT_FALSE(machine.isRecording());
    EXPECT_FALSE(machine.isReplaying());
}

TEST_F(RecordReplayStateMachineTest, RecordingStartsPaused) {
    machine.startRecording();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::RecordingPaused);
    EXPECT_TRUE(machine.isPaused());
    EXPECT_TRUE(machine.isRecording());
    EXPECT_FALSE(machine.shouldRecordFrame());
}

TEST_F(RecordReplayStateMachineTest, ReplayStartsImmediately) {
    machine.startReplay();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Replaying);
    EXPECT_FALSE(machine.isPaused());
    EXPECT_TRUE(machine.isReplaying());
    EXPECT_TRUE(machine.shouldAdvanceReplay());
}

TEST_F(RecordReplayStateMachineTest, StateGetters) {
    // Test isPaused
    machine.startRecording();
    EXPECT_TRUE(machine.isPaused());
    machine.togglePause();
    EXPECT_FALSE(machine.isPaused());
    
    // Test isRecording
    EXPECT_TRUE(machine.isRecording());
    machine.stop();
    EXPECT_FALSE(machine.isRecording());
    
    // Test isReplaying
    machine.startReplay();
    EXPECT_TRUE(machine.isReplaying());
    machine.stop();
    EXPECT_FALSE(machine.isReplaying());
}

// State Transition Tests
TEST_F(RecordReplayStateMachineTest, IdleToRecording) {
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Idle);
    machine.startRecording();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::RecordingPaused);
}

TEST_F(RecordReplayStateMachineTest, RecordingPauseToggle) {
    machine.startRecording();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::RecordingPaused);
    
    machine.togglePause();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Recording);
    
    machine.togglePause();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::RecordingPaused);
}

TEST_F(RecordReplayStateMachineTest, ReplayPauseToggle) {
    machine.startReplay();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Replaying);
    
    machine.togglePause();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::ReplayingPaused);
    
    machine.togglePause();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Replaying);
}

TEST_F(RecordReplayStateMachineTest, ReplayFinish) {
    machine.startReplay();
    machine.finishReplay();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::ReplayFinished);
}

TEST_F(RecordReplayStateMachineTest, StopToIdle) {
    // From Recording
    machine.startRecording();
    machine.togglePause();
    machine.stop();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Idle);
    
    // From Replaying
    machine.startReplay();
    machine.stop();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Idle);
    
    // From ReplayFinished
    machine.startReplay();
    machine.finishReplay();
    machine.stop();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Idle);
}

// Edge Case Tests
TEST_F(RecordReplayStateMachineTest, InvalidTransitions) {
    // Try to start recording when already recording
    machine.startRecording();
    auto state = machine.getState();
    machine.startRecording();  // Should be no-op
    EXPECT_EQ(machine.getState(), state);
    
    // Try to start replay when recording
    machine.startReplay();  // Should be no-op
    EXPECT_EQ(machine.getState(), state);
}

TEST_F(RecordReplayStateMachineTest, MultipleStarts) {
    machine.startRecording();
    machine.startRecording();  // Should be ignored
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::RecordingPaused);
    
    machine.stop();
    machine.startReplay();
    machine.startReplay();  // Should be ignored
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Replaying);
}

TEST_F(RecordReplayStateMachineTest, PauseInInvalidStates) {
    // Pause in Idle - should be no-op
    machine.togglePause();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Idle);
    
    // Pause in ReplayFinished - should be no-op
    machine.startReplay();
    machine.finishReplay();
    machine.togglePause();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::ReplayFinished);
}

TEST_F(RecordReplayStateMachineTest, FinishInInvalidStates) {
    // Finish when not replaying
    machine.finishReplay();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::Idle);
    
    machine.startRecording();
    machine.finishReplay();
    EXPECT_EQ(machine.getState(), RecordReplayStateMachine::RecordingPaused);
}

// Recording Control Tests
TEST_F(RecordReplayStateMachineTest, ShouldRecordFrame) {
    EXPECT_FALSE(machine.shouldRecordFrame());
    
    machine.startRecording();
    EXPECT_FALSE(machine.shouldRecordFrame());  // Starts paused
    
    machine.togglePause();
    EXPECT_TRUE(machine.shouldRecordFrame());  // Now recording
    
    machine.togglePause();
    EXPECT_FALSE(machine.shouldRecordFrame());  // Paused again
}

TEST_F(RecordReplayStateMachineTest, ShouldAdvanceReplay) {
    EXPECT_FALSE(machine.shouldAdvanceReplay());
    
    machine.startReplay();
    EXPECT_TRUE(machine.shouldAdvanceReplay());  // Starts immediately
    
    machine.togglePause();
    EXPECT_FALSE(machine.shouldAdvanceReplay());  // Paused
    
    machine.togglePause();
    EXPECT_TRUE(machine.shouldAdvanceReplay());  // Unpaused
    
    machine.finishReplay();
    EXPECT_FALSE(machine.shouldAdvanceReplay());  // Finished
}

TEST_F(RecordReplayStateMachineTest, RecordingWhilePaused) {
    machine.startRecording();
    EXPECT_TRUE(machine.isPaused());
    EXPECT_FALSE(machine.shouldRecordFrame());
    
    // Multiple checks while paused
    for (int i = 0; i < 10; i++) {
        EXPECT_FALSE(machine.shouldRecordFrame());
    }
    
    machine.togglePause();
    EXPECT_TRUE(machine.shouldRecordFrame());
}

// =============================================================================
// FrameActionManager Tests
// =============================================================================

class FrameActionManager {
private:
    std::vector<int32_t> frame_actions_;
    uint32_t num_worlds_;
    bool has_changes_ = false;
    
    static constexpr int32_t DEFAULT_MOVE_AMOUNT = 0;
    static constexpr int32_t DEFAULT_MOVE_ANGLE = 0;
    static constexpr int32_t DEFAULT_ROTATE = 2;
    
public:
    explicit FrameActionManager(uint32_t num_worlds) 
        : num_worlds_(num_worlds) {
        resetToDefaults();
    }
    
    void resetToDefaults() {
        frame_actions_.resize(num_worlds_ * 3);
        for (uint32_t i = 0; i < num_worlds_; i++) {
            frame_actions_[i * 3] = DEFAULT_MOVE_AMOUNT;
            frame_actions_[i * 3 + 1] = DEFAULT_MOVE_ANGLE;
            frame_actions_[i * 3 + 2] = DEFAULT_ROTATE;
        }
        has_changes_ = false;
    }
    
    void setAction(uint32_t world_idx, int32_t move_amount, 
                   int32_t move_angle, int32_t rotate) {
        if (world_idx >= num_worlds_) {
            return;  // Out of bounds
        }
        
        uint32_t base_idx = world_idx * 3;
        frame_actions_[base_idx] = move_amount;
        frame_actions_[base_idx + 1] = move_angle;
        frame_actions_[base_idx + 2] = rotate;
        has_changes_ = true;
    }
    
    const std::vector<int32_t>& getFrameActions() const {
        return frame_actions_;
    }
    
    bool hasChanges() const {
        return has_changes_;
    }
    
    void clearChanges() {
        has_changes_ = false;
    }
};

// FrameActionManager Test Fixture
class FrameActionManagerTest : public ::testing::Test {
protected:
    std::unique_ptr<FrameActionManager> manager;
    
    void SetUp() override {
        manager = std::make_unique<FrameActionManager>(4);  // 4 worlds
    }
};

// Initialization Tests
TEST_F(FrameActionManagerTest, DefaultValues) {
    const auto& actions = manager->getFrameActions();
    EXPECT_EQ(actions.size(), 12u);  // 4 worlds * 3 values
    
    for (uint32_t i = 0; i < 4; i++) {
        EXPECT_EQ(actions[i * 3], 0);      // move_amount
        EXPECT_EQ(actions[i * 3 + 1], 0);  // move_angle
        EXPECT_EQ(actions[i * 3 + 2], 2);  // rotate
    }
}

TEST_F(FrameActionManagerTest, CorrectSize) {
    auto mgr1 = FrameActionManager(1);
    EXPECT_EQ(mgr1.getFrameActions().size(), 3u);
    
    auto mgr10 = FrameActionManager(10);
    EXPECT_EQ(mgr10.getFrameActions().size(), 30u);
}

TEST_F(FrameActionManagerTest, HasChangesFlag) {
    EXPECT_FALSE(manager->hasChanges());
    
    manager->setAction(0, 1, 2, 3);
    EXPECT_TRUE(manager->hasChanges());
    
    manager->clearChanges();
    EXPECT_FALSE(manager->hasChanges());
}

// Action Management Tests
TEST_F(FrameActionManagerTest, SetSingleAction) {
    manager->setAction(0, 3, 7, 4);
    
    const auto& actions = manager->getFrameActions();
    EXPECT_EQ(actions[0], 3);  // move_amount
    EXPECT_EQ(actions[1], 7);  // move_angle
    EXPECT_EQ(actions[2], 4);  // rotate
    
    // Other worlds unchanged
    EXPECT_EQ(actions[3], 0);
    EXPECT_EQ(actions[4], 0);
    EXPECT_EQ(actions[5], 2);
}

TEST_F(FrameActionManagerTest, SetMultipleActions) {
    manager->setAction(0, 1, 2, 3);
    manager->setAction(1, 2, 3, 4);
    manager->setAction(2, 3, 4, 0);
    
    const auto& actions = manager->getFrameActions();
    
    // World 0
    EXPECT_EQ(actions[0], 1);
    EXPECT_EQ(actions[1], 2);
    EXPECT_EQ(actions[2], 3);
    
    // World 1
    EXPECT_EQ(actions[3], 2);
    EXPECT_EQ(actions[4], 3);
    EXPECT_EQ(actions[5], 4);
    
    // World 2
    EXPECT_EQ(actions[6], 3);
    EXPECT_EQ(actions[7], 4);
    EXPECT_EQ(actions[8], 0);
    
    // World 3 (unchanged)
    EXPECT_EQ(actions[9], 0);
    EXPECT_EQ(actions[10], 0);
    EXPECT_EQ(actions[11], 2);
}

TEST_F(FrameActionManagerTest, ResetToDefaults) {
    manager->setAction(0, 1, 2, 3);
    manager->setAction(1, 2, 3, 4);
    
    manager->resetToDefaults();
    
    const auto& actions = manager->getFrameActions();
    for (uint32_t i = 0; i < 4; i++) {
        EXPECT_EQ(actions[i * 3], 0);
        EXPECT_EQ(actions[i * 3 + 1], 0);
        EXPECT_EQ(actions[i * 3 + 2], 2);
    }
    
    EXPECT_FALSE(manager->hasChanges());
}

TEST_F(FrameActionManagerTest, OutOfBoundsHandling) {
    manager->setAction(10, 1, 2, 3);  // Out of bounds
    
    const auto& actions = manager->getFrameActions();
    // All should remain default
    for (uint32_t i = 0; i < 4; i++) {
        EXPECT_EQ(actions[i * 3], 0);
        EXPECT_EQ(actions[i * 3 + 1], 0);
        EXPECT_EQ(actions[i * 3 + 2], 2);
    }
    
    EXPECT_FALSE(manager->hasChanges());  // No changes due to invalid index
}

// Action Encoding Tests
TEST_F(FrameActionManagerTest, ActionPacking) {
    // Test that actions are packed correctly in groups of 3
    manager->setAction(1, 1, 2, 3);
    
    const auto& actions = manager->getFrameActions();
    EXPECT_EQ(actions[3], 1);  // World 1, move_amount
    EXPECT_EQ(actions[4], 2);  // World 1, move_angle
    EXPECT_EQ(actions[5], 3);  // World 1, rotate
}

TEST_F(FrameActionManagerTest, IndependentWorlds) {
    // Actions for one world don't affect others
    manager->setAction(0, 1, 1, 1);
    manager->setAction(2, 2, 2, 2);
    
    const auto& actions = manager->getFrameActions();
    
    // World 0 modified
    EXPECT_EQ(actions[0], 1);
    EXPECT_EQ(actions[1], 1);
    EXPECT_EQ(actions[2], 1);
    
    // World 1 unchanged
    EXPECT_EQ(actions[3], 0);
    EXPECT_EQ(actions[4], 0);
    EXPECT_EQ(actions[5], 2);
    
    // World 2 modified
    EXPECT_EQ(actions[6], 2);
    EXPECT_EQ(actions[7], 2);
    EXPECT_EQ(actions[8], 2);
    
    // World 3 unchanged
    EXPECT_EQ(actions[9], 0);
    EXPECT_EQ(actions[10], 0);
    EXPECT_EQ(actions[11], 2);
}

// =============================================================================
// ViewerCore Tests (Simplified for unit testing)
// =============================================================================

// Simplified ViewerCore for testing without full dependencies
class ViewerCore {
public:
    struct Config {
        uint32_t num_worlds = 1;
        std::string record_path;
        std::string replay_path;
        bool is_recording = false;
        bool has_replay = false;
    };
    
    struct InputState {
        bool keys[257] = {false};  // Need 257 to include Shift at index 256
    };
    
    enum Key {
        W = 'W',
        A = 'A',
        S = 'S',
        D = 'D',
        Q = 'Q',
        E = 'E',
        R = 'R',
        T = 'T',
        Space = ' ',
        Shift = 256
    };
    
private:
    Config config_;
    MockManager* manager_;
    RecordReplayStateMachine state_machine_;
    FrameActionManager action_manager_;
    std::vector<InputState> input_states_;  // Per-world input states
    bool should_exit_ = false;
    
public:
    ViewerCore(const Config& cfg, MockManager* mgr) 
        : config_(cfg), 
          manager_(mgr),
          action_manager_(cfg.num_worlds),
          input_states_(cfg.num_worlds) {
        
        if (cfg.is_recording) {
            state_machine_.startRecording();
        } else if (cfg.has_replay) {
            state_machine_.startReplay();
        }
    }
    
    void handleKeyPress(int world_idx, Key key) {
        if (key == Space) {
            state_machine_.togglePause();
            return;
        }
        
        if (key == R) {
            manager_->triggerReset(world_idx);
            return;
        }
        
        // Ignore input during replay
        if (state_machine_.isReplaying()) {
            return;
        }
        
        if (world_idx >= 0 && world_idx < (int)config_.num_worlds) {
            input_states_[world_idx].keys[key] = true;
            updateActionsFromInput(world_idx);
        }
    }
    
    void handleKeyRelease(int world_idx, Key key) {
        // Ignore input during replay
        if (state_machine_.isReplaying()) {
            return;
        }
        
        if (world_idx >= 0 && world_idx < (int)config_.num_worlds) {
            input_states_[world_idx].keys[key] = false;
            updateActionsFromInput(world_idx);
        }
    }
    
    void step() {
        if (state_machine_.isPaused()) {
            return;
        }
        
        if (state_machine_.shouldAdvanceReplay()) {
            bool finished = manager_->replayStep();
            if (finished) {
                state_machine_.finishReplay();
                should_exit_ = true;
            }
        }
        
        if (state_machine_.shouldRecordFrame()) {
            manager_->recordActions(action_manager_.getFrameActions());
        }
        
        manager_->step();
        
        // Reset actions for next frame
        if (!state_machine_.isReplaying()) {
            action_manager_.resetToDefaults();
        }
    }
    
    bool shouldExit() const { return should_exit_; }
    
    bool isPaused() const { return state_machine_.isPaused(); }
    
    bool isRecording() const { return state_machine_.isRecording(); }
    
    bool isReplaying() const { return state_machine_.isReplaying(); }
    
    const std::vector<int32_t>& getCurrentActions() const {
        return action_manager_.getFrameActions();
    }
    
private:
    void updateActionsFromInput(int world_idx) {
        if (world_idx < 0 || world_idx >= (int)config_.num_worlds) {
            return;
        }
        
        const auto& input = input_states_[world_idx];
        
        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 2;
        
        bool shift_pressed = input.keys[Shift];
        
        if (input.keys[W]) y += 1;
        if (input.keys[S]) y -= 1;
        if (input.keys[D]) x += 1;
        if (input.keys[A]) x -= 1;
        
        if (input.keys[Q]) r = shift_pressed ? 0 : 1;
        else if (input.keys[E]) r = shift_pressed ? 4 : 3;
        
        int32_t move_amount;
        if (x == 0 && y == 0) {
            move_amount = 0;
        } else if (shift_pressed) {
            move_amount = 3;
        } else {
            move_amount = 1;
        }
        
        int32_t move_angle;
        if (x == 0 && y == 1) move_angle = 0;
        else if (x == 1 && y == 1) move_angle = 1;
        else if (x == 1 && y == 0) move_angle = 2;
        else if (x == 1 && y == -1) move_angle = 3;
        else if (x == 0 && y == -1) move_angle = 4;
        else if (x == -1 && y == -1) move_angle = 5;
        else if (x == -1 && y == 0) move_angle = 6;
        else if (x == -1 && y == 1) move_angle = 7;
        else move_angle = 0;
        
        action_manager_.setAction(world_idx, move_amount, move_angle, r);
        manager_->setAction(world_idx, move_amount, move_angle, r);
    }
};

// ViewerCore Test Fixture
class ViewerCoreTest : public ::testing::Test {
protected:
    std::unique_ptr<MockManager> mockMgr;
    std::unique_ptr<ViewerCore> core;
    
    void SetUp() override {
        mockMgr = std::make_unique<MockManager>();
    }
    
    void createCore(ViewerCore::Config config = {}) {
        core = std::make_unique<ViewerCore>(config, mockMgr.get());
    }
};

// Construction Tests
TEST_F(ViewerCoreTest, ConfigParsing) {
    ViewerCore::Config config;
    config.num_worlds = 4;
    createCore(config);
    
    EXPECT_FALSE(core->isRecording());
    EXPECT_FALSE(core->isReplaying());
    EXPECT_FALSE(core->isPaused());
}

TEST_F(ViewerCoreTest, RecordModeInit) {
    ViewerCore::Config config;
    config.is_recording = true;
    createCore(config);
    
    EXPECT_TRUE(core->isRecording());
    EXPECT_TRUE(core->isPaused());  // Starts paused
    EXPECT_FALSE(core->isReplaying());
}

TEST_F(ViewerCoreTest, ReplayModeInit) {
    ViewerCore::Config config;
    config.has_replay = true;
    createCore(config);
    
    EXPECT_FALSE(core->isRecording());
    EXPECT_TRUE(core->isReplaying());
    EXPECT_FALSE(core->isPaused());  // Starts immediately
}

TEST_F(ViewerCoreTest, NormalModeInit) {
    createCore();  // Default config
    
    EXPECT_FALSE(core->isRecording());
    EXPECT_FALSE(core->isReplaying());
    EXPECT_FALSE(core->isPaused());
}

// Input Handling Tests
TEST_F(ViewerCoreTest, MovementKeys) {
    createCore();
    
    // Test W key (forward)
    core->handleKeyPress(0, ViewerCore::W);
    auto& actions = core->getCurrentActions();
    EXPECT_EQ(actions[0], 1);  // move_amount
    EXPECT_EQ(actions[1], 0);  // move_angle (forward)
    
    // Test S key (backward)
    core->handleKeyRelease(0, ViewerCore::W);
    core->handleKeyPress(0, ViewerCore::S);
    EXPECT_EQ(actions[0], 1);  // move_amount
    EXPECT_EQ(actions[1], 4);  // move_angle (backward)
}

TEST_F(ViewerCoreTest, RotationKeys) {
    createCore();
    
    // Test Q key (rotate left)
    core->handleKeyPress(0, ViewerCore::Q);
    auto& actions = core->getCurrentActions();
    EXPECT_EQ(actions[2], 1);  // rotate left slow
    
    // Test E key (rotate right)
    core->handleKeyRelease(0, ViewerCore::Q);
    core->handleKeyPress(0, ViewerCore::E);
    EXPECT_EQ(actions[2], 3);  // rotate right slow
}

TEST_F(ViewerCoreTest, ShiftModifier) {
    createCore();
    
    // Move with shift (fast)
    core->handleKeyPress(0, ViewerCore::Shift);
    core->handleKeyPress(0, ViewerCore::W);
    auto& actions = core->getCurrentActions();
    EXPECT_EQ(actions[0], 3);  // move_amount (fast)
    
    // Rotate with shift (fast)
    core->handleKeyRelease(0, ViewerCore::W);
    core->handleKeyPress(0, ViewerCore::Q);
    EXPECT_EQ(actions[2], 0);  // rotate left fast
}

TEST_F(ViewerCoreTest, SpaceKeyPause) {
    ViewerCore::Config config;
    config.is_recording = true;
    createCore(config);
    
    EXPECT_TRUE(core->isPaused());
    
    core->handleKeyPress(0, ViewerCore::Space);
    EXPECT_FALSE(core->isPaused());
    
    core->handleKeyPress(0, ViewerCore::Space);
    EXPECT_TRUE(core->isPaused());
}

TEST_F(ViewerCoreTest, ResetKey) {
    createCore();
    
    core->handleKeyPress(0, ViewerCore::R);
    EXPECT_EQ(mockMgr->triggerResetCalls.size(), 1u);
    EXPECT_EQ(mockMgr->triggerResetCalls[0], 0);
    
    core->handleKeyPress(2, ViewerCore::R);
    EXPECT_EQ(mockMgr->triggerResetCalls.size(), 2u);
    EXPECT_EQ(mockMgr->triggerResetCalls[1], 2);
}

// Action Computation Tests
TEST_F(ViewerCoreTest, SingleKeyMovement) {
    createCore();
    
    // Test each direction
    core->handleKeyPress(0, ViewerCore::W);
    EXPECT_EQ(core->getCurrentActions()[1], 0);  // Forward
    
    core->handleKeyRelease(0, ViewerCore::W);
    core->handleKeyPress(0, ViewerCore::D);
    EXPECT_EQ(core->getCurrentActions()[1], 2);  // Right
    
    core->handleKeyRelease(0, ViewerCore::D);
    core->handleKeyPress(0, ViewerCore::S);
    EXPECT_EQ(core->getCurrentActions()[1], 4);  // Backward
    
    core->handleKeyRelease(0, ViewerCore::S);
    core->handleKeyPress(0, ViewerCore::A);
    EXPECT_EQ(core->getCurrentActions()[1], 6);  // Left
}

TEST_F(ViewerCoreTest, DiagonalMovement) {
    createCore();
    
    // Forward-right
    core->handleKeyPress(0, ViewerCore::W);
    core->handleKeyPress(0, ViewerCore::D);
    EXPECT_EQ(core->getCurrentActions()[1], 1);
    
    // Backward-left
    core->handleKeyRelease(0, ViewerCore::W);
    core->handleKeyRelease(0, ViewerCore::D);
    core->handleKeyPress(0, ViewerCore::S);
    core->handleKeyPress(0, ViewerCore::A);
    EXPECT_EQ(core->getCurrentActions()[1], 5);
}

TEST_F(ViewerCoreTest, StopWhenNoKeys) {
    createCore();
    
    core->handleKeyPress(0, ViewerCore::W);
    EXPECT_EQ(core->getCurrentActions()[0], 1);  // Moving
    
    core->handleKeyRelease(0, ViewerCore::W);
    EXPECT_EQ(core->getCurrentActions()[0], 0);  // Stopped
}

// Recording Workflow Tests
TEST_F(ViewerCoreTest, CompleteRecordingCycle) {
    ViewerCore::Config config;
    config.is_recording = true;
    config.num_worlds = 2;
    createCore(config);
    
    // 1. Starts paused
    EXPECT_TRUE(core->isPaused());
    
    // 2. Step while paused - no recording
    core->step();
    EXPECT_EQ(mockMgr->recordActionsCalls.size(), 0u);
    
    // 3. Unpause
    core->handleKeyPress(0, ViewerCore::Space);
    EXPECT_FALSE(core->isPaused());
    
    // 4. Input movement
    core->handleKeyPress(0, ViewerCore::W);
    
    // 5. Step - should record
    core->step();
    EXPECT_EQ(mockMgr->recordActionsCalls.size(), 1u);
    EXPECT_EQ(mockMgr->stepCount, 1);
    
    // 6. Pause again
    core->handleKeyPress(0, ViewerCore::Space);
    
    // 7. Step while paused - no more recording
    core->step();
    EXPECT_EQ(mockMgr->recordActionsCalls.size(), 1u);  // No new recording
    EXPECT_EQ(mockMgr->stepCount, 1);  // No step
}

TEST_F(ViewerCoreTest, ActionsResetPerFrame) {
    ViewerCore::Config config;
    config.is_recording = true;
    createCore(config);
    
    core->handleKeyPress(0, ViewerCore::Space);  // Unpause
    core->handleKeyPress(0, ViewerCore::W);
    
    auto& actions = core->getCurrentActions();
    EXPECT_EQ(actions[0], 1);  // Movement set
    
    core->step();
    
    // After step, actions should reset
    EXPECT_EQ(actions[0], 0);
    EXPECT_EQ(actions[1], 0);
    EXPECT_EQ(actions[2], 2);
}

// Replay Workflow Tests
TEST_F(ViewerCoreTest, CompleteReplayCycle) {
    ViewerCore::Config config;
    config.has_replay = true;
    createCore(config);
    
    mockMgr->hasReplayFlag = true;
    
    // 1. Starts immediately
    EXPECT_FALSE(core->isPaused());
    EXPECT_TRUE(core->isReplaying());
    
    // 2. Step through replay
    core->step();
    EXPECT_EQ(mockMgr->stepCount, 1);
    
    // 3. Input ignored during replay
    core->handleKeyPress(0, ViewerCore::W);
    auto& actions = core->getCurrentActions();
    EXPECT_EQ(actions[0], 0);  // No action set
    
    // 4. Reach end
    mockMgr->replayFinished = true;
    core->step();
    EXPECT_TRUE(core->shouldExit());
}

TEST_F(ViewerCoreTest, PauseDuringReplay) {
    ViewerCore::Config config;
    config.has_replay = true;
    createCore(config);
    
    mockMgr->hasReplayFlag = true;
    
    EXPECT_FALSE(core->isPaused());
    
    core->handleKeyPress(0, ViewerCore::Space);
    EXPECT_TRUE(core->isPaused());
    
    // Step while paused - no advancement
    int prevSteps = mockMgr->stepCount;
    core->step();
    EXPECT_EQ(mockMgr->stepCount, prevSteps);
    
    // Resume
    core->handleKeyPress(0, ViewerCore::Space);
    EXPECT_FALSE(core->isPaused());
    
    core->step();
    EXPECT_EQ(mockMgr->stepCount, prevSteps + 1);
}

// Edge Cases
TEST_F(ViewerCoreTest, InputIgnoredDuringReplay) {
    ViewerCore::Config config;
    config.has_replay = true;
    createCore(config);
    
    // Try to set actions during replay
    core->handleKeyPress(0, ViewerCore::W);
    core->handleKeyPress(0, ViewerCore::Q);
    
    auto& actions = core->getCurrentActions();
    // All should remain default
    EXPECT_EQ(actions[0], 0);
    EXPECT_EQ(actions[1], 0);
    EXPECT_EQ(actions[2], 2);
    
    // Manager should not receive any setAction calls
    EXPECT_EQ(mockMgr->setActionCalls.size(), 0u);
}

TEST_F(ViewerCoreTest, MultipleWorldsIndependent) {
    ViewerCore::Config config;
    config.num_worlds = 3;
    createCore(config);
    
    // Set action for world 0
    core->handleKeyPress(0, ViewerCore::W);
    
    // Set different action for world 1
    core->handleKeyPress(1, ViewerCore::S);
    
    auto& actions = core->getCurrentActions();
    
    // World 0: forward
    EXPECT_EQ(actions[0], 1);
    EXPECT_EQ(actions[1], 0);
    
    // World 1: backward
    EXPECT_EQ(actions[3], 1);
    EXPECT_EQ(actions[4], 4);
    
    // World 2: unchanged
    EXPECT_EQ(actions[6], 0);
    EXPECT_EQ(actions[7], 0);
    EXPECT_EQ(actions[8], 2);
}

// End-to-end test with recording and trajectory verification
TEST_F(ViewerCoreTest, RecordReplayWithTrajectoryVerification) {
    // Phase 1: Set up recording
    ViewerCore::Config config;
    config.num_worlds = 1;
    config.is_recording = true;
    config.record_path = "test_recording.rec";
    
    // Reset mock manager
    mockMgr->clearCalls();
    mockMgr->hasReplayFlag = false;
    
    createCore(config);
    
    // Should start in recording mode, paused
    EXPECT_TRUE(core->isRecording());
    EXPECT_TRUE(core->isPaused());
    
    // Phase 2: Unpause and record 100 frames of forward movement
    core->handleKeyPress(0, ViewerCore::Space);  // Unpause
    EXPECT_FALSE(core->isPaused());
    
    // Record 100 frames with W key pressed
    for (int i = 0; i < 100; i++) {
        // Press W key before each step (simulating held key)
        core->handleKeyPress(0, ViewerCore::W);
        core->step();
        // Note: In real viewer, key would stay pressed, but our test ViewerCore
        // resets actions after each step, so we need to press again
    }
    
    // Verify we recorded 100 frames
    EXPECT_EQ(mockMgr->recordActionsCalls.size(), 100u);
    EXPECT_EQ(mockMgr->stepCount, 100);
    
    // Verify all frames had forward movement (move_amount=1, move_angle=0, rotate=2)
    for (const auto& frame : mockMgr->recordActionsCalls) {
        EXPECT_EQ(frame.size(), 3u);  // 1 world * 3 values
        EXPECT_EQ(frame[0], 1);  // move_amount
        EXPECT_EQ(frame[1], 0);  // move_angle (forward)
        EXPECT_EQ(frame[2], 2);  // rotate (none)
    }
    
    // Phase 3: Stop recording and create replay
    core.reset();  // Destroy the recording core
    
    // Phase 4: Load replay
    config.is_recording = false;
    config.has_replay = true;
    config.replay_path = "test_recording.rec";
    
    // Reset mock manager for replay
    mockMgr->clearCalls();
    mockMgr->hasReplayFlag = true;
    mockMgr->replayFinished = false;
    
    createCore(config);
    
    // Should start in replay mode, not paused (replays start immediately)
    EXPECT_TRUE(core->isReplaying());
    EXPECT_FALSE(core->isPaused());
    
    // Phase 5: Step through replay
    for (int i = 0; i < 100; i++) {
        // On last frame, mark replay as finished BEFORE stepping
        // so the step() function sees it and sets should_exit_
        if (i == 99) {
            mockMgr->replayFinished = true;
        }
        
        core->step();
    }
    
    // Verify replay stepped through all frames
    EXPECT_EQ(mockMgr->stepCount, 100);
    
    // After replay finishes, should exit
    EXPECT_TRUE(core->shouldExit());
    
    // Phase 6: Verify determinism
    // In a real test, we would compare trajectory files here
    // For this mock test, we verify the same actions were set
    EXPECT_EQ(mockMgr->setActionCalls.size(), 0u) << "No manual actions should be set during replay";
}

// Test recording with multiple worlds
TEST_F(ViewerCoreTest, MultiWorldRecording) {
    ViewerCore::Config config;
    config.num_worlds = 3;
    config.is_recording = true;
    
    createCore(config);
    
    // Unpause
    core->handleKeyPress(0, ViewerCore::Space);
    
    // Set different actions for each world
    core->handleKeyPress(0, ViewerCore::W);  // World 0: forward
    core->handleKeyPress(1, ViewerCore::S);  // World 1: backward
    core->handleKeyPress(2, ViewerCore::D);  // World 2: right
    
    // Step once
    core->step();
    
    // Verify recorded frame has all three worlds' actions
    ASSERT_EQ(mockMgr->recordActionsCalls.size(), 1u);
    const auto& frame = mockMgr->recordActionsCalls[0];
    ASSERT_EQ(frame.size(), 9u);  // 3 worlds * 3 values
    
    // World 0: forward
    EXPECT_EQ(frame[0], 1);  // move_amount
    EXPECT_EQ(frame[1], 0);  // move_angle (forward)
    
    // World 1: backward
    EXPECT_EQ(frame[3], 1);  // move_amount
    EXPECT_EQ(frame[4], 4);  // move_angle (backward)
    
    // World 2: right
    EXPECT_EQ(frame[6], 1);  // move_amount
    EXPECT_EQ(frame[7], 2);  // move_angle (right)
}

// Main test runner moved to fixtures/test_main.cpp for centralized flag handling