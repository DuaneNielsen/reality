# Viewer Refactoring Plan for Testability

## Problem Statement
The viewer's logic is entangled with I/O (window management, keyboard input, rendering), making it impossible to test critical behaviors like:
- Recording/replay state machine
- Pause/resume logic during recording
- Frame action batching and reset
- Deterministic replay execution

## Core Issue
The recording/replay system has bugs where:
1. User starts paused → presses space → records → closes window
2. On replay, agent takes completely different trajectory
3. Logic bugs are hidden in the monolithic main() function

## Refactoring Approach

### Phase 1: Extract ViewerCore Class
Create a testable core that handles all logic WITHOUT any I/O:

```cpp
// viewer_core.hpp
class ViewerCore {
public:
    struct Config {
        uint32_t num_worlds;
        uint32_t rand_seed;
        bool auto_reset;
        std::string load_path;
        std::string record_path;
        std::string replay_path;
    };
    
    struct InputEvent {
        enum Type { KeyPress, KeyRelease, KeyHit };
        enum Key { W, A, S, D, Q, E, R, T, Space, Shift };
        Type type;
        Key key;
    };
    
    struct FrameState {
        bool is_paused;
        bool is_recording;
        bool has_replay;
        bool should_exit;
        std::vector<int32_t> frame_actions;
    };

    ViewerCore(const Config& cfg, Manager* mgr);
    
    // Pure logic functions - fully testable
    void handleInput(int world_idx, const InputEvent& event);
    void updateFrameActions(int world_idx, int agent_idx);
    void stepSimulation();
    FrameState getFrameState() const;
    
    // Recording/replay logic
    void startRecording(const std::string& path);
    void stopRecording();
    void loadReplay(const std::string& path);
    bool replayStep();
    
private:
    // State management
    bool is_paused_;
    bool is_recording_;
    bool has_replay_;
    std::vector<int32_t> frame_actions_;
    std::vector<int32_t> default_actions_;
    
    // Input state
    struct InputState {
        bool keys_pressed[Key::Count];
        bool keys_hit[Key::Count];
        int32_t move_x, move_y;
        int32_t rotate;
        bool shift_pressed;
    } input_state_[MAX_WORLDS];
    
    // Convert input state to actions
    void computeActionsFromInput(int world_idx);
    void resetFrameActions();
    void applyFrameActions();
};
```

### Phase 2: Separate I/O Layer
Keep I/O in main() but delegate all logic to ViewerCore:

```cpp
// viewer.cpp main()
int main(int argc, char* argv[]) {
    // ... parse options ...
    
    // Create core with config
    ViewerCore::Config core_config{
        .num_worlds = num_worlds,
        .rand_seed = rand_seed,
        .record_path = record_path,
        .replay_path = replay_path
    };
    
    Manager mgr(...);
    ViewerCore core(core_config, &mgr);
    
    // Setup I/O
    WindowManager wm;
    viz::Viewer viewer(...);
    
    // Main loop - I/O only, logic in core
    viewer.loop(
        [&core](int world_idx, const Viewer::UserInput& input) {
            // Convert raw input to events
            if (input.keyHit(Key::Space)) {
                core.handleInput(world_idx, ViewerCore::InputEvent{
                    .type = InputEvent::KeyHit, 
                    .key = InputEvent::Space
                });
            }
            // ... other keys ...
        },
        [&core](int world_idx, int agent_idx, const Viewer::UserInput& input) {
            // Convert WASD to input events
            if (input.keyPressed(Key::W)) {
                core.handleInput(world_idx, ViewerCore::InputEvent{
                    .type = InputEvent::KeyPress,
                    .key = InputEvent::W
                });
            }
            // ... other keys ...
            
            // Core computes actions from input state
            core.updateFrameActions(world_idx, agent_idx);
        },
        [&core]() {
            // Core handles all simulation logic
            core.stepSimulation();
            
            auto state = core.getFrameState();
            if (state.should_exit) {
                viewer.stopLoop();
            }
        }
    );
}
```

### Phase 3: Testable Recording/Replay State Machine

Extract the recording/replay logic into a proper state machine:

```cpp
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
    
    State getState() const { return state_; }
    
    // State transitions - all testable
    void startRecording() {
        if (state_ == Idle) {
            state_ = RecordingPaused;
            // Start paused as per viewer behavior
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
        }
    }
    
    bool shouldRecordFrame() const {
        return state_ == Recording;
    }
    
    bool shouldAdvanceReplay() const {
        return state_ == Replaying;
    }
    
private:
    State state_ = Idle;
};
```

### Phase 4: Frame Action Manager

Separate frame action batching logic:

```cpp
class FrameActionManager {
public:
    FrameActionManager(uint32_t num_worlds)
        : num_worlds_(num_worlds),
          frame_actions_(num_worlds * 3),
          default_actions_(num_worlds * 3) {
        resetToDefaults();
    }
    
    void setAction(uint32_t world, int32_t move_amount, 
                  int32_t move_angle, int32_t rotate) {
        uint32_t base = world * 3;
        frame_actions_[base] = move_amount;
        frame_actions_[base + 1] = move_angle;
        frame_actions_[base + 2] = rotate;
        has_changes_ = true;
    }
    
    void resetToDefaults() {
        for (uint32_t i = 0; i < num_worlds_; i++) {
            frame_actions_[i * 3] = 0;      // move_amount
            frame_actions_[i * 3 + 1] = 0;  // move_angle
            frame_actions_[i * 3 + 2] = 2;  // rotate
        }
        has_changes_ = false;
    }
    
    const std::vector<int32_t>& getFrameActions() const {
        return frame_actions_;
    }
    
    bool hasChanges() const { return has_changes_; }
    
private:
    uint32_t num_worlds_;
    std::vector<int32_t> frame_actions_;
    std::vector<int32_t> default_actions_;
    bool has_changes_ = false;
};
```

## Testing Strategy

### Unit Tests for ViewerCore
```cpp
TEST(ViewerCoreTest, RecordingStartsPaused) {
    MockManager mgr;
    ViewerCore::Config config{.record_path = "test.rec"};
    ViewerCore core(config, &mgr);
    
    EXPECT_TRUE(core.getFrameState().is_paused);
    EXPECT_TRUE(core.getFrameState().is_recording);
}

TEST(ViewerCoreTest, SpaceKeyUnpausesRecording) {
    ViewerCore core(config, &mgr);
    
    // Initially paused
    EXPECT_TRUE(core.getFrameState().is_paused);
    
    // Space key toggles
    core.handleInput(0, {InputEvent::KeyHit, InputEvent::Space});
    EXPECT_FALSE(core.getFrameState().is_paused);
    
    // Actions should now be recorded
    core.handleInput(0, {InputEvent::KeyPress, InputEvent::W});
    core.updateFrameActions(0, 0);
    core.stepSimulation();
    
    // Verify action was recorded
    EXPECT_TRUE(mgr.wasActionRecorded());
}

TEST(ViewerCoreTest, FrameActionsResetAfterStep) {
    ViewerCore core(config, &mgr);
    
    // Set some actions
    core.handleInput(0, {InputEvent::KeyPress, InputEvent::W});
    core.updateFrameActions(0, 0);
    
    auto state1 = core.getFrameState();
    EXPECT_NE(state1.frame_actions[0], 0);  // Has movement
    
    // Step should reset
    core.stepSimulation();
    
    auto state2 = core.getFrameState();
    EXPECT_EQ(state2.frame_actions[0], 0);  // Reset to stop
}

TEST(ViewerCoreTest, ReplayDeterminism) {
    // Record a sequence
    ViewerCore core1(config, &mgr);
    core1.startRecording("test.rec");
    core1.handleInput(0, {InputEvent::KeyHit, InputEvent::Space}); // Unpause
    
    // Specific action sequence
    core1.handleInput(0, {InputEvent::KeyPress, InputEvent::W});
    core1.stepSimulation();
    core1.handleInput(0, {InputEvent::KeyPress, InputEvent::D});
    core1.stepSimulation();
    
    core1.stopRecording();
    
    // Replay the sequence
    ViewerCore core2(config, &mgr);
    core2.loadReplay("test.rec");
    
    // Should replay exact same actions
    core2.stepSimulation();
    EXPECT_EQ(mgr.getLastAction(0), {MOVE_FORWARD, 0, ROTATE_NONE});
    
    core2.stepSimulation();
    EXPECT_EQ(mgr.getLastAction(0), {MOVE_RIGHT, 0, ROTATE_NONE});
}
```

### State Machine Tests
```cpp
TEST(RecordReplayStateMachine, InitialState) {
    RecordReplayStateMachine sm;
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Idle);
}

TEST(RecordReplayStateMachine, RecordingStartsPaused) {
    RecordReplayStateMachine sm;
    sm.startRecording();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    EXPECT_FALSE(sm.shouldRecordFrame());
}

TEST(RecordReplayStateMachine, PauseToggle) {
    RecordReplayStateMachine sm;
    sm.startRecording();
    
    // Start paused
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    
    // Toggle to recording
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Recording);
    EXPECT_TRUE(sm.shouldRecordFrame());
    
    // Toggle back to paused
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    EXPECT_FALSE(sm.shouldRecordFrame());
}
```

## Benefits

1. **Pure Logic Testing**: Test state transitions without any I/O
2. **Determinism Verification**: Ensure replay produces exact same actions
3. **State Machine Validation**: Test all pause/record/replay states
4. **Frame Action Correctness**: Verify batching and reset logic
5. **Bug Prevention**: Catch issues like actions being recorded while paused

## Implementation Order

1. **Extract ViewerCore** - Move logic out of main()
2. **Add State Machine** - Formalize recording/replay states
3. **Create Frame Manager** - Isolate action batching
4. **Write Unit Tests** - Test each component
5. **Integration Tests** - Test components together
6. **Fix Discovered Bugs** - Use tests to fix replay determinism issues

This refactoring will expose the bugs causing replay trajectory differences and make them fixable through proper testing.