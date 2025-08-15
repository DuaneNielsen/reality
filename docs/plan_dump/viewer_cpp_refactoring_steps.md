# Viewer.cpp Refactoring Implementation Plan

## Current State
- ViewerCore library is built and contains the testable logic
- RecordReplayStateMachine and FrameActionManager are extracted
- viewer.cpp still contains all the mixed logic and I/O

## Refactoring Steps

### Step 1: Prepare ViewerCore Integration
1. Include viewer_core.hpp in viewer.cpp
2. Remove duplicate state variables that now live in ViewerCore:
   - `is_paused` 
   - `is_recording`
   - `has_replay`
   - `frame_actions` vector
   - Track trajectory variables

### Step 2: Create ViewerCore Instance
Replace the manual state management with ViewerCore initialization:

```cpp
// Before Manager creation, prepare ViewerCore config
ViewerCore::Config core_config{
    .num_worlds = num_worlds,
    .rand_seed = rand_seed,
    .auto_reset = has_replay || is_recording,
    .load_path = load_path,
    .record_path = record_path,
    .replay_path = replay_path
};

// After Manager creation
ViewerCore core(core_config, &mgr);
```

### Step 3: Refactor Input Callbacks
Transform the viewer.loop callbacks to delegate to ViewerCore:

#### World-level input callback (Reset, Track, Pause):
```cpp
[&core](CountT world_idx, const Viewer::UserInput &input) {
    using Key = Viewer::KeyboardKey;
    
    // Convert Viewer input to ViewerCore events
    if (input.keyHit(Key::R)) {
        core.handleInput(world_idx, {
            ViewerCore::InputEvent::KeyHit, 
            ViewerCore::InputEvent::R
        });
    }
    
    if (input.keyHit(Key::T)) {
        core.handleInput(world_idx, {
            ViewerCore::InputEvent::KeyHit,
            ViewerCore::InputEvent::T
        });
    }
    
    if (input.keyHit(Key::Space)) {
        core.handleInput(world_idx, {
            ViewerCore::InputEvent::KeyHit,
            ViewerCore::InputEvent::Space
        });
    }
}
```

#### Agent control callback (WASD movement):
```cpp
[&core](CountT world_idx, CountT agent_idx, const Viewer::UserInput &input) {
    using Key = Viewer::KeyboardKey;
    
    // Convert key states to ViewerCore events
    auto sendKeyEvent = [&](Key vk, ViewerCore::InputEvent::Key ck, bool pressed) {
        if (pressed != core.wasKeyPressed(world_idx, ck)) {
            core.handleInput(world_idx, {
                pressed ? ViewerCore::InputEvent::KeyPress 
                        : ViewerCore::InputEvent::KeyRelease,
                ck
            });
        }
    };
    
    sendKeyEvent(Key::W, ViewerCore::InputEvent::W, input.keyPressed(Key::W));
    sendKeyEvent(Key::A, ViewerCore::InputEvent::A, input.keyPressed(Key::A));
    sendKeyEvent(Key::S, ViewerCore::InputEvent::S, input.keyPressed(Key::S));
    sendKeyEvent(Key::D, ViewerCore::InputEvent::D, input.keyPressed(Key::D));
    sendKeyEvent(Key::Q, ViewerCore::InputEvent::Q, input.keyPressed(Key::Q));
    sendKeyEvent(Key::E, ViewerCore::InputEvent::E, input.keyPressed(Key::E));
    sendKeyEvent(Key::Shift, ViewerCore::InputEvent::Shift, input.keyPressed(Key::Shift));
    
    // Let core compute and apply actions
    core.updateFrameActions(world_idx, agent_idx);
}
```

#### Step callback:
```cpp
[&core, &viewer]() {
    // Core handles all simulation logic
    core.stepSimulation();
    
    // Check if we should exit
    auto state = core.getFrameState();
    if (state.should_exit) {
        viewer.stopLoop();
    }
}
```

### Step 4: Remove Redundant Code
Delete from viewer.cpp:
1. Manual replay step function
2. Frame action management code  
3. Recording/replay state management
4. Direct mgr.setAction() calls (now handled by core)
5. Direct mgr.startRecording/stopRecording calls
6. Manual pause state tracking

### Step 5: Simplify Initialization
Move complex initialization logic into ViewerCore:
- Replay metadata reading
- Recording setup
- Initial pause state setting

The main() function becomes focused on:
1. Argument parsing
2. Window/GPU setup
3. Manager creation
4. ViewerCore creation
5. Viewer loop with thin I/O adapters

### Step 6: Add Debug Helpers (Optional)
Add methods to ViewerCore for debugging:
```cpp
void ViewerCore::dumpState() const {
    printf("State Machine: %s\n", getStateString(state_machine_.getState()));
    printf("Is Paused: %s\n", state_machine_.isPaused() ? "Yes" : "No");
    printf("Frame Actions: ");
    for (auto a : action_manager_.getFrameActions()) {
        printf("%d ", a);
    }
    printf("\n");
}
```

## Testing Strategy After Refactoring

### 1. Manual Testing
- Test record → pause → resume → stop workflow
- Test replay with pause/resume
- Verify trajectory tracking toggle
- Check reset functionality

### 2. Integration Tests  
Create integration tests that:
- Record a sequence of actions
- Save to file
- Load and replay
- Verify deterministic replay

### 3. Regression Tests
Ensure no regressions in:
- Multi-world support
- GPU mode
- Trajectory logging
- Level loading

## Benefits of This Refactoring

1. **Testability**: Core logic can be unit tested without window/GPU dependencies
2. **Maintainability**: Clear separation between I/O and logic
3. **Debuggability**: State machine makes it easy to track recording/replay state
4. **Reusability**: ViewerCore could be used in other contexts (e.g., headless mode with synthetic input)
5. **Bug Prevention**: Explicit state transitions prevent invalid state combinations

## Risks and Mitigations

### Risk 1: Breaking Existing Functionality
**Mitigation**: Keep old viewer.cpp as viewer_old.cpp during transition

### Risk 2: Performance Impact
**Mitigation**: ViewerCore adds minimal overhead (one indirection layer)

### Risk 3: Complex Input Mapping
**Mitigation**: Create InputAdapter helper class if mapping becomes complex

## Implementation Order

1. **Phase 1**: Basic integration (2-3 hours)
   - Create ViewerCore instance
   - Wire up basic callbacks
   - Test basic movement

2. **Phase 2**: Recording/Replay (1-2 hours)
   - Migrate recording logic
   - Migrate replay logic
   - Test record/replay cycle

3. **Phase 3**: Polish (1 hour)
   - Remove dead code
   - Add debug output
   - Document changes

4. **Phase 4**: Testing (2-3 hours)
   - Write integration tests
   - Fix discovered bugs
   - Verify all workflows

## Success Criteria

The refactoring is successful when:
1. All existing viewer functionality works identically
2. The recording pause bug is fixed (record while paused → replay shows correct trajectory)
3. Unit tests pass for ViewerCore components
4. Code is cleaner and more maintainable
5. New features (like input replay from file) become trivial to add