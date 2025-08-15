# ViewerCore Unit Test Plan

## Test Strategy
Create comprehensive unit tests for ViewerCore components using GoogleTest framework. Focus on testing the pure logic without requiring actual Manager or rendering dependencies.

## Component Test Coverage

### 1. RecordReplayStateMachine Tests

#### Basic State Tests
- **InitialState**: Verify machine starts in Idle state
- **RecordingStartsPaused**: Verify recording begins in paused state
- **ReplayStartsImmediately**: Verify replay begins without pause
- **StateGetters**: Test isPaused(), isRecording(), isReplaying()

#### State Transition Tests
- **IdleToRecording**: Idle → RecordingPaused transition
- **RecordingPauseToggle**: RecordingPaused ↔ Recording transitions
- **ReplayPauseToggle**: ReplayingPaused ↔ Replaying transitions
- **ReplayFinish**: Replaying → ReplayFinished transition
- **StopToIdle**: Any state → Idle via stop()

#### Edge Case Tests
- **InvalidTransitions**: Test transitions that should be no-ops
- **MultipleStarts**: Starting recording/replay when already started
- **PauseInInvalidStates**: Toggling pause in Idle/Finished states
- **FinishInInvalidStates**: Calling finishReplay when not replaying

#### Recording Control Tests
- **ShouldRecordFrame**: Returns true only in Recording state
- **ShouldAdvanceReplay**: Returns true only in Replaying state
- **RecordingWhilePaused**: Verify no frames recorded when paused

### 2. FrameActionManager Tests

#### Initialization Tests
- **DefaultValues**: All actions initialize to correct defaults (0,0,2)
- **CorrectSize**: Frame actions vector sized correctly for num_worlds
- **HasChangesFlag**: Initially false, true after modifications

#### Action Management Tests
- **SetSingleAction**: Set action for one world, verify storage
- **SetMultipleActions**: Set actions for multiple worlds
- **ResetToDefaults**: Verify reset clears all actions to defaults
- **OutOfBoundsHandling**: Setting action for invalid world index

#### Action Encoding Tests
- **ActionPacking**: Verify 3-value packing (move_amount, move_angle, rotate)
- **IndependentWorlds**: Actions for one world don't affect others
- **GetFrameActions**: Retrieved vector matches set values

### 3. ViewerCore Integration Tests

#### Construction Tests
- **ConfigParsing**: Different config combinations initialize correctly
- **RecordModeInit**: Record path triggers paused recording state
- **ReplayModeInit**: Replay path triggers immediate replay state
- **NormalModeInit**: No record/replay paths = normal operation

#### Input Handling Tests
- **MovementKeys**: W/A/S/D generate correct move vectors
- **RotationKeys**: Q/E generate correct rotation values
- **ShiftModifier**: Shift affects move speed correctly
- **SpaceKeyPause**: Space toggles pause in appropriate states
- **ResetKey**: R key triggers reset for specific world
- **TrackKey**: T key toggles trajectory tracking

#### Action Computation Tests
- **SingleKeyMovement**: W→forward, S→backward, etc.
- **DiagonalMovement**: W+D→forward-right, etc.
- **StopWhenNoKeys**: No keys pressed → stop action
- **RotationLogic**: Q/E with/without shift modifier
- **ActionBoundaries**: Values stay within valid ranges

#### Frame Management Tests
- **FrameActionReset**: Actions reset after step
- **RecordingFrameCapture**: Actions captured only when recording
- **PausedNoCapture**: No action recording when paused
- **ReplayIgnoresInput**: Input ignored during replay

#### Mock Manager Tests
Since we can't use real Manager in unit tests, create MockManager:

```cpp
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
    
    void step() { stepCount++; }
};
```

### 4. Recording Workflow Tests

#### Complete Recording Cycle
1. Start recording (paused)
2. Verify no frames recorded while paused
3. Unpause with space
4. Input movement commands
5. Verify frames recorded
6. Pause again
7. Verify recording stops
8. Stop recording
9. Verify state returns to idle

#### Recording State Consistency
- **PausePreservesState**: Pausing doesn't lose recording state
- **ActionsResetPerFrame**: Each frame starts with default actions
- **ConsecutiveFrames**: Multiple frames record correctly

### 5. Replay Workflow Tests

#### Complete Replay Cycle
1. Load replay
2. Verify starts immediately (not paused)
3. Step through replay
4. Verify input ignored
5. Reach end
6. Verify exit flag set

#### Replay Control Tests
- **PauseDuringReplay**: Pause stops replay advancement
- **ResumeReplay**: Unpause resumes replay
- **ReplayCompletion**: Proper state on finish

### 6. Edge Cases and Error Handling

#### Boundary Conditions
- **MaxWorlds**: Test with maximum supported worlds
- **EmptyConfig**: Handle empty paths gracefully
- **InvalidPaths**: Non-existent replay files
- **ZeroWorlds**: Handle edge case of 0 worlds

#### State Machine Robustness
- **RapidStateChanges**: Fast pause/unpause cycles
- **ConcurrentOperations**: Recording while trying to replay
- **UnexpectedTransitions**: Invalid state transitions

### 7. Performance Tests (Optional)

#### Efficiency Tests
- **LargeWorldCount**: Performance with many worlds
- **RapidInputChanges**: Handle high-frequency input
- **MemoryUsage**: Verify no memory leaks

## Test Implementation Priority

### Phase 1: Core Components (High Priority)
1. RecordReplayStateMachine basic tests
2. FrameActionManager basic tests
3. Basic ViewerCore construction

### Phase 2: Integration (Medium Priority)
1. Input handling tests
2. Action computation tests
3. Recording workflow tests

### Phase 3: Advanced (Low Priority)
1. Edge cases
2. Error handling
3. Performance tests

## Success Criteria

Tests are successful when:
1. All state transitions work as documented
2. Recording starts paused and captures frames only when unpaused
3. Replay ignores user input and advances deterministically
4. Frame actions reset properly between steps
5. All edge cases handled without crashes
6. Mock verification confirms correct Manager interaction

## Known Issues to Test For

### Critical Bugs
1. **Recording While Paused Bug**: Ensure no actions recorded when paused
2. **Frame Action Persistence**: Actions must reset between frames
3. **Replay Determinism**: Same replay produces same trajectory

### Integration Issues
1. **Input State Leakage**: Keys from one world affecting another
2. **State Machine Corruption**: Invalid states after edge cases
3. **Memory Management**: Proper cleanup in destructors

## Test Data Requirements

### Mock Data
- Sample replay files (can be synthetic)
- Test level configurations
- Input sequences for determinism testing

### Test Fixtures
```cpp
class ViewerCoreTestFixture : public ::testing::Test {
protected:
    std::unique_ptr<MockManager> mockMgr;
    std::unique_ptr<ViewerCore> core;
    
    void SetUp() override {
        mockMgr = std::make_unique<MockManager>();
    }
    
    void createCore(ViewerCore::Config config) {
        core = std::make_unique<ViewerCore>(config, mockMgr.get());
    }
    
    // Helper methods
    void simulateKeyPress(int world, ViewerCore::InputEvent::Key key);
    void simulateKeyRelease(int world, ViewerCore::InputEvent::Key key);
    void verifyAction(int world, int moveAmount, int moveAngle, int rotate);
};
```

## Coverage Goals

- Line coverage: > 90%
- Branch coverage: > 85%
- State transition coverage: 100%
- Edge case coverage: 100%

## Documentation

Each test should include:
1. Clear test name describing what's being tested
2. Comments explaining the test scenario
3. Explicit assertions with meaningful messages
4. Cleanup verification where applicable