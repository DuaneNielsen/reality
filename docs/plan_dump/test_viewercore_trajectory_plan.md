# ViewerCore Trajectory Verification Test Plan

## Objective
Write a comprehensive test for `viewer_core.cpp` (NOT Manager, NOT C API) that verifies ViewerCore correctly orchestrates recording, replay, and trajectory tracking to achieve deterministic replay.

## Key Point
- We are testing **ViewerCore** functionality, not Manager
- ViewerCore uses Manager as a dependency (passed via pointer)
- We create a real Manager instance only to provide it to ViewerCore
- All test interactions are through ViewerCore's interface

## Test Overview
Create a test that:
1. Uses ViewerCore to record 100 frames of forward movement with trajectory tracking
2. Uses ViewerCore to replay the recording with trajectory tracking
3. Verifies the two trajectory CSV files are identical (proving deterministic replay)

## Detailed Test Steps

### Phase 1: Setup and Recording
1. **Load Level**
   - Load `./tests/cpp/test_levels/quick_test.lvl` into a CompiledLevel struct
   
2. **Create Manager Instance**
   - Create Manager with CPU mode, 1 world, seed 42
   - Pass the loaded level to Manager
   - This is just setup - we're not testing Manager
   
3. **Create ViewerCore in Recording Mode**
   ```cpp
   ViewerCore::Config config;
   config.num_worlds = 1;
   config.rand_seed = 42;
   config.record_path = "test_viewercore_recording.rec";
   ViewerCore core(config, &manager);
   ```
   
4. **Start Recording with Trajectory Tracking**
   - Call `core.startRecording("test_viewercore_recording.rec")`
   - Call `core.toggleTrajectoryTracking(0)` to enable tracking for world 0
   - ViewerCore should start in PAUSED state (as per viewer behavior)
   
5. **Unpause and Simulate Movement**
   - Send SPACE key event to unpause: `core.handleInput(0, {KeyHit, Space})`
   - For 100 frames:
     - Send W key press: `core.handleInput(0, {KeyPress, W})`
     - Call `core.stepSimulation()`
     - This simulates holding W key for forward movement
   
6. **Stop Recording**
   - Call `core.stopRecording()`
   - Trajectory CSV should be written to file by Manager (via ViewerCore's commands)

### Phase 2: Replay with Trajectory Tracking
1. **Create New Manager Instance**
   - Create fresh Manager (same config, no level - will use embedded from recording)
   
2. **Create ViewerCore in Replay Mode**
   ```cpp
   ViewerCore::Config config;
   config.num_worlds = 1;
   config.replay_path = "test_viewercore_recording.rec";
   ViewerCore core(config, &manager);
   ```
   
3. **Load Replay and Enable Tracking**
   - Call `core.loadReplay("test_viewercore_recording.rec")`
   - Call `core.toggleTrajectoryTracking(0)` with different output file
   - Replay should start immediately (NOT paused, unlike recording)
   
4. **Step Through Replay**
   - For 100 frames:
     - Call `core.stepSimulation()`
     - ViewerCore should handle replay advancement internally
     - No input needed - replay ignores input
   
5. **Verify Replay Completion**
   - Check `core.getFrameState().should_exit` is true after replay

### Phase 3: Trajectory Verification
1. **Load Both CSV Files**
   - Use existing `TrajectoryComparer::parseTrajectoryFile()` helper
   - Parse "trajectory1.csv" (from recording)
   - Parse "trajectory2.csv" (from replay)
   
2. **Compare Trajectories**
   - Verify both have 100 points
   - Verify all points match (position, rotation, progress)
   - Use `TrajectoryComparer::compareTrajectories()` with small epsilon
   
3. **Verify Movement Characteristics**
   - Agent moved forward (Y increased)
   - Rotation stayed constant (no rotation input given)
   - Progress increased monotonically

## Test File Location
`tests/cpp/unit/test_viewercore_trajectory.cpp`

## Key ViewerCore Methods Being Tested
- `handleInput()` - Process keyboard events
- `stepSimulation()` - Advance simulation, handle recording/replay
- `startRecording()` / `stopRecording()` - Recording control
- `loadReplay()` - Replay control  
- `toggleTrajectoryTracking()` - Trajectory logging control
- `getFrameState()` - Query current state

## ViewerCore Internal Components Being Exercised
- `RecordReplayStateMachine` - State transitions for record/replay
- `FrameActionManager` - Action batching for recording
- `computeActionsFromInput()` - Convert keyboard to actions
- Input state management per world

## Expected Outcomes
1. Recording file created with 100 frames of actions
2. Two trajectory CSV files created (one during record, one during replay)
3. Trajectories are byte-for-byte identical
4. ViewerCore correctly manages state transitions:
   - Recording starts paused
   - Replay starts immediately
   - Pause/unpause works correctly
   - Exit flag set when replay completes

## What We're NOT Testing
- Manager's internal implementation
- C API functions
- Physics simulation accuracy
- Rendering
- Network features

## Success Criteria
The test passes if:
1. ViewerCore successfully records 100 frames
2. ViewerCore successfully replays those 100 frames
3. The two trajectory files are identical (deterministic replay achieved)
4. All state transitions occur as expected

## Implementation Notes
- Use real Manager instance, not mocks (need actual trajectory file writing)
- Use real level file for realistic testing
- Keep test focused on ViewerCore's orchestration role
- Verify through observable outputs (files, state) not internal Manager state