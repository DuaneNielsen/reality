# Test Coverage Analysis

## What We Actually Test vs. What We Think We Test

After reviewing the test suite, here's what each test file actually tests:

### 1. test_viewer_integration.cpp → **ManagerIntegrationTest**
**What it actually tests:** Manager C API functions, NOT viewer.cpp code

| Test Name | What It Actually Tests |
|-----------|------------------------|
| ManagerCreationWithLevelFile | `mer_create_manager()` C API |
| ManagerRecordingAPI | `mer_start_recording()`, `mer_stop_recording()` |
| ManagerReplayAPI | `mer_read_replay_metadata()`, `mer_load_replay()` |
| ManagerTrajectoryLogging | `mer_enable_trajectory_logging()` |
| MockViewerPauseResume | MockViewer pause logic (simulated, not real viewer) |
| MockViewerResetInput | `mer_trigger_reset()` with mock input |
| MockViewerTrajectoryToggle | Trajectory toggle with mock viewer |
| ManagerEmbeddedLevelRecording | Recording format with embedded levels |

**Viewer code coverage: ~10%** (only mock behaviors)

### 2. test_viewer_input.cpp → **ViewerInputMappingTest**
**What it actually tests:** Input-to-action mapping logic that MIRRORS viewer.cpp

| Test Name | What It Actually Tests |
|-----------|------------------------|
| WASDToMovementActions | Keyboard to move_amount/move_angle conversion |
| DiagonalInputMapping | 8-directional movement mapping |
| QERotationMapping | Q/E key to rotation values |
| ShiftSpeedModifier | Shift key affecting movement speed |
| ShiftRotationModifier | Shift key affecting rotation speed |
| RKeyTriggersReset | R key triggering reset |
| ComplexInputCombinations | Combined input processing |
| MultiFrameInputSequence | Input state across frames |

**Viewer code coverage: ~40%** (simulates viewer input logic)

### 3. test_viewer_errors.cpp → **OptionParsingAndFileErrorTest**
**What it actually tests:** Option parsing and file handling, minimal viewer code

| Test Name | What It Actually Tests |
|-----------|------------------------|
| MissingLevelFile | File I/O error handling |
| CorruptRecordingFile | `mer_read_replay_metadata()` error |
| InvalidViewerOptionCombinations | Option parser validation |
| InvalidNumericArguments | Argument parsing |
| FileExtensionValidation | String operations |
| ReplayMetadataMismatch | Metadata reading |
| ManagerGPUInitFailure | `mer_create_manager()` GPU errors |
| TrajectoryFileWriteError | File write permissions |
| LargeRecordingFileCreation | Recording file size |
| UnknownCommandLineOptions | Option parser behavior |

**Viewer code coverage: ~5%** (only option parsing logic)

### 4. test_viewer_workflows.cpp → **SimulatedViewerWorkflowTest**
**What it actually tests:** Manager API with simulated viewer workflows

| Test Name | What It Actually Tests |
|-----------|------------------------|
| MockViewerRecordingSession | Recording workflow with mock pause/play |
| ManagerReplayDeterminism | Manager replay determinism |
| MockViewerTrajectoryWorkflow | Trajectory toggle workflow |
| ManagerMultiWorldRecording | Multi-world Manager API |
| MockViewerPauseDuringRecording | Pause state with recording |

**Viewer code coverage: ~15%** (simulated workflows)

## Actual Viewer Code Not Tested

The following viewer.cpp behaviors are NOT tested:

1. **Real Window Creation**: `WindowManager`, `WindowHandle`, GPU initialization
2. **Rendering Pipeline**: `viz::Viewer`, render manager, camera controls
3. **Frame Action Batching**: The viewer's `frame_actions` array management
4. **Action Reset Logic**: Resetting actions to defaults each frame
5. **Replay Auto-Stop**: `viewer.stopLoop()` when replay finishes
6. **World Count Auto-Adjustment**: Adjusting `num_worlds` from replay metadata
7. **Recording Start State**: Starting paused with "press SPACE to start"
8. **Main Loop Integration**: The actual `viewer.loop()` with real callbacks
9. **ImGui Menu**: `--hide-menu` option and UI interaction
10. **Camera Controls**: Initial camera position/rotation setup

## Summary

**Total actual viewer.cpp coverage: ~15-20%**

Most tests actually test:
- Manager C API (60%)
- Mock implementations (20%)
- Option parsing (10%)
- Simulated input mapping (10%)

To truly test viewer.cpp, we would need:
- Integration tests that run the actual viewer binary
- Tests that exercise the real main() function
- Window/rendering mocks that work with the real viewer code
- Tests for frame action batching and reset logic
- Tests for the actual pause/record/replay state machine