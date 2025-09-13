# Complete Multi-Level Replay Implementation Plan

## Validation from C++ Tests

The C++ test at `tests/cpp/e2e/test_viewer_workflows.cpp` (lines 240-266) shows the CORRECT pattern:
1. Read metadata from replay
2. **Extract embedded level from replay** (currently done, but not used everywhere)
3. Create Manager with embedded level + replay parameters
4. Load replay actions

This pattern needs to be encapsulated and used everywhere.

## Phase 1: Add Manager::fromReplay() Static Method

**Location**: `src/mgr.cpp` and `src/mgr.hpp`

```cpp
static std::unique_ptr<Manager> Manager::fromReplay(
    const std::string& filepath,
    ExecMode execMode,
    int gpuID,
    bool enableBatchRenderer = false)
{
    // 1. Read metadata
    auto metadata = readReplayMetadata(filepath);
    if (!metadata.has_value()) return nullptr;
    
    // 2. Extract ALL embedded levels (v3 format has one per world)
    auto levels = ReplayLoader::loadAllEmbeddedLevels(filepath);
    if (!levels.has_value()) return nullptr;
    
    // 3. Build config from replay data
    Config cfg;
    cfg.execMode = execMode;
    cfg.gpuID = gpuID;
    cfg.numWorlds = metadata->num_worlds;
    cfg.randSeed = metadata->seed;
    cfg.autoReset = true;
    cfg.enableBatchRenderer = enableBatchRenderer;
    cfg.perWorldCompiledLevels = levels.value();
    
    // 4. Create manager with replay configuration
    auto mgr = std::make_unique<Manager>(cfg);
    
    // 5. Load replay actions
    if (!mgr->loadReplay(filepath)) {
        return nullptr;
    }
    
    return mgr;
}
```

## Phase 2: Add C API Function

**Location**: `src/madrona_escape_room_c_api.cpp` and `.h`

```c
MER_ManagerHandle mer_create_manager_from_replay(
    const char* filepath,
    MER_ExecMode exec_mode,
    int32_t gpu_id,
    bool enable_batch_renderer)
```

## Phase 3: Fix Python SimManager.from_replay()

**Location**: `madrona_escape_room/manager.py`

```python
@classmethod
def from_replay(cls, replay_filepath, exec_mode, gpu_id, enable_batch_renderer=False):
    # Single C call that handles everything
    handle = lib.mer_create_manager_from_replay(
        replay_filepath.encode('utf-8'),
        exec_mode.value,
        gpu_id,
        enable_batch_renderer
    )
    if not handle:
        raise RuntimeError(f"Failed to create manager from replay: {replay_filepath}")
    
    # Create Python wrapper
    manager = cls.__new__(cls)
    manager._handle = handle
    
    # Read metadata to set Python attributes
    metadata = cls.read_replay_metadata(replay_filepath)
    manager.num_worlds = metadata.num_worlds
    manager.seed = metadata.seed
    
    return manager
```

## Phase 4: Update Headless Mode

**Location**: `src/headless.cpp` (around line 250)

```cpp
if (replay_mode) {
    // Use the new factory method
    auto mgr = Manager::fromReplay(replay_file, exec_mode, gpu_id);
    if (!mgr) {
        std::cerr << "Failed to create manager from replay\n";
        return 1;
    }
    // Continue with mgr...
} else {
    // Normal creation path with levels from --load or embedded default
    Manager mgr(cfg);
    // ...
}
```

## Phase 5: Update Viewer Mode

**Location**: `src/viewer.cpp`

Same pattern as headless - use `Manager::fromReplay()` when `--replay` is specified

## Phase 6: Add C++ Test for fromReplay

**Location**: `tests/cpp/integration/test_manager_from_replay.cpp`

```cpp
TEST(ManagerFromReplay, CreatesWithEmbeddedLevels) {
    // Record with specific level
    {
        auto level = createTestLevel();
        Config cfg;
        cfg.numWorlds = 2;
        cfg.perWorldCompiledLevels = {level, level};
        
        Manager mgr(cfg);
        mgr.startRecording("test.rec");
        for (int i = 0; i < 10; i++) {
            mgr.step();
        }
        mgr.stopRecording();
    }
    
    // Create from replay - should use embedded level
    auto replay_mgr = Manager::fromReplay("test.rec", ExecMode::CPU, -1);
    ASSERT_NE(replay_mgr, nullptr);
    
    // Verify it works
    for (int i = 0; i < 10; i++) {
        replay_mgr->replayStep();
        replay_mgr->step();
    }
}
```

## Phase 7: Update Existing C++ Test

**Location**: `tests/cpp/e2e/test_viewer_workflows.cpp` (line 240-266)

Replace manual steps with:

```cpp
// Phase 2: Replay using fromReplay
auto replay_mgr = Manager::fromReplay("demo.rec", ExecMode::CPU, -1);
ASSERT_NE(replay_mgr, nullptr);
// Continue with replay...
```

## What This Fixes

- **Ensures replay always uses embedded levels** from the recording file
- **Single implementation** used by Python, C++, headless, and viewer
- **No possibility of level mismatch** - the replay file is self-contained
- **Fixes viewer divergence** because correct levels are always used

## Clean Implementation

- No duplication across different entry points
- Clear separation: replay files are self-contained with levels + actions
- Factory pattern: `Manager::fromReplay()` handles all complexity internally

## Root Cause Analysis

The viewer divergence occurs because:
1. **Recording**: Saves actions for worlds with specific level geometries
2. **Current Bug**: Replay creates Manager with current/default levels instead of embedded levels
3. **Replay Execution**: Applies recorded actions to WRONG level geometry
4. **Result**: Agents hit walls or move incorrectly because recorded actions don't match current level layout

## Architecture Benefits

- **Batch Mode Preserved**: All worlds still step together synchronously
- **Self-Contained Replay Files**: Include both level data and action sequences
- **Guaranteed Consistency**: Impossible to load replay with wrong levels
- **Single Source of Truth**: One implementation of replay loading logic