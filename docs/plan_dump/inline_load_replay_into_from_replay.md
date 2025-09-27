# Plan: Inline load_replay() into from_replay() and Remove load_replay() API

## Overview

Remove the problematic `load_replay()` method entirely by inlining its functionality into `from_replay()`. This eliminates the entire class of bugs caused by loading replay data into existing Managers with wrong simulation state.

## 1. Pre-reading List

### Core Implementation Files
- `src/mgr.cpp` - Study `Manager::loadReplay()` (line ~1470) and `Manager::fromReplay()` (line ~1730)
- `src/mgr.hpp` - Review Manager class declarations and `loadReplay()` signature (line ~243)
- `src/madrona_escape_room_c_api.cpp` - Examine `mer_load_replay()` C API implementation (line ~501)
- `src/madrona_escape_room_c_api.h` - Check C API declaration (line ~125)

### Python API Files
- `madrona_escape_room/manager.py` - Review `load_replay()` method (line ~211) and deprecation warnings
- `madrona_escape_room/ctypes_bindings.py` - Check C API bindings for `mer_load_replay` (line ~299)

### Viewer Implementation
- `src/viewer_core.cpp` - Examine `ViewerCore::loadReplay()` usage (line ~384)
- `src/viewer_core.hpp` - Check ViewerCore interface (line ~109)
- `src/viewer.cpp` - Check viewer application usage (line ~454)

### Test Files to Understand Usage Patterns
- `tests/python/test_checksum_verification.py` - Current failing test using `load_replay()`
- `tests/python/test_native_replay.py` - Mixed usage of both APIs
- `tests/cpp/unit/test_viewercore_trajectory.cpp` - C++ ViewerCore tests

### Documentation
- `docs/specs/mgr.md` - Review `loadReplay` specification (line ~538)

## 2. Instructions to Inline load_replay into from_replay

### Step 2.1: Extract load_replay Logic
In `src/mgr.cpp`, copy the entire body of `Manager::loadReplay()` method:
- File reading and metadata parsing
- Level loading logic
- Action data reading
- Checksum verification points loading
- ReplayData construction

### Step 2.2: Inline into fromReplay
In `Manager::fromReplay()` method, replace this line:
```cpp
if (!mgr->loadReplay(filepath)) {
    std::cerr << "Error: Failed to load replay actions from " << filepath << "\n";
    return nullptr;
}
```

With the complete inlined logic from `loadReplay()`:
```cpp
// [INLINE] Begin loadReplay logic
std::ifstream replay_file(filepath, std::ios::binary);
if (!replay_file.is_open()) {
    std::cerr << "Error: Failed to open replay file: " << filepath << "\n";
    return nullptr;
}

// [Continue with all loadReplay logic...]
// - Read metadata header
// - Validate metadata
// - Read world levels
// - Read mixed ACTION/CHECKSUM records
// - Construct ReplayData
mgr->impl_->replayData = ReplayData{metadata, std::move(actions), std::move(checksums), std::move(checksum_steps)};
mgr->impl_->currentReplayStep = 0;
// [END INLINE] No reset needed - Manager is fresh
```

### Step 2.3: Handle Error Returns
Convert `Manager::loadReplay()` return values to `fromReplay()` return pattern:
- `return false` → `return nullptr`
- Ensure all error paths return `nullptr`

## 3. Remove load_replay from C++ and C API

### Step 3.1: Remove C++ Implementation
- Delete `Manager::loadReplay()` method from `src/mgr.cpp` (line ~1470)
- Remove `loadReplay` declaration from `src/mgr.hpp` (line ~243)

### Step 3.2: Remove C API
- Delete `mer_load_replay()` function from `src/madrona_escape_room_c_api.cpp` (line ~501)
- Remove `MER_EXPORT MER_Result mer_load_replay(...)` declaration from `src/madrona_escape_room_c_api.h` (line ~125)

### Step 3.3: Remove C API Bindings
- Remove `mer_load_replay` bindings from `madrona_escape_room/ctypes_bindings.py` (lines ~299-300):
  ```python
  lib.mer_load_replay.argtypes = [MER_ManagerHandle, c_char_p]
  lib.mer_load_replay.restype = c_int
  ```

## 4. Update Viewer to Use Only from_replay

### Step 4.1: Modify ViewerCore Interface
In `src/viewer_core.hpp`:
- Change `void loadReplay(const std::string& path)` to take additional parameters or redesign
- Consider: `void loadReplay(const std::string& path, ExecMode execMode, int gpuID = -1)`

### Step 4.2: Update ViewerCore Implementation
In `src/viewer_core.cpp`:
```cpp
void ViewerCore::loadReplay(const std::string& path, ExecMode execMode, int gpuID) {
    // Replace mgr_->loadReplay(path) with:
    auto replay_mgr = Manager::fromReplay(path, execMode, gpuID);
    if (!replay_mgr) {
        std::cerr << "Failed to create replay manager\n";
        return;
    }

    // Replace existing manager with replay manager
    mgr_ = std::move(replay_mgr);
    state_machine_.startReplay();
}
```

### Step 4.3: Update Viewer Application
In `src/viewer.cpp` (line ~454):
- Update call to `viewer_core.loadReplay()` to pass execution mode parameters
- Extract exec mode from viewer configuration

## 5. Remove from Python API

### Step 5.1: Remove Python Method
In `madrona_escape_room/manager.py`:
- Delete entire `load_replay()` method (lines ~211-235)
- Remove deprecation warning imports if no longer used

### Step 5.2: Remove Internal Helper
- Delete `_load_replay_internal()` method (line ~403) if it exists

### Step 5.3: Update Python Tests
Convert all Python tests from `mgr.load_replay()` to `SimManager.from_replay()` pattern:
```python
# OLD PATTERN:
mgr.load_replay(recording_path)

# NEW PATTERN:
replay_mgr = SimManager.from_replay(recording_path, ExecMode.CPU)
```

## 6. Update docs/specs/mgr.md

### Step 6.1: Remove loadReplay Section
- Delete entire `#### loadReplay` section (line ~538)
- Remove all references to `loadReplay()` method

### Step 6.2: Update Cross-References
Search and update all mentions:
- "Replay must be loaded via loadReplay() or Manager created via from_replay()" → Update to only mention `from_replay()`
- "Returns true after successful loadReplay() or from_replay creation" → Update to only mention `from_replay()`

### Step 6.3: Update Examples
Replace any example code showing `loadReplay()` usage with `from_replay()` examples.

## 7. Search and Update Tests Using load_replay()

### Step 7.1: Audit Test Files
Files to check based on grep results:
- `tests/python/test_checksum_verification.py` - 4 uses
- `tests/python/test_native_replay.py` - 8 uses
- `tests/python/test_multi_level_recording_replay.py` - 1 use
- `tests/cpp/unit/test_viewercore_trajectory.cpp` - 3 uses

### Step 7.2: Conversion Strategy

For each test using `load_replay()`:

**Option A: Convert to from_replay**
```python
# Before:
mgr.load_replay(recording_path)

# After:
replay_mgr = SimManager.from_replay(recording_path, ExecMode.CPU)
```

**Option B: Remove if Duplicate**
- If equivalent `from_replay()` test already exists, remove the `load_replay()` version
- Prefer keeping tests that use the correct API

### Step 7.3: Specific Test Updates

**test_checksum_verification.py:**
- Convert all 4 test functions to use `from_replay()`
- Update test assertions to work with new manager instance

**test_native_replay.py:**
- Lines showing "Also test old interface for compatibility" - remove these sections
- Keep core functionality tests but use `from_replay()`

**ViewerCore C++ tests:**
- Update to use new ViewerCore interface requiring exec mode parameters

### Step 7.4: Test Validation
After conversions:
- Run all affected tests to ensure they still pass
- Verify no tests are accidentally testing the same functionality twice
- Ensure test coverage remains equivalent

## Implementation Order

1. **Step 2**: Inline logic first (preserves functionality)
2. **Step 7**: Update tests to use `from_replay()`
3. **Step 5**: Remove Python API
4. **Step 4**: Update ViewerCore (may require careful manager lifecycle management)
5. **Step 3**: Remove C++ implementation
6. **Step 6**: Update documentation

## Expected Benefits

- **Eliminates entire bug class**: No more non-deterministic replay issues
- **Simplifies API**: Only one correct way to load replays
- **Improves code quality**: Removes deprecated/broken functionality
- **Better testing**: All tests use the correct API pattern

## Risk Mitigation

- **Test thoroughly**: Ensure ViewerCore manager replacement works correctly
- **Manager lifecycle**: Verify proper cleanup when replacing manager in ViewerCore
- **Build verification**: Ensure all references are found and updated during removal