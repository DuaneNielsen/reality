# Recording/Replay Integration Plan - Simplified

## Executive Summary
Implement recording/replay with embedded level data and explicit level requirements. No default levels - user must always provide either a .lvl file or .rec file.

## Problem Statement
- Tests use custom ASCII levels but recordings don't capture level data
- Need seamless workflow: test → record → replay with exact environment
- Eliminate ambiguity by requiring explicit level specification

## Solution Architecture

### File Format Specifications

**Level Files (.lvl)**
- Pure `MER_CompiledLevel` binary data
- Created by level compiler from ASCII art

**Recording Files (.rec)**  
- Format: `[MER_ReplayMetadata] [MER_CompiledLevel] [ActionFrames...]`
- Always embed current level data (no flags needed)
- Self-contained replay files

### Command Line Interface

**Explicit Level Requirements:**
```bash
# Load level, start live simulation
./build/viewer --load level.lvl

# Replay recording (uses embedded level), then continue live
./build/viewer --replay session.rec  

# Load level, start live simulation + recording
./build/viewer --load level.lvl --record output.rec

# ERROR: No level specified
./build/viewer
```

**Conflict Rules:**
- `--replay` + `--record` → ERROR (cannot record while replaying)
- `--load` + `--replay` → ERROR (conflicting operations)
- No level specified → ERROR (must provide --load or --replay)

### Implementation Components

## TODO List

### 1. Level Compiler Binary Output Mode
**File**: `madrona_escape_room/level_compiler.py`
- [ ] Add `save_compiled_level_binary(compiled_dict, filepath)` function
- [ ] Implement binary serialization of `MER_CompiledLevel` struct
- [ ] Add command line interface: `python -m madrona_escape_room.level_compiler input.txt output.lvl`
- [ ] Add error handling for file I/O operations

### 2. Recording Format Simplification
**Files**: `include/madrona_escape_room_c_api.h`, `src/madrona_escape_room_c_api.cpp`
- [ ] Keep `MER_ReplayMetadata` struct unchanged (no new flags)
- [ ] Modify recording implementation to always embed current level data
- [ ] Update `mer_start_recording()` to capture and store current level
- [ ] Ensure consistent binary format: metadata → level → actions

### 3. Viewer Interface - Explicit Level Requirements
**File**: `src/viewer.cpp`
- [ ] Add `--load <file>` command line option for .lvl files
- [ ] Add `--record <file>` command line option for .rec output  
- [ ] Update `--replay <file>` to work with .rec files
- [ ] Remove default level support - require explicit --load or --replay
- [ ] Implement conflict detection and error messages
- [ ] Add file extension validation (.lvl vs .rec)

### 4. Level Loading Implementation
**File**: `src/viewer.cpp`
- [ ] Add `load_level_file()` function for .lvl files
- [ ] Add `load_recording_file()` function for .rec files
- [ ] Implement manager creation with loaded level data
- [ ] Add error handling for missing/corrupted files
- [ ] Remove default level initialization code

### 5. Recording Implementation
**File**: `src/madrona_escape_room_c_api.cpp`
- [ ] Update recording to always embed current level data
- [ ] Detect current level from manager state
- [ ] Write level data after metadata in .rec files
- [ ] Ensure recording works regardless of level source (--load or previous replay)

### 6. Headless Mode Integration  
**File**: `src/headless.cpp`
- [ ] Add same explicit level requirements as viewer
- [ ] Implement `--load`, `--replay`, `--record` options
- [ ] Remove default level support
- [ ] Add `--continue-steps` option for post-replay simulation
- [ ] Ensure consistent error handling

### 7. Python Bindings Updates
**File**: `madrona_escape_room/ctypes_bindings.py`
- [ ] Add functions for .lvl file I/O
- [ ] Add binary level file loading support
- [ ] Update recording functions to always embed level data
- [ ] Remove default level creation methods

### 8. SimManager Integration
**File**: `madrona_escape_room/__init__.py`
- [ ] Add `SimManager.from_level_file(filepath)` class method
- [ ] Update recording context managers to always embed level data
- [ ] Ensure `SimManager.from_replay()` uses embedded level data
- [ ] Remove default level fallbacks where appropriate

### 9. pytest Integration
**Files**: Test configuration
- [ ] Update test recording to use .rec extension
- [ ] Ensure test recordings always embed level data when `level_ascii` used
- [ ] Update `--record-actions` flag to create .rec files
- [ ] Verify `--visualize` flag works with .rec files
- [ ] Handle tests that don't use custom levels (still need to embed default)

### 10. Error Handling Enhancement
**Files**: Viewer, headless, Python bindings
- [ ] Clear error messages when no level specified
- [ ] Helpful suggestions in error messages ("Use --load or --replay")
- [ ] Validate file extensions and provide clear feedback
- [ ] Handle file not found scenarios gracefully

### 11. Testing and Validation
- [ ] Test .lvl file creation and loading
- [ ] Test recording always embeds level data (custom and default)
- [ ] Test replay uses embedded level data correctly
- [ ] Verify error handling for missing level specifications
- [ ] Test conflict detection for invalid flag combinations
- [ ] Validate live continuation after replay completion

### 12. Documentation Updates
- [ ] Update viewer help text emphasizing level requirements
- [ ] Update headless help text  
- [ ] Create examples in CLAUDE.md showing explicit level usage
- [ ] Update recording/debugging documentation
- [ ] Document .lvl and .rec file formats

## Expected Workflows

### Level Development
```bash
# 1. Create ASCII level
echo "##########\n#S......#\n##########" > maze.txt

# 2. Compile to binary level  
python -m madrona_escape_room.level_compiler maze.txt maze.lvl

# 3. Test interactively (explicit level required)
./build/viewer --load maze.lvl

# 4. Record gameplay session  
./build/viewer --load maze.lvl --record playtest.rec

# 5. Replay session (uses embedded level from .rec)
./build/viewer --replay playtest.rec
```

### Test-Driven Development
```bash
# 1. Test with custom level creates .rec with embedded level
pytest test_maze_navigation.py --record-actions

# 2. Replay exact test environment
./build/viewer --replay test_recordings/test_maze_navigation/actions.rec

# 3. Performance test with embedded level
./build/headless --replay test_recordings/test_maze_navigation/actions.rec --continue-steps 5000
```

## Key Simplifications
- **No flags in recording format** - level data always embedded
- **No default level support** - explicit level always required
- **Clear error messages** - when level not specified
- **Simplified logic** - no conditional level loading
- **Self-contained recordings** - .rec files have everything needed

## Success Criteria  
- [ ] Viewer/headless error when no level specified
- [ ] .lvl files load correctly for live simulation
- [ ] .rec files always contain embedded level data
- [ ] Replay shows exact environment as original
- [ ] Clear error messages for invalid flag combinations
- [ ] pytest integration creates .rec files with embedded levels
- [ ] Live continuation after replay works seamlessly

## Implementation Priority
1. Level compiler binary output (enables .lvl creation)
2. Recording format enhancement (enables level embedding)  
3. Viewer three-flag interface (enables new workflows)
4. Python integration updates (enables pytest integration)
5. Testing and validation (ensures correctness)
6. Documentation updates (enables user adoption)