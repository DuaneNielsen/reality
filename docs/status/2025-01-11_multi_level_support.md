# Multi-Level Support Implementation

**Date:** January 11, 2025  
**Branch:** train/curriculum  
**Status:** Core Implementation Complete

## Overview

Implemented comprehensive multi-level support for the Madrona Escape Room level compiler, enabling curriculum learning scenarios where multiple levels with shared tilesets can be compiled and loaded simultaneously. Each level maintains its own `CompiledLevel` struct with unique naming.

## Changes Made

### 1. Level Compiler Core (`madrona_escape_room/level_compiler.py`)

**New Functions:**
- `_validate_multi_level_json(data)` - Validates multi-level JSON structure
- `compile_multi_level(json_data)` - Returns `List[CompiledLevel]` from multi-level JSON
- `_compile_single_level(data)` - Internal helper to avoid code duplication

**Modified Functions:**
- `_validate_json_level(data)` - Auto-detects single vs multi-level format
- `compile_level(json_data)` - Returns `Union[CompiledLevel, List[CompiledLevel]]` with auto-detection

**Key Features:**
- Shared tileset across all levels reduces JSON duplication
- Per-level names stored in individual `CompiledLevel.level_name` fields
- Per-level agent_facing angles supported
- Full backwards compatibility maintained
- Comprehensive validation with descriptive error messages

### 2. Progressive Level Generator (`levels/generate_progressive_levels.py`)

**New Features:**
- Added `--single` command-line flag
- `generate_multi_level_json()` function creates single file with all 20 progressive levels
- Shared tileset definition with obstacle randomization parameters
- Prevents conflicting `--single` and `--level` arguments

**Usage:**
```bash
# Generate separate files (existing behavior)
./levels/generate_progressive_levels.py

# Generate single multi-level file
./levels/generate_progressive_levels.py --single
```

### 3. Test Suite (`tests/python/test_ascii_level_compiler.py`)

**New Tests:**
- `test_multi_level_compilation()` - Validates successful multi-level compilation
- `test_multi_level_validation_errors()` - Tests error handling for invalid JSON

**Coverage:**
- Both `compile_level()` and `compile_multi_level()` functions
- Auto-detection functionality  
- Validation of individual level properties
- Error cases (missing fields, empty arrays, etc.)

## Multi-Level JSON Format

```json
{
    "levels": [
        {
            "ascii": ["####", "#S.#", "####"],
            "name": "simple_level",
            "agent_facing": [0.0]  // Optional per-level
        },
        {
            "ascii": ["######", "#S..C#", "######"],
            "name": "cube_level",
            "agent_facing": [1.57]
        }
    ],
    "tileset": {               // Shared across all levels
        "#": {"asset": "wall", "done_on_collision": True},
        "C": {"asset": "cube", "done_on_collision": True},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5,              // Shared scale
    "name": "multi_level_set"   // Optional set name
}
```

## Technical Details

### Compilation Process
1. `compile_level()` detects format by checking for "levels" key
2. Multi-level: delegates to `compile_multi_level()` 
3. Each level compiled individually using shared tileset
4. Returns `List[CompiledLevel]` with unique names per level

### Manager Integration
- `SimManager` already accepts `List[CompiledLevel]` (existing functionality)
- Each world can use a different level from the compiled list
- Level names preserved in `CompiledLevel.level_name` field (64 bytes)

## Files Modified

### Core Implementation
- `madrona_escape_room/level_compiler.py` - Multi-level compilation logic
- `levels/generate_progressive_levels.py` - Single-file generation mode

### Tests
- `tests/python/test_ascii_level_compiler.py` - Comprehensive multi-level tests

### Documentation  
- Updated docstrings and examples in level compiler
- This status report

## Verification

### Tests Passing
- All 19 existing level compiler tests ✅
- 2 new multi-level tests ✅  
- Full backwards compatibility verified ✅

### Manual Testing
- Generated 20-level progressive multi-level file ✅
- Successfully compiled to `List[CompiledLevel]` ✅
- Verified individual level properties ✅

## Next Steps

### 1. SimManager Integration Testing
- [ ] Load compiled multi-level into `SimManager` with multiple worlds
- [ ] Verify each world uses correct level from the list
- [ ] Test level switching/curriculum progression scenarios

### 2. Binary Level File Support  
- [ ] Extend binary .lvl format to support multiple levels
- [ ] Update `save_compiled_level_binary()` for multi-level output
- [ ] Update `load_compiled_level_binary()` for multi-level input
- [ ] Verify viewer can load and display multi-level .lvl files

### 3. Viewer Enhancements
- [ ] Multi-level navigation in viewer interface
- [ ] Level selection/switching during playback
- [ ] Multi-level recording and replay support

### 4. Documentation
- [ ] Update `VIEWER_GUIDE.md` with multi-level instructions
- [ ] Create curriculum learning example in documentation
- [ ] Update level format specification

## Usage Examples

### Generate Multi-Level File
```bash
./levels/generate_progressive_levels.py --single --output-dir levels
# Creates: levels/progressive_levels_1_to_20_multi.json
```

### Compile and Use
```python
from madrona_escape_room.level_compiler import compile_level

# Load multi-level JSON
with open('levels/multi_level.json') as f:
    multi_level = json.load(f)

# Compile (auto-detects format)
compiled_levels = compile_level(multi_level)  # Returns List[CompiledLevel]

# Use with SimManager
mgr = SimManager(
    exec_mode=ExecMode.CPU,
    gpu_id=-1,
    num_worlds=5,
    rand_seed=42,
    auto_reset=True,
    compiled_levels=compiled_levels
)
```

This implementation provides a solid foundation for curriculum learning and multi-level scenarios while maintaining full backwards compatibility with existing single-level workflows.