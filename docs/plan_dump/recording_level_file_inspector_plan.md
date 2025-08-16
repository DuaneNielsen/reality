# Recording and Level File Inspector Tool Plan

## Reading List

Before implementing this plan, developers should read the following files to understand the current architecture:

### Core Data Structures
- `src/types.hpp` - Study the `CompiledLevel` struct definition, field meanings, and constraints
- `src/replay_metadata.hpp` - Understand `ReplayMetadata` struct and `ReplayLoader` utility functions
- `src/consts.hpp` - Review action space constants and game parameters

### File Format Understanding
- `test_replay.cpp` - Current minimal implementation to be enhanced
- `src/headless.cpp` - See how `.lvl` files are loaded (search for "Load from .lvl file")
- `src/viewer.cpp` - Another example of `.lvl` file loading
- `src/mgr.cpp` - Study `loadReplay()` method for `.rec` file structure

### Level Compilation Process
- `madrona_escape_room/level_compiler.py` - JSON to CompiledLevel conversion logic
- `levels/default.json` - Example JSON level format to understand input structure
- `docs/development/LEVEL_FORMAT.md` - Level format specification

### Related Test Code
- `tests/cpp/unit/test_level_utilities.cpp` - Level validation patterns
- `tests/cpp/unit/test_recording_utilities.cpp` - Recording file validation examples

## Plan: Recording File Metadata Extractor with Level Name Support and .lvl File Support

### Verification: .lvl File Format

Based on the code analysis, I can confirm that **.lvl files contain just the CompiledLevel struct data** and follow the same binary format specification:

- **File Format**: `.lvl` files contain only the serialized `CompiledLevel` struct (12,408 bytes for default.lvl)
- **Usage Pattern**: Both viewer and headless tools read `.lvl` files using `sizeof(CompiledLevel)` 
- **Data Structure**: Same `CompiledLevel` struct used in both `.lvl` and `.rec` files
- **File Structure Comparison**:
  - `.lvl` files: `[CompiledLevel]`
  - `.rec` files: `[ReplayMetadata][CompiledLevel][Actions...]`

### Goal
Convert `test_replay.cpp` to a unified tool that extracts metadata, validates file format compliance, and supports both `.rec` (recording) and `.lvl` (level) files.

### Proposed Changes

#### 1. Add Level Name Field to Data Structures

##### A. Extend JSON Level Format
- Add `"name"` field to JSON level files (e.g., `"name": "default_room"`)
- Update level compiler to read and store this name
- Maintain backward compatibility with levels that don't have a name field

##### B. Extend CompiledLevel Structure
- Add `char level_name[64]` field to `CompiledLevel` struct in `types.hpp`
- Update level compilation process to copy name from JSON
- Set default name if not provided in JSON

##### C. Update ReplayMetadata Structure
- Add `char level_name[64]` field to `ReplayMetadata` in `replay_metadata.hpp`
- Increment `REPLAY_VERSION` to 2 for format change
- Update `createDefault()` method to include level name

#### 2. Dual File Format Support

##### A. File Type Detection
- Auto-detect file type by extension (`.rec` vs `.lvl`)
- Validate file size expectations for each format
- Support both formats with unified interface

##### B. .lvl File Processing
- Read CompiledLevel struct directly from `.lvl` files
- Display level metadata (dimensions, tiles, spawns, scale)
- Validate level data integrity

##### C. .rec File Processing
- Use existing ReplayMetadata reading
- Extract embedded CompiledLevel from recording
- Display both recording and level information

#### 3. Enhanced Metadata Display

##### For .rec Files:
- **File validation**: Check magic number and version
- **Complete metadata**: Display all fields including new level name
- **Recording info**: Steps, worlds, agents, timestamp
- **Embedded level**: Show level data from embedded CompiledLevel

##### For .lvl Files:
- **Level validation**: Check struct completeness and data ranges
- **Level details**: Dimensions, tile count, spawn points, scale
- **Level name**: Display name from CompiledLevel struct
- **File integrity**: Validate spawn data and tile arrays

#### 4. Data Specification Validation

##### For .rec Files:
- **Magic number check**: Verify `REPLAY_MAGIC` (0x4D455352 "MESR")
- **Version validation**: Support both version 1 and 2
- **Structure integrity**: Validate metadata + level + actions layout
- **Size consistency**: File size matches expected content

##### For .lvl Files:
- **Size validation**: File size == `sizeof(CompiledLevel)`
- **Data ranges**: Validate dimensions, spawn counts, coordinate ranges
- **Array bounds**: Check tile and spawn data within limits
- **Scale validation**: Ensure reasonable scale factor

### Implementation Steps

1. **Update data structures**: Add level_name fields to CompiledLevel and ReplayMetadata
2. **Modify level compiler**: Update JSON parsing to include name field
3. **Create unified inspector tool**: 
   - Auto-detect file type by extension
   - Use appropriate parsing logic for each format
   - Unified validation and display interface
4. **Add comprehensive validation**: Format-specific checks for both file types
5. **Maintain compatibility**: Handle both v1 and v2 replay formats

### Files to Modify
- `src/types.hpp`: Add level_name to CompiledLevel
- `src/replay_metadata.hpp`: Add level_name to ReplayMetadata, bump version
- `madrona_escape_room/level_compiler.py`: Parse name from JSON
- `test_replay.cpp`: Rename to `file_inspector.cpp` with dual format support
- Level JSON files: Add name fields where desired

### Expected Output

#### For .rec files:
```
Recording File: demo.rec
✓ Valid magic number (MESR)
✓ Valid version (2)
✓ File structure intact
✓ Metadata fields within valid ranges

Recording Metadata:
  Simulation: madrona_escape_room
  Created: 2025-08-16 14:30:22 UTC
  Worlds: 4, Agents per world: 1
  Steps recorded: 200, Actions per step: 3
  Random seed: 42

Embedded Level:
  Name: default_room
  Dimensions: 8x5 grid, Scale: 2.5
  Tiles: 23, Spawns: 1
  File size: 12,345 bytes (matches expected)
```

#### For .lvl files:
```
Level File: default.lvl
✓ Valid file size (12,408 bytes)
✓ Level data within valid ranges
✓ Spawn data validated

Level Details:
  Name: default_room
  Dimensions: 8x5 grid, Scale: 2.5
  Tiles: 23, Max entities: 50
  Spawns: 1 at (4.0, 2.0) facing 0.0°
  Tile data: 23 valid tiles in bounds
```