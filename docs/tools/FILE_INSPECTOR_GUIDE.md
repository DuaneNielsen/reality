# File Inspector Guide

The File Inspector is a unified command-line tool for examining and validating Madrona Escape Room files. It supports both recording files (`.rec`) and compiled level files (`.lvl`), providing detailed metadata analysis and integrity validation.

## Overview

The file inspector automatically detects file types and provides comprehensive validation and information display for:

- **Recording files (`.rec`)**: Session recordings with metadata, embedded levels, and action data
- **Level files (`.lvl`)**: Compiled level geometry and spawn data

## Building

The file inspector is built automatically with the main project:

```bash
cmake -B build
make -C build file_inspector -j8
```

The executable will be located at `./build/file_inspector`.

## Usage

### Basic Usage

```bash
./build/file_inspector <file.(rec|lvl)>
```

### Examples

```bash
# Inspect a level file
./build/file_inspector levels/default.lvl

# Inspect a recording file  
./build/file_inspector recordings/demo.rec

# Get help
./build/file_inspector
```

## Output Format

### Recording Files (.rec)

For recording files, the inspector displays:

#### Validation Status
- ✅ Magic number verification (MESR)
- ✅ Version compatibility (supports v1 and v2)
- ✅ File structure integrity
- ✅ Metadata field validation

#### Recording Metadata
- **Simulation name**: Usually "madrona_escape_room"
- **Level name**: Name of the level being played (v2+ format)
- **Creation timestamp**: When recording was created
- **World/Agent counts**: Number of parallel worlds and agents
- **Steps recorded**: Total simulation steps captured
- **Random seed**: Seed used for deterministic replay

#### Embedded Level Information
- **Level name**: Name from the embedded level data
- **Dimensions**: Grid size and world scale factor
- **Entity counts**: Tiles, spawns, max entities
- **File size verification**: Confirms expected structure

**Example output:**
```
Recording File: demo.rec
✓ Valid magic number (MESR)
✓ Valid version (2)
✓ File structure intact
✓ Metadata fields within valid ranges

Recording Metadata:
  Simulation: madrona_escape_room
  Level: custom_maze
  Created: 2023-08-16 14:30:22 UTC
  Worlds: 4, Agents per world: 1
  Steps recorded: 200, Actions per step: 3
  Random seed: 42

Embedded Level:
  Name: maze_level_01
  Dimensions: 16x12 grid, Scale: 2.5
  Tiles: 89, Spawns: 2
  File size: 15,432 bytes (matches expected)

✓ File validation completed successfully
```

### Level Files (.lvl)

For level files, the inspector displays:

#### Validation Status
- ✅ File size verification (matches CompiledLevel struct)
- ✅ Data range validation
- ✅ Spawn point validation

#### Level Details
- **Level name**: Identifier for the level
- **Dimensions**: Grid width/height and scale factor
- **Entity information**: Tile count and maximum entity capacity
- **Spawn points**: Position and facing angle for each spawn
- **Tile data summary**: Count of valid tiles

**Example output:**
```
Level File: maze.lvl
✓ Valid file size (12,472 bytes)
✓ Level data within valid ranges
✓ Spawn data validated

Level Details:
  Name: advanced_maze
  Dimensions: 24x18 grid, Scale: 2.0
  Tiles: 156, Max entities: 200
  Spawn 0: (12.0, 8.0) facing 90°
  Spawn 1: (-12.0, -8.0) facing 270°
  Tile data: 156 valid tiles in bounds

✓ File validation completed successfully
```

## Level Name Support

Starting with format version 2, both recording and level files support level names:

### In JSON Level Files
```json
{
    "name": "my_custom_level",
    "ascii": "########\n#S.....#\n########",
    "scale": 2.5
}
```

### In Compiled Files
- Level files store the name in the `CompiledLevel.level_name` field
- Recording files store it in both `ReplayMetadata.level_name` and the embedded level
- Names are limited to 63 characters (null-terminated 64-byte field)

## Error Handling

The file inspector provides clear error messages for common issues:

### File Access Errors
```
Error: Cannot access file or file is empty: missing.rec
```

### Unsupported File Types
```
Error: Unsupported file type '.xyz'
Supported extensions: .rec (recording files), .lvl (level files)
```

### Corrupted Files
```
Level File: corrupted.lvl
✗ Invalid file size: 1,024 bytes (expected 12,472)
✗ File validation failed
```

### Invalid Recording Data
```
Recording File: bad.rec
✗ Invalid magic number: 0x12345678
✗ File validation failed
```

## Integration with Development Workflow

### Testing Level Compilation
Use the inspector to verify level compilation:

```bash
# Compile a JSON level
uv run python -m madrona_escape_room.level_compiler level.json level.lvl

# Inspect the result
./build/file_inspector level.lvl
```

### Debugging Recordings
Use the inspector to analyze recording sessions:

```bash
# Create a recording
./build/headless --load level.lvl --record session.rec --num-worlds 2 --num-steps 100

# Analyze the recording
./build/file_inspector session.rec
```

### Validation in Scripts
The inspector returns appropriate exit codes for automation:

```bash
#!/bin/bash
if ./build/file_inspector my_level.lvl > /dev/null; then
    echo "Level file is valid"
else
    echo "Level file has issues"
    exit 1
fi
```

## File Format Specifications

### Recording File Structure (.rec)
```
[ReplayMetadata]     - File header with simulation info
[CompiledLevel]      - Embedded level data
[ActionData...]      - Recorded action sequences
```

### Level File Structure (.lvl)
```
[CompiledLevel]      - Complete level definition
```

### Backward Compatibility
- The inspector supports both v1 and v2 recording formats
- V1 recordings lack the `level_name` field in metadata
- V2 recordings include level names in both metadata and embedded level data

## Common Use Cases

### 1. Level Development
```bash
# Design level in JSON
vim my_level.json

# Compile to binary
uv run python -m madrona_escape_room.level_compiler my_level.json my_level.lvl

# Verify compilation
./build/file_inspector my_level.lvl
```

### 2. Recording Analysis
```bash
# Record a session
./build/headless --load level.lvl --record test.rec --num-worlds 1 --num-steps 50

# Analyze the recording
./build/file_inspector test.rec

# Check embedded level matches original
./build/file_inspector level.lvl
```

### 3. File Validation
```bash
# Batch validate all levels
for level in levels/*.lvl; do
    echo "Checking $level..."
    ./build/file_inspector "$level" || echo "FAILED: $level"
done

# Validate recordings
find . -name "*.rec" -exec ./build/file_inspector {} \;
```

### 4. Debugging File Issues
```bash
# Check file sizes
ls -la suspicious.rec
./build/file_inspector suspicious.rec

# Compare working vs broken files
./build/file_inspector working.lvl > working.txt
./build/file_inspector broken.lvl > broken.txt
diff working.txt broken.txt
```

## Exit Codes

- **0**: File validation successful
- **1**: File validation failed or tool error

This makes the inspector suitable for use in scripts and automated testing workflows.

## Implementation Notes

The file inspector is implemented in `file_inspector.cpp` and includes:

- Automatic file type detection by extension
- Comprehensive validation for both file formats
- Support for both v1 and v2 recording formats
- Detailed error reporting with specific failure reasons
- Human-readable output formatting
- Proper exit code handling for automation

The tool leverages existing project infrastructure including the Manager class for reading recording metadata and direct binary file I/O for level files.

## Testing

Unit tests are available in `tests/cpp/unit/test_file_inspector.cpp` covering:

- JSON level compilation and inspection
- Text level compilation and inspection
- Recording file inspection
- Error handling scenarios
- Command-line argument validation
- Level name consistency verification

Run tests with:
```bash
./build/mad_escape_tests --gtest_filter="FileInspectorTest.*"
```