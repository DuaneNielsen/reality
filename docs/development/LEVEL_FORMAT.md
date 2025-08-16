# Level Format Documentation

This document describes the level format used in the Madrona Escape Room for defining environments using ASCII art and JSON configuration.

## Overview

The level system supports two input formats:
1. **Plain ASCII strings** - Simple visual level definition
2. **JSON format** - ASCII with additional parameters (scale, agent facing angles)

Both formats compile to a binary `.lvl` format optimized for GPU processing.

## ASCII Character Map

| Character | Tile Type | Description |
|-----------|-----------|-------------|
| `.` or ` ` | TILE_EMPTY | Empty space (passable) |
| `#` | TILE_WALL | Wall (solid collision) |
| `C` | TILE_CUBE | Cube object (movable/interactive) |
| `S` | TILE_SPAWN | Agent spawn point |

### Future Tile Types
```
'D': TILE_DOOR     // Future expansion
'B': TILE_BUTTON   // Future expansion  
'G': TILE_GOAL     // Future expansion
```

## ASCII Format

### Basic Structure
```
##########
#S.......#
#..####..#
#........#
##########
```

### Rules
- Each character represents one grid cell
- Grid coordinates convert to world coordinates centered at origin
- Y-axis is inverted (ASCII Y=0 at top, world Y+ up)
- At least one spawn point (`S`) required
- Trailing whitespace automatically trimmed

### Coordinate System
```python
# Grid to world coordinate conversion
world_x = (x - width / 2.0 + 0.5) * scale
world_y = -(y - height / 2.0 + 0.5) * scale
```

### Dimension Limits
- **Width**: 3-64 characters
- **Height**: 3-64 characters  
- **Total tiles**: Limited by `MAX_TILES_C_API` (default 1024)

## JSON Format

### Structure
```json
{
    "ascii": "Multi-line ASCII level string",
    "scale": 2.5,
    "agent_facing": [0.0, 1.57]
}
```

### Parameters

#### `ascii` (required)
Multi-line string containing the ASCII level layout using the character map above.

#### `scale` (optional, default: 2.5)
World units per ASCII character. Determines the physical size of each grid cell.

#### `agent_facing` (optional, default: all agents face 0.0)
Array of initial facing angles in radians for each agent:
- `0.0` - North (forward)
- `π/2` - East (right) 
- `π` - South (backward)
- `3π/2` - West (left)

### Example
```json
{
    "ascii": "#####\n#S.S#\n#####",
    "scale": 1.5,
    "agent_facing": [0.0, 1.57]
}
```

## Compiled Level Structure

The compiler outputs a dictionary matching the C++ `MER_CompiledLevel` struct:

### Core Fields
- `num_tiles`: Number of entities to create
- `max_entities`: BVH size hint (includes persistent entities)
- `width`, `height`: Grid dimensions
- `scale`: World scale factor
- `array_size`: Compiler-calculated size (width × height)

### Tile Arrays (fixed size: `MAX_TILES_C_API`)
- `tile_types[]`: Tile type constants
- `tile_x[]`: World X coordinates 
- `tile_y[]`: World Y coordinates

### Spawn Arrays (fixed size: `MAX_SPAWNS_C_API`) 
- `num_spawns`: Number of spawn points found
- `spawn_x[]`: Spawn X coordinates
- `spawn_y[]`: Spawn Y coordinates  
- `spawn_facing[]`: Agent facing angles in radians

## Usage Examples

### Python API
```python
from madrona_escape_room.level_compiler import compile_level

# Simple ASCII compilation
level = '''
##########
#S.......#
#..####..#
#........#
##########
'''
compiled = compile_level(level, scale=2.5)

# With agent facing
compiled = compile_level(level, agent_facing=[math.pi/2])  # Face right

# JSON compilation
json_level = {
    "ascii": level,
    "scale": 1.5,
    "agent_facing": [0.0, math.pi/2]
}
compiled = compile_level_from_json(json_level)
```

### Command Line
```bash
# Compile ASCII file to binary
python -m madrona_escape_room.level_compiler maze.txt maze.lvl

# With custom scale
python -m madrona_escape_room.level_compiler maze.txt maze.lvl --scale 1.5

# Display level info
python -m madrona_escape_room.level_compiler --info maze.lvl

# Run test suite
python -m madrona_escape_room.level_compiler --test
```

## Binary Format

Compiled levels save to `.lvl` files using little-endian binary format:

### Header
1. `int32_t num_tiles`
2. `int32_t max_entities` 
3. `int32_t width`
4. `int32_t height`
5. `float scale`
6. `int32_t num_spawns`

### Arrays
1. `spawn_x[MAX_SPAWNS_C_API]` (float array)
2. `spawn_y[MAX_SPAWNS_C_API]` (float array) 
3. `spawn_facing[MAX_SPAWNS_C_API]` (float array)
4. `tile_types[MAX_TILES_C_API]` (int32 array)
5. `tile_x[MAX_TILES_C_API]` (float array)
6. `tile_y[MAX_TILES_C_API]` (float array)

## Entity Allocation

The compiler calculates `max_entities` for BVH allocation:
```
max_entities = entity_count + persistent_entities + buffer
             = walls_and_cubes + 6 + 30
```

### Persistent Entities (6 total)
- 1 floor entity
- 2 agent entities  
- 3 origin marker entities

All persistent entities have ObjectID components and consume BVH slots.

## Validation Rules

### Compile-Time Validation
- Non-empty level string
- Valid dimensions (3-64 × 3-64)
- At least one spawn point
- All characters in character map
- Total tiles ≤ `MAX_TILES_C_API`

### Runtime Validation
- Required fields present
- Array sizes match C++ expectations
- Positive scale and entity counts
- Spawn count ≤ `MAX_SPAWNS_C_API`

## Error Handling

Common compilation errors:

```python
# Empty level
ValueError: "Empty level string"

# No spawn points  
ValueError: "No spawn points (S) found in level - at least one required"

# Unknown character
ValueError: "Unknown character 'X' at grid position (3, 1)"

# Too large
ValueError: "Level too large: 65×65 = 4225 tiles > 1024 max (from C API)"

# Invalid dimensions
ValueError: "Level width 2 must be between 3 and 64"
```

## Integration with Tests

The level compiler enables test-driven development:

```python
def test_movement():
    level = '''
    #####
    #S..#
    #...#
    #####
    '''
    compiled = compile_level(level)
    mgr = SimManager(compiled_level=compiled)
    # Test agent movement in this specific layout
```

This allows tests to define their exact environment requirements using visual ASCII art.