# Level Format Documentation

This document describes the level format used in the Madrona Escape Room for defining environments using ASCII art and JSON configuration.

## Overview

The level system supports three input formats:
1. **Plain ASCII strings** - Simple visual level definition with legacy character map
2. **JSON format** - ASCII with additional parameters (scale, agent facing angles, level name)
3. **JSON with tileset** - Custom asset mapping for any character

All formats compile to a binary `.lvl` format optimized for GPU processing.

## Character Mapping

### Default Tileset
| Character | Asset | Description |
|-----------|-------|-------------|
| `.` or ` ` | empty | Empty space (passable) |
| `#` | wall | Wall (solid collision) |
| `C` | cube | Cube object (movable/interactive) |
| `O` | cylinder | Cylinder object |
| `S` | spawn | Agent spawn point |

### Legacy Character Map
When using plain ASCII format, the compiler uses actual asset IDs from the C API:
- Characters map directly to physics asset object IDs
- Wall and cube IDs are fetched dynamically from `get_physics_asset_object_id()`

### Custom Tilesets
JSON format allows defining custom character-to-asset mappings for any available physics or render asset.

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
- **Total tiles**: Limited by `CompiledLevel::MAX_TILES` (1024)
- **Max grid**: 32×32 (1024 tiles)
- **Spawn points**: Maximum 8 (CompiledLevel::MAX_SPAWNS)
- **Level name**: Maximum 64 characters

## JSON Formats

### Simple Format (Default Tileset)
```json
{
    "ascii": "Multi-line ASCII level string",
    "name": "level_name",        // Optional, default "unknown_level"
    "scale": 2.5,                // Optional, default 2.5
    "agent_facing": [0.0, 1.57]  // Optional, radians for each agent
}
```

### Tileset Format (Custom Asset Mapping)
```json
{
    "ascii": "###O###\n#S...C#\n#######",
    "tileset": {
        "#": {"asset": "wall"},
        "C": {
            "asset": "cube",
            "rand_x": 1.0,      // Optional: Random X offset range
            "rand_y": 1.0,      // Optional: Random Y offset range
            "rand_z": 0.5,      // Optional: Random Z offset range
            "rand_rot_z": 3.14  // Optional: Random Z rotation range (radians)
        },
        "O": {"asset": "cylinder"},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5,
    "agent_facing": [0.0]
}
```

### Parameters

#### `ascii` (required)
Multi-line string containing the ASCII level layout.

#### `name` (optional, default: "unknown_level")
Level identifier string for debugging and tracking.

#### `scale` (optional, default: 2.5)
World units per ASCII character. Determines the physical size of each grid cell.

#### `agent_facing` (optional, default: all agents face 0.0)
Array of initial facing angles in radians for each agent:
- `0.0` - North (forward)
- `π/2` - East (right) 
- `π` - South (backward)
- `3π/2` - West (left)

#### `tileset` (optional)
Dictionary mapping single characters to asset definitions with optional randomization:
```json
"tileset": {
    "<char>": {
        "asset": "<asset_name>",
        "rand_x": <float>,      // Optional: Random X offset range (0.0 to value)
        "rand_y": <float>,      // Optional: Random Y offset range (0.0 to value)
        "rand_z": <float>,      // Optional: Random Z offset range (0.0 to value)
        "rand_rot_z": <float>   // Optional: Random Z rotation range in radians (0.0 to value)
    }
}
```
- Asset names must match registered physics or render assets in the C API
- Randomization values specify the total range; actual offsets are -range/2 to +range/2
- Randomization is applied per episode reset, creating variety without changing level layout

## Compiled Level Structure

The compiler outputs a dictionary matching the C++ `CompiledLevel` struct (types.hpp):

### Header Fields
- `num_tiles`: Number of entities to create (actual tiles used)
- `max_entities`: BVH size hint for physics system (includes persistent entities)
- `width`, `height`: Grid dimensions
- `scale`: World scale factor (units per ASCII character)
- `level_name`: Level identifier string (max 64 characters)
- `array_size`: Compiler-calculated size (width × height) - for validation

### World Boundaries
Calculated from grid dimensions × scale:
- `world_min_x`, `world_max_x`: X-axis boundaries in world units
- `world_min_y`, `world_max_y`: Y-axis boundaries in world units
- `world_min_z`, `world_max_z`: Z-axis boundaries in world units

### Spawn Arrays (fixed size: MAX_SPAWNS = 8)
- `num_spawns`: Number of spawn points found
- `spawn_x[MAX_SPAWNS]`: Spawn X positions in world units
- `spawn_y[MAX_SPAWNS]`: Spawn Y positions in world units
- `spawn_facing[MAX_SPAWNS]`: Agent facing angles in radians

### Tile Data Arrays (fixed size: MAX_TILES = 1024)
All arrays are packed for GPU efficiency:

#### Position & Identity
- `object_ids[MAX_TILES]`: Asset ID for each tile (from AssetIDs namespace)
- `tile_x[MAX_TILES]`: World X position
- `tile_y[MAX_TILES]`: World Y position
- `tile_z[MAX_TILES]`: World Z position

#### Flags & Types
- `tile_persistent[MAX_TILES]`: Whether tile persists across episodes (bool)
- `tile_render_only[MAX_TILES]`: Whether tile is render-only without physics (bool)
- `tile_entity_type[MAX_TILES]`: EntityType enum values
  - 0 = EntityType::None
  - 1 = EntityType::Cube
  - 2 = EntityType::Wall
  - 3 = EntityType::Agent
- `tile_response_type[MAX_TILES]`: Physics ResponseType values
  - 0 = Dynamic (movable)
  - 1 = Kinematic (scripted motion)
  - 2 = Static (immovable, default)

#### Transform Arrays
- `tile_scale_x/y/z[MAX_TILES]`: Local scale factors per axis
- `tile_rot_w/x/y/z[MAX_TILES]`: Rotation quaternion components (w,x,y,z)

#### Randomization Arrays
Enable procedural variation per episode reset:
- `tile_rand_x/y/z[MAX_TILES]`: Random position offset ranges (-range/2 to +range/2)
- `tile_rand_rot_z[MAX_TILES]`: Random Z-axis rotation range in radians

When non-zero, the simulation applies random offsets within these ranges each episode, creating variety while maintaining the same logical level layout. This is particularly useful for RL training to prevent overfitting to exact positions.

### Special Behaviors
- **Wall Scaling**: Walls are automatically scaled to match tile spacing (scale parameter)
- **Default Values**: Non-specified tiles use identity transforms and static physics

## Usage Examples

### Python API
```python
from madrona_escape_room.level_compiler import compile_level, compile_level_from_json

# Simple ASCII compilation with legacy character map
level = '''
##########
#S.......#
#..####..#
#...CC...#
##########
'''
compiled = compile_level(level, scale=2.5, level_name="maze_01")

# With agent facing
compiled = compile_level(level, agent_facing=[math.pi/2])  # Face right

# JSON compilation with default tileset
json_level = {
    "ascii": level,
    "name": "test_level",
    "scale": 1.5,
    "agent_facing": [0.0, math.pi/2]
}
compiled = compile_level_from_json(json_level)

# Custom tileset for new assets
json_level = {
    "ascii": "###O###\n#S...C#\n#######",
    "tileset": {
        "#": {"asset": "wall"},
        "C": {"asset": "cube"},
        "O": {"asset": "cylinder"},  # Custom asset
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.0
}
compiled = compile_level_from_json(json_level)

# Tileset with randomization for dynamic environments
json_level = {
    "ascii": "###\n#SC#\n###",
    "tileset": {
        "#": {"asset": "wall"},
        "C": {
            "asset": "cube",
            "rand_x": 1.0,       # ±0.5 units X offset per episode
            "rand_y": 1.0,       # ±0.5 units Y offset per episode
            "rand_rot_z": 3.14   # ±90° rotation per episode
        },
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5
}
compiled = compile_level_from_json(json_level)

# Direct tileset usage in compile_level
tileset = {
    "#": {"asset": "wall"},
    "X": {"asset": "custom_obstacle"},  # Custom mapping
    "S": {"asset": "spawn"},
    ".": {"asset": "empty"}
}
compiled = compile_level("###\n#SX#\n###", tileset=tileset)
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

Compiled levels save to `.lvl` files using little-endian binary format via C API:

### File I/O Functions
```python
# Save compiled level to binary file
save_compiled_level_binary(compiled, "level.lvl")

# Load compiled level from binary file
loaded = load_compiled_level_binary("level.lvl")
```

The binary format is handled entirely by the C API through:
- `mer_write_compiled_level()` - Write level to file
- `mer_read_compiled_level()` - Read level from file

### Constants from C API
The compiler dynamically fetches these values to match the C++ CompiledLevel struct:
- `MAX_TILES_C_API` - Maximum tiles per level (1024 - supports up to 32x32 grid)
- `MAX_SPAWNS_C_API` - Maximum spawn points (8)
- `MAX_LEVEL_NAME_LENGTH` - Maximum level name length (64 characters)

These values are retrieved at runtime using:
```python
MAX_TILES_C_API = _get_max_tiles_from_c_api()   # CompiledLevel::MAX_TILES
MAX_SPAWNS_C_API = _get_max_spawns_from_c_api()  # CompiledLevel::MAX_SPAWNS
```

## Entity Allocation

The compiler calculates `max_entities` for BVH allocation:
```
max_entities = entity_count + persistent_entities + buffer
             = level_tiles + 6 + 30
```

### Persistent Entities (6 total)
- 1 floor entity
- 2 agent entities  
- 3 origin marker entities

All persistent entities have ObjectID components and consume BVH slots.

### Entity Type Assignment
Based on asset object IDs (matching EntityType enum in types.hpp):
- Wall assets → EntityType::Wall (2)
- Cube assets → EntityType::Cube (1)
- Agent spawn → EntityType::Agent (3) - set during agent creation
- Other assets → EntityType::None (0)

### Asset ID Resolution
The compiler uses C API functions to resolve asset names:
1. `get_physics_asset_object_id()` - Check physics assets first
2. `get_render_asset_object_id()` - Check render-only assets
3. Special values: "spawn" → -1, "empty" → 0

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

def test_custom_assets():
    # Test with custom tileset
    json_level = {
        "ascii": "###\n#SO#\n###",
        "tileset": {
            "#": {"asset": "wall"},
            "O": {"asset": "cylinder"},
            "S": {"asset": "spawn"},
        },
        "scale": 1.0
    }
    compiled = compile_level_from_json(json_level)
    mgr = SimManager(compiled_level=compiled)
    # Test interaction with cylinder asset

def test_randomized_level():
    # Test with randomized cube positions
    json_level = {
        "ascii": "###\n#SC#\n###",
        "tileset": {
            "#": {"asset": "wall"},
            "C": {
                "asset": "cube",
                "rand_x": 0.8,       # Cube position varies by ±0.4 units
                "rand_y": 0.8,
                "rand_rot_z": 1.57   # Cube rotation varies by ±45°
            },
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"}
        },
        "scale": 2.0
    }
    compiled = compile_level_from_json(json_level)
    mgr = SimManager(compiled_level=compiled)
    # Each episode reset will place cube at slightly different position
```

This allows tests to:
- Define exact environment requirements using visual ASCII art
- Test specific asset configurations with custom tilesets
- Control spawn positions and agent facing angles precisely