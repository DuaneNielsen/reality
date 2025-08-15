# JSON Level Format with Agent Facing Parameter

## Summary
Added JSON format support to the level compiler allowing levels to have associated parameters. The first parameter implemented is `agent_facing` which sets the initial rotation angle for agents when they spawn.

## Changes Made

### 1. Python Level Compiler (`madrona_escape_room/level_compiler.py`)
- Added `compile_level_from_json()` function to parse JSON level format
- Updated `compile_level()` to accept optional `agent_facing` parameter
- Added `spawn_facing` field to compiled level output
- Updated binary save/load functions to include spawn_facing data

### 2. C++ Structures
- **`src/types.hpp`**: Added `spawn_facing[MAX_SPAWNS]` field to `CompiledLevel` struct
- **`include/madrona_escape_room_c_api.h`**: Added `spawn_facing[8]` field to `MER_CompiledLevel` struct
- **`src/madrona_escape_room_c_api.cpp`**: Updated to copy spawn_facing data

### 3. Level Generation (`src/level_gen.cpp`)
- Modified `resetPersistentEntities()` to use spawn_facing angles from compiled level
- Agents now spawn with specified rotation instead of always facing forward (0.0)

### 4. Python Bindings (`madrona_escape_room/ctypes_bindings.py`)
- Updated `MER_CompiledLevel` ctypes structure to include spawn_facing field
- Modified `dict_to_compiled_level()` to handle spawn_facing data

### 5. Tests
- Added `TestJSONLevelFormat` class with comprehensive tests
- Tests verify JSON parsing, agent facing angles, and SimManager integration

## JSON Level Format

```json
{
    "ascii": "Multi-line ASCII level string",
    "scale": 2.0,                        // Optional, default 2.0
    "agent_facing": [0.0, 1.57, 3.14]   // Optional, radians for each agent
}
```

### Parameters:
- **ascii**: Required. The ASCII art level layout using standard characters (#, S, C, .)
- **scale**: Optional. World units per ASCII character (default: 2.0)
- **agent_facing**: Optional. List of initial facing angles in radians for each spawn point
  - 0.0 = North (forward)
  - π/2 = East (right)
  - π = South (backward)
  - 3π/2 = West (left)

## Usage Examples

### Python API
```python
from madrona_escape_room.level_compiler import compile_level_from_json
import math

json_level = {
    "ascii": """######
#S..S#
######""",
    "scale": 1.5,
    "agent_facing": [0.0, math.pi/2]  # First faces north, second faces east
}

compiled = compile_level_from_json(json_level)
```

### Command Line
```bash
# Compile JSON file to binary
echo '{"ascii": "###\\n#S#\\n###", "agent_facing": [1.57]}' > level.json
python -m madrona_escape_room.level_compiler level.json level.lvl
```

## Future Parameters
The JSON format is extensible. Future parameters could include:
- Agent movement speed
- Reward values
- Episode length
- Object properties
- Room-specific physics settings
- Lighting/visual parameters

## Backward Compatibility
- Existing ASCII levels continue to work unchanged
- Default facing angle is 0.0 (forward) when not specified
- SimManager still accepts `level_ascii` parameter for simple ASCII levels