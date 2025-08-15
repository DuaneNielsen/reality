# Madrona Escape Room Levels

This directory contains level definitions in various formats for the Madrona Escape Room environment.

## File Formats

- **`.txt`** - Plain ASCII text levels (legacy format, agents always face forward)
- **`.json`** - JSON format with parameters (supports agent facing angles and other settings)
- **`.lvl`** - Compiled binary format for use with the simulation

## Available Levels

### Basic Rooms

#### `default` (8x5)
- Simple rectangular room for basic testing
- Single spawn point at top-left
- Versions: `.txt`, `.json`, `.lvl`

#### `room` (10x7)
- Slightly larger rectangular room
- Single spawn point at top-left
- Versions: `.txt`, `.json`, `.lvl`

#### `east_corridor` (23x4)
- Long horizontal corridor
- Spawn point at the east end
- JSON version: Agent faces west (looking back down corridor)
- Versions: `.txt`, `.json`, `.lvl`

### Multi-Agent Levels

#### `two_agents_facing` (10x5)
- Two spawn points on opposite sides
- Agents face each other (0° and 180°)
- Format: `.json`, `.lvl`

#### `four_corners` (10x6)
- Four spawn points in corners
- Each agent faces toward the center (diagonal angles)
- Format: `.json`, `.lvl`

#### `patrol_route` (16x7)
- Four spawn points arranged for patrol pattern
- Agents face in clockwise directions (E, S, N, W)
- Scale: 1.5 units per tile (smaller than default)
- Format: `.json`, `.lvl`

### Obstacle Courses

#### `maze_with_cubes` (12x7)
- Contains walls forming a maze pattern
- Two cube obstacles (C) that can be pushed
- Agent starts facing east toward the maze
- Format: `.json`, `.lvl`

## Agent Facing Angles

In JSON format, agent facing is specified in radians:

| Direction | Radians | Degrees | Description |
|-----------|---------|---------|-------------|
| North | 0.0 | 0° | Forward (default) |
| East | π/2 ≈ 1.5708 | 90° | Right |
| South | π ≈ 3.14159 | 180° | Backward |
| West | 3π/2 ≈ 4.71239 | 270° | Left |
| Northeast | π/4 ≈ 0.785 | 45° | Diagonal |
| Southeast | 3π/4 ≈ 2.356 | 135° | Diagonal |
| Southwest | 5π/4 ≈ 3.927 | 225° | Diagonal |
| Northwest | 7π/4 ≈ 5.498 | 315° | Diagonal |

## Level Format Examples

### ASCII Text Format (.txt)
```
########
#S.....#
#..CC..#
#......#
########
```

### JSON Format (.json)
```json
{
    "ascii": "########\n#S.....#\n#..CC..#\n#......#\n########",
    "scale": 2.0,
    "agent_facing": [1.5708],
    "_comment": "Description of the level"
}
```

## Character Legend

- `#` - Wall
- `.` or ` ` - Empty space
- `S` - Spawn point
- `C` - Cube (pushable obstacle)

## Compiling Levels

To compile a level to binary format:

```bash
# From ASCII text
python -m madrona_escape_room.level_compiler level.txt level.lvl

# From JSON
python -m madrona_escape_room.level_compiler level.json level.lvl

# With custom scale (text files only)
python -m madrona_escape_room.level_compiler level.txt level.lvl --scale 1.5
```

## Using Levels in Code

### Python
```python
from madrona_escape_room import SimManager, madrona

# Using ASCII directly
mgr = SimManager(
    exec_mode=madrona.ExecMode.CPU,
    num_worlds=1,
    level_ascii=open("levels/room.txt").read()
)

# Using JSON with parameters
from madrona_escape_room.level_compiler import compile_level_from_json
import json

with open("levels/maze_with_cubes.json") as f:
    json_level = json.load(f)
    
compiled = compile_level_from_json(json_level)
# Note: SimManager currently uses ASCII internally
# Full compiled level support coming soon
```

### C++ (Viewer/Headless)
```bash
# Load level from binary file
./build/viewer --level levels/maze_with_cubes.lvl

# Multiple worlds with same level
./build/headless -n 100 -s 1000 --level levels/patrol_route.lvl
```