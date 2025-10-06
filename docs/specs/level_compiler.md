# Level Compiler Specification

## Overview
The Level Compiler module provides JSON-based level compilation for the Madrona Escape Room simulation. It converts human-readable JSON level definitions with ASCII maps and explicit tilesets into optimized CompiledLevel binary format for efficient GPU processing. The compiler supports both single-level and multi-level formats with comprehensive validation, automatic boundary wall generation, and flexible tileset definitions.

## Key Files

### Source Code
Primary implementation files:

- `madrona_escape_room/level_compiler.py` - Main compiler implementation with JSON parsing, validation, and binary generation
- `madrona_escape_room/dataclass_utils.py` - Utility functions for creating properly initialized CompiledLevel dataclass instances
- `madrona_escape_room/generated_dataclasses.py` - Auto-generated Python dataclasses from C++ structs (CompiledLevel)

### Test Files

#### Python Tests
- `tests/python/test_ascii_level_compiler.py` - Core compilation functionality, validation, and error handling
- `tests/python/test_level_compiler_c_api.py` - C API integration testing and binary format compatibility

## Architecture

### System Integration
The level compiler serves as a critical bridge between human-authored content and the high-performance simulation engine. It integrates with:

- **Asset System**: Resolves asset names to object IDs using C API functions (`get_physics_asset_object_id`, `get_render_asset_object_id`)
- **C API Layer**: Produces CompiledLevel structures compatible with the ctypes-based C API
- **Level Generation System**: Works alongside `level_gen.cpp` which instantiates compiled levels into ECS entities
- **Training Pipeline**: Enables rapid level iteration for curriculum learning and procedural generation

### GPU/CPU Code Separation
- **GPU (NVRTC) Code**: CompiledLevel data consumed by `level_gen.cpp` for entity instantiation on GPU
- **CPU-Only Code**: All compilation logic in `level_compiler.py` runs on CPU during development/training setup
- **Shared Headers**: CompiledLevel structure definition shared between Python (via dataclasses) and C++ (`types.hpp`)

## Data Structures

### Primary Structure

#### CompiledLevel
```cpp
struct CompiledLevel {
    // Level metadata
    int32_t num_tiles;                    // Number of non-empty tiles in level
    int32_t max_entities;                 // BVH allocation limit (tiles + agents + buffer)
    int32_t width;                        // Level width in grid units
    int32_t height;                       // Level height in grid units
    float world_scale;                    // World units per tile (typically 2.5)
    bool done_on_collide;                 // Global collision termination flag
    char level_name[64];                  // Human-readable level identifier

    // World boundaries (calculated from grid dimensions and scale)
    float world_min_x, world_max_x;      // X-axis world boundaries
    float world_min_y, world_max_y;      // Y-axis world boundaries
    float world_min_z, world_max_z;      // Z-axis world boundaries

    // Spawn configuration
    int32_t num_spawns;                   // Number of spawn points in level
    bool spawn_random;                    // Use random spawn selection vs sequential
    bool auto_boundary_walls;             // Generate perimeter walls automatically
    float spawn_x[MAX_SPAWNS];           // Spawn X coordinates in world space
    float spawn_y[MAX_SPAWNS];           // Spawn Y coordinates in world space
    float spawn_facing[MAX_SPAWNS];      // Initial agent facing angles (radians)

    // Tile arrays (sized to MAX_TILES for C API compatibility)
    int32_t object_ids[MAX_TILES];       // Asset object IDs for each tile
    float tile_x[MAX_TILES];             // Tile X positions in world space
    float tile_y[MAX_TILES];             // Tile Y positions in world space
    float tile_z[MAX_TILES];             // Tile Z positions (typically 0.0)
    float tile_scale_x[MAX_TILES];       // X-axis scaling factors
    float tile_scale_y[MAX_TILES];       // Y-axis scaling factors
    float tile_scale_z[MAX_TILES];       // Z-axis scaling factors
    bool tile_persistent[MAX_TILES];     // Persist across episode resets
    bool tile_render_only[MAX_TILES];    // Visual-only (no physics)
    bool tile_done_on_collide[MAX_TILES]; // Individual tile collision termination
    int32_t tile_entity_type[MAX_TILES]; // EntityType enum values
    int32_t tile_response_type[MAX_TILES]; // ResponseType enum values
    float tile_rotation[MAX_TILES][4];   // Quaternion rotations (w,x,y,z)

    // Randomization parameters for procedural variation
    float tile_rand_x[MAX_TILES];        // X-position randomization range
    float tile_rand_y[MAX_TILES];        // Y-position randomization range
    float tile_rand_z[MAX_TILES];        // Z-position randomization range
    float tile_rand_rot_z[MAX_TILES];    // Z-rotation randomization range
    float tile_rand_scale_x[MAX_TILES];  // X-scale randomization range
    float tile_rand_scale_y[MAX_TILES];  // Y-scale randomization range
    float tile_rand_scale_z[MAX_TILES];  // Z-scale randomization range

    // Target configuration for chase rabbit entities
    int32_t num_targets;                 // Number of targets in level
    float target_x[MAX_TARGETS];         // Target X positions in world space
    float target_y[MAX_TARGETS];         // Target Y positions in world space
    float target_z[MAX_TARGETS];         // Target Z positions in world space
    int32_t target_motion_type[MAX_TARGETS]; // Motion equation type (0=static, 1=harmonic)
    float target_params[MAX_TARGETS][8]; // Generic parameter array for motion equations
    // Parameter interpretation based on motion_type:
    // Static (0): No params used
    // Harmonic (1): [0]=omega_x, [1]=omega_y, [2]=center_x, [3]=center_y, [4]=center_z, [5]=mass
};
```

**Key Points:**
- Fixed-size arrays enable efficient C API interop and GPU memory layout
- World coordinates calculated from grid coordinates with configurable scale factor
- Supports up to 1024 tiles, 8 spawn points, and 8 target entities per level (see `generated_constants.py`)
- Target entities use configurable motion equations with generic parameter arrays for future extensibility

### Supporting Structures

#### JSON Level Format (Single Level)
```json
{
    "ascii": ["########", "#S....#", "########"],
    "tileset": {
        "#": {"asset": "wall"},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5,
    "agent_facing": [0.0],
    "spawn_random": false,
    "auto_boundary_walls": false,
    "boundary_wall_offset": 0.5,
    "targets": [
        {
            "position": [5.0, 10.0, 1.0],
            "motion_type": "static"
        },
        {
            "position": [10.0, 5.0, 1.0],
            "motion_type": "harmonic",
            "params": {
                "omega_x": 1.0,
                "omega_y": 0.5,
                "center": [10.0, 5.0, 1.0],
                "mass": 1.0
            }
        }
    ],
    "name": "level_name"
}
```

**Purpose:** Human-readable level definition with ASCII visual layout and explicit asset mappings

**Note:** Sensor configuration (lidar noise, etc.) is now configured separately via the `lidar_config` parameter when creating the SimManager, not embedded in level files.

#### JSON Multi-Level Format

The multi-level format supports three different tileset configurations:

**Option 1 - Shared Tileset (Original Format):**
```json
{
    "levels": [
        {
            "ascii": ["###", "#S#", "###"],
            "name": "level_1",
            "agent_facing": [0.0]
        }
    ],
    "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
    "scale": 2.5,
    "spawn_random": false,
    "name": "curriculum_set"
}
```

**Option 2 - Per-Level Tilesets:**
```json
{
    "levels": [
        {
            "ascii": ["####", "#S.#", "####"],
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "name": "simple_level"
        },
        {
            "ascii": ["######", "#S..C#", "######"],
            "tileset": {
                "#": {"asset": "wall", "done_on_collision": true},
                "C": {"asset": "cube"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "name": "complex_level"
        }
    ],
    "scale": 2.5,
    "name": "mixed_curriculum"
}
```

**Option 3 - Mixed (Global with Per-Level Overrides):**
```json
{
    "levels": [
        {
            "ascii": ["####", "#S.#", "####"],
            "name": "uses_global_tileset"
        },
        {
            "ascii": ["######", "#S..C#", "######"],
            "tileset": {
                "#": {"asset": "wall", "done_on_collision": true},
                "C": {"asset": "cube"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"}
            },
            "name": "custom_tileset_level"
        }
    ],
    "tileset": {
        "#": {"asset": "wall"},
        "S": {"asset": "spawn"},
        ".": {"asset": "empty"}
    },
    "scale": 2.5,
    "name": "mixed_curriculum"
}
```

**Purpose:** Curriculum learning support with flexible tileset configuration - shared, per-level, or mixed approaches

#### Invariants
- All levels must contain at least one spawn point ('S' character)
- Level dimensions must be between 3x3 and 64x64 tiles
- Total tile count cannot exceed MAX_TILES (1024)
- Each level must have access to a tileset (either per-level or global)
- Per-level tilesets take precedence over global tileset when both are present
- Asset names in tileset must resolve to valid object IDs via C API
- Spawn coordinates are automatically converted from grid space to world space

## Module Interface

### LevelCompiler

#### compile_level

**Purpose:** Main entry point for compiling JSON level definitions to CompiledLevel format

**Parameters:**
- `json_data`: Union[str, Dict] - JSON string or dictionary with level definition

**Returns:** List[CompiledLevel] - Always returns list for API consistency (single levels wrapped in list)

**Preconditions:**
- JSON must contain required fields: 'ascii', 'tileset'
- All asset names in tileset must be resolvable via C API
- Level dimensions must be within valid ranges

**Specs:**
- Automatically detects single vs multi-level format based on presence of 'levels' field
- Performs comprehensive validation of JSON structure and constraints
- Converts ASCII grid coordinates to world coordinates using scale factor
- Resolves asset names to object IDs using physics and render asset registries
- Generates boundary walls and corners when auto_boundary_walls=true with configurable offset
- Calculates world boundaries and max_entities for BVH allocation

**Error Handling:**
- **Invalid JSON:** Raises ValueError with parse error details
- **Missing Required Fields:** Raises ValueError specifying missing field
- **Unknown Assets:** Raises ValueError with asset name and character context
- **Dimension Violations:** Raises ValueError with actual vs allowed ranges

#### compile_ascii_level

**Purpose:** Convenience wrapper for ASCII levels using default tileset

**Parameters:**
- `ascii_str`: str - Multi-line ASCII level string
- `scale`: float - World units per tile (default 2.5)
- `agent_facing`: Optional[List[float]] - Agent facing angles in radians
- `level_name`: str - Level identifier (default "unknown_level")

**Returns:** CompiledLevel - Single compiled level structure

**Preconditions:**
- ASCII string must contain valid characters from default tileset
- Must contain at least one spawn point ('S')
- Resulting dimensions must be within valid ranges

**Specs:**
- Uses predefined tileset: '#'=wall, 'C'=cube, 'O'=cylinder, 'S'=spawn, '.'=empty
- Automatically builds JSON structure and delegates to compile_level()
- Extracts single level from returned list for convenience
- Sets default collision behavior: walls non-terminal, cubes/cylinders terminal

**Error Handling:**
- **Unknown Characters:** Raises ValueError with character and grid position
- **No Spawn Points:** Raises ValueError indicating missing spawn requirement
- **Invalid Dimensions:** Raises ValueError with dimension constraints

#### compile_multi_level

**Purpose:** Specialized compiler for multi-level curriculum learning sets

**Parameters:**
- `json_data`: Union[str, Dict] - JSON with multi-level format requiring 'levels' field

**Returns:** List[CompiledLevel] - One CompiledLevel per level in curriculum

**Preconditions:**
- JSON must contain 'levels' array and shared 'tileset'
- Each level must have valid 'ascii' field
- All levels must use compatible tileset asset names

**Specs:**
- Validates multi-level JSON structure with shared and per-level fields
- Compiles each level independently using shared configuration
- Generates level names automatically if not provided per-level
- Preserves level ordering for curriculum progression
- Applies shared settings (scale, spawn_random) to all levels unless overridden

**Error Handling:**
- **Missing 'levels' Field:** Raises ValueError indicating multi-level format requirement
- **Empty Levels Array:** Raises ValueError requiring non-empty level list
- **Per-Level Validation:** Raises ValueError with level index and specific error

### ValidationSubsystem

#### validate_compiled_level

**Purpose:** Final validation of CompiledLevel before C API consumption

**Parameters:**
- `compiled`: CompiledLevel - Compiled level structure from compilation process

**Returns:** None - Raises exception if validation fails

**Preconditions:**
- CompiledLevel must be output of compile_level() or related functions
- All array fields must be properly sized to MAX_TILES

**Specs:**
- Validates numeric ranges for tiles, entities, dimensions, and spawns
- Verifies array lengths match C API expectations (MAX_TILES)
- Checks positive values for scale, max_entities, and spawn counts
- Ensures non-zero dimensions and entity limits

**Error Handling:**
- **Range Violations:** Raises ValueError with actual vs expected ranges
- **Array Size Mismatches:** Raises ValueError with array name and size requirements
- **Invalid Configurations:** Raises ValueError with specific constraint violation

### UtilityFunctions

#### print_level_info

**Purpose:** Debug output for compiled level inspection and verification

**Parameters:**
- `compiled`: CompiledLevel - Compiled level to display information about

**Returns:** None - Prints formatted information to stdout

**Preconditions:**
- CompiledLevel must have valid metadata fields
- level_name field must be valid UTF-8 or safely decodable

**Specs:**
- Displays human-readable summary of level metadata
- Shows dimensions, scale, tile count, and entity limits
- Lists all spawn points with world coordinates and facing angles
- Converts facing angles from radians to degrees for readability

**Error Handling:**
- **Decode Errors:** Handles invalid UTF-8 in level_name gracefully using error='ignore'
- **Array Access:** Safely accesses spawn arrays within num_spawns bounds

### BoundaryWallGeneration

#### _add_boundary_walls

**Purpose:** Generate perimeter walls and corner blocks around level boundaries with configurable offset for collision detection

**Parameters:**
- `level`: CompiledLevel - Level structure to add boundary walls to
- `start_index`: int - Starting tile index for boundary wall placement
- `width`: int - Level width in grid tiles
- `height`: int - Level height in grid tiles
- `scale`: float - World units per tile conversion factor
- `boundary_wall_offset`: float - Additional offset distance beyond calculated world boundaries (default 0.0)

**Returns:** int - Number of boundary tiles added (4 walls + 4 corner blocks = 8)

**Preconditions:**
- Level must have valid world boundary calculations completed
- Must have sufficient space in tile arrays (start_index + 8 ≤ MAX_TILES)
- boundary_wall_offset must be non-negative

**Specs:**
- Creates 4 perimeter walls (north, south, east, west) spanning full level dimensions
- Adds 4 corner blocks at wall intersections to prevent gap collisions
- Positions wall inner edges at world_boundaries + boundary_wall_offset
- Allows agents to travel slightly outside level boundaries before hitting walls
- Enables proper collision detection for out-of-bounds termination conditions
- Uses standard wall thickness (1.0 units) and height (2.0 units)
- Sets walls as persistent, non-render-only, with Static response type
- Corner blocks use same dimensions as wall thickness for seamless coverage
- CompiledLevel world boundaries (world_min_x, world_max_x, etc.) remain unchanged; only physical wall positions are offset

**Error Handling:**
- **Insufficient Space:** Raises ValueError if not enough tile array space remains
- **Invalid Offset:** Raises ValueError for negative boundary_wall_offset values
- **Asset Resolution:** Raises ValueError if wall asset cannot be found

## Configuration

### Build Configuration
The level compiler integrates with the build system through:
- **Generated Constants:** `generated_constants.py` provides MAX_TILES, MAX_SPAWNS, and other limits
- **Generated Dataclasses:** `generated_dataclasses.py` provides CompiledLevel structure definition
- **C API Bindings:** `ctypes_bindings.py` provides asset resolution functions

### Runtime Configuration
- **Default Scale:** 2.5 world units per tile balances movement granularity and performance
- **Default Tileset:** Supports common ASCII characters for rapid prototyping
- **Boundary Walls:** Optional automatic perimeter generation for closed environments
- **Boundary Wall Offset:** Configurable offset (default 0.0) to position walls beyond level boundaries for proper collision detection
- **Spawn Behavior:** Configurable random vs sequential spawn selection
- **Randomization:** Per-tile position and rotation randomization for procedural variation

### Sensor Configuration

Sensor configuration (lidar beam count, FOV, noise parameters, etc.) is now configured separately via the `LidarConfig` parameter when creating a SimManager, not embedded in level files. This allows the same level geometry to be used with different sensor configurations for experimentation and curriculum learning.

**Related Documentation:**
- **Sensor Configuration**: `madrona_escape_room/sensor_config.py` - LidarConfig class for sensor parameters
- **Simulation Spec**: `docs/specs/sim.md` - lidarSystem section documents sensor implementation
- **Testing**: `tests/python/test_configurable_lidar.py` and `test_lidar_noise.py` - Sensor configuration tests