# Test-Driven Level System - Feature Proposal

## Executive Summary
Implement a test-driven level definition system that allows tests to define their level layouts inline using ASCII art representations, which are compiled to GPU-compatible buffers at runtime.

## Problem Statement

### Current Issues
1. **Test-Level Coupling**: Tests are tightly coupled to a single hardcoded level design
2. **Test Fragility**: Changing the level for experimentation breaks existing tests
3. **Lack of Clarity**: Tests don't clearly show what environment they're testing against
4. **GPU Constraints**: Level generation runs on GPU, limiting our options for dynamic level loading

### Requirements
- Tests must be able to define their own level layouts
- Level definitions must be clear and visual in test code
- System must work within GPU execution constraints (no file I/O, no dynamic allocation)
- Must maintain test stability while allowing experimentation

## Proposed Solution

### ASCII Art Level Definition
Tests define levels using ASCII art strings directly in the test code:

```python
def test_narrow_corridor():
    level = """
    ##########
    #S.......#
    #.#####..#
    #........#
    ##########
    """
    mgr = SimManager(level_ascii=level)
```

### Character Mapping
- `.` = Empty space
- `#` = Wall/obstacle
- `S` = Agent spawn point
- `C` = Cube obstacle
- `D` = Door (future)
- `G` = Goal (future)
- `B` = Button (future)

## Architecture

### Component Overview
```
┌─────────────────┐
│   Python Test   │
│  (ASCII string) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ LevelCompiler   │
│   (Python)      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ CompiledLevel   │
│ (Binary Buffer) │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  GPU Transfer   │
│  (cudaMemcpy)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ generateLevel() │
│    (GPU Code)   │
└─────────────────┘
```

### Data Flow

1. **Test Definition Phase**
   - Test defines level as multi-line string
   - ASCII art provides visual representation

2. **Compilation Phase** (Python)
   - Parse ASCII string into 2D grid
   - Extract obstacle positions and types
   - Pack into fixed-size buffer structure

3. **Transfer Phase** (Manager)
   - Allocate GPU buffer for level data
   - Copy compiled level from CPU to GPU
   - Store pointer in singleton component

4. **Generation Phase** (GPU)
   - Read from fixed-size buffer
   - Create entities based on buffer data
   - Simple array iteration (GPU-friendly)

## Implementation Plan

### Phase 1: GPU-First Hardcoded Level

#### Next Steps

##### 1.1 Implement Hardcoded 16x16 Room in generateLevel()
```cpp
// level_gen.cpp

// Add helper function to create a wall entity (similar to makeCube)
static Entity makeWall(Engine &ctx,
                      float wall_x,
                      float wall_y,
                      float scale = 1.f)
{
    Entity wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        wall,
        Vector3 {
            wall_x,
            wall_y,
            1.f * scale,  // Half height above ground
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            scale,
            scale,
            2.f * scale,  // Walls are taller than wide
        });
    registerRigidBodyEntity(ctx, wall, SimObject::Wall);

    return wall;
}

// Replace existing generateLevel() with hardcoded room
static void generateLevel(Engine &ctx)
{
    // Clear any existing level state
    LevelState &level = ctx.singleton<LevelState>();
    
    // Hardcoded 16x16 empty room
    static constexpr int32_t ROOM_SIZE = 16;
    static constexpr float TILE_SIZE = 2.0f;  // World units per tile
    
    CountT entity_count = 0;
    
    // Create walls around perimeter
    for (int32_t x = 0; x < ROOM_SIZE; x++) {
        for (int32_t y = 0; y < ROOM_SIZE; y++) {
            // Only create walls on edges
            if (x == 0 || x == ROOM_SIZE-1 || 
                y == 0 || y == ROOM_SIZE-1) {
                
                float world_x = (x - ROOM_SIZE/2.0f) * TILE_SIZE;  // Center room at origin
                float world_y = (y - ROOM_SIZE/2.0f) * TILE_SIZE;
                
                Entity wall = makeWall(ctx, world_x, world_y, TILE_SIZE/2.0f);
                
                // Store in level state for tracking
                if (entity_count < consts::maxEntitiesPerRoom) {
                    level.rooms[0].entities[entity_count++] = wall;
                }
            }
        }
    }
    
    // Fill remaining slots with none
    for (CountT i = entity_count; i < consts::maxEntitiesPerRoom; i++) {
        level.rooms[0].entities[i] = Entity::none();
    }
}

// Update resetPersistentEntities to place agents in center of new room
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

    for (CountT i = 0; i < 4; i++) {
        Entity wall_entity = ctx.data().borders[i];
        registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
    }

    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity agent_entity = ctx.data().agents[i];
        registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

        // Place agents in center of 16x16 room (centered at origin)
        Vector3 pos {
            i * 2.0f - 1.0f,  // Slight offset between agents
            0.0f,              // Center of room
            1.0f,              // Above ground
        };

        ctx.get<Position>(agent_entity) = pos;
        ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
            0.0f,  // Face forward initially
            math::up);

        ctx.get<Progress>(agent_entity).maxY = pos.y;

        // Reset velocities and forces
        ctx.get<Velocity>(agent_entity) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
        ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
        ctx.get<Action>(agent_entity) = Action {
            .moveAmount = 0,
            .moveAngle = 0,
            .rotate = consts::numTurnBuckets / 2,
        };

        ctx.get<StepsRemaining>(agent_entity).t = consts::episodeLen;
    }
}
```

##### 1.2 Test Both CPU and GPU Execution
- Build and run on CPU mode
- Build and run on GPU mode  
- Use screenshot tool (now supports PNG) to verify visual output
- Ensure identical behavior between CPU and GPU

### Phase 2: Core Infrastructure

#### 2.1 GPU Data Structure (C++)
```cpp
// types.hpp
struct CompiledLevel {
    static constexpr int32_t MAX_TILES = 256;  // 16x16 grid max
    
    // Tile data (packed for GPU efficiency)
    int32_t tile_types[MAX_TILES];    // Type enum for each tile
    float tile_x[MAX_TILES];          // World X position
    float tile_y[MAX_TILES];          // World Y position
    int32_t num_tiles;                // Actual tiles used
    
    // Agent spawn data
    float spawn_x[consts::numAgents];
    float spawn_y[consts::numAgents];
    float spawn_rot[consts::numAgents];
    
    // Level metadata
    int32_t width;                    // Grid width
    int32_t height;                   // Grid height
    float scale;                       // World scale factor
};

enum TileType : int32_t {
    TILE_EMPTY = 0,
    TILE_WALL = 1,
    TILE_CUBE = 2,
    TILE_SPAWN = 3,
    TILE_DOOR = 4,    // Future
    TILE_BUTTON = 5,  // Future
    TILE_GOAL = 6,    // Future
};
```

#### 2.2 Python Compiler
```python
# madrona_escape_room/level_compiler.py
class LevelCompiler:
    """Compiles ASCII level strings to GPU buffer format"""
    
    CHAR_MAP = {
        '.': TileType.EMPTY,
        '#': TileType.WALL,
        'C': TileType.CUBE,
        'S': TileType.SPAWN,
        ' ': TileType.EMPTY,  # Whitespace = empty
    }
    
    @staticmethod
    def compile(ascii_str: str, scale: float = 2.0) -> bytes:
        """
        Compile ASCII string to binary buffer.
        
        Args:
            ascii_str: Multi-line string defining level
            scale: World units per ASCII cell
            
        Returns:
            Binary buffer matching CompiledLevel struct
        """
        lines = ascii_str.strip().split('\n')
        height = len(lines)
        width = max(len(line) for line in lines)
        
        # Parse tiles
        tiles = []
        spawns = []
        
        for y, line in enumerate(lines):
            for x, char in enumerate(line):
                if char == 'S':
                    spawns.append((x * scale, y * scale))
                elif char != '.' and char != ' ':
                    tile_type = CHAR_MAP.get(char, TileType.EMPTY)
                    tiles.append({
                        'type': tile_type,
                        'x': x * scale,
                        'y': y * scale
                    })
        
        # Pack into binary format matching C++ struct
        return pack_compiled_level(tiles, spawns, width, height, scale)
```

#### 2.3 Manager Integration
```cpp
// mgr.hpp
struct Config {
    // ... existing fields ...
    bool use_compiled_level = false;
    const void* compiled_level_data = nullptr;
    size_t compiled_level_size = 0;
};

// mgr.cpp
void Manager::Impl::init() {
    // ... existing init ...
    
    if (cfg.use_compiled_level) {
        // Copy level data to GPU-accessible buffer
        CompiledLevel* level_buffer = allocate_gpu_buffer<CompiledLevel>();
        
        if (cfg.execMode == ExecMode::CUDA) {
            cudaMemcpy(level_buffer, cfg.compiled_level_data, 
                      sizeof(CompiledLevel), cudaMemcpyHostToDevice);
        } else {
            memcpy(level_buffer, cfg.compiled_level_data, sizeof(CompiledLevel));
        }
        
        // Store in singleton for GPU access
        set_singleton<CompiledLevel>(level_buffer);
    }
}
```

#### 2.4 GPU Level Generation
```cpp
// level_gen.cpp
void generateLevel(Engine &ctx) {
    // Check if we have a compiled level
    CompiledLevel* compiled = ctx.singleton<CompiledLevel>();
    
    if (compiled && compiled->num_tiles > 0) {
        // Generate from compiled level
        generateFromCompiled(ctx, compiled);
    } else {
        // Fall back to default generation
        generateDefaultLevel(ctx);
    }
}

inline void generateFromCompiled(Engine &ctx, CompiledLevel* level) {
    // Simple array iteration - GPU friendly
    for (int32_t i = 0; i < level->num_tiles; i++) {
        TileType type = (TileType)level->tile_types[i];
        float x = level->tile_x[i];
        float y = level->tile_y[i];
        
        switch(type) {
            case TILE_WALL:
                makeWall(ctx, x, y);
                break;
            case TILE_CUBE:
                makeCube(ctx, x, y, 1.5f);
                break;
            // ... other types
        }
    }
}
```

### Phase 3: Python Integration with ctypes/C API

#### 3.1 C API Enhancement - CompiledLevel Struct
```c
// include/madrona_escape_room_c_api.h

// Add CompiledLevel struct to C API
typedef struct {
    int32_t num_tiles;
    int32_t max_entities; 
    int32_t width;
    int32_t height;
    float scale;
    int32_t tile_types[256];   // MAX_TILES from CompiledLevel::MAX_TILES
    float tile_x[256];
    float tile_y[256];
} MER_CompiledLevel;

// Modify manager creation function signature
MER_Result mer_create_manager(
    const MER_ManagerConfig* config,
    const MER_CompiledLevel* compiled_level,  // **NEW PARAMETER**
    MER_ManagerHandle* out_handle
);

// Optional: Add validation function
MER_Result mer_validate_compiled_level(const MER_CompiledLevel* level);
```

#### 3.2 C API Implementation
```cpp
// src/madrona_escape_room_c_api.cpp

MER_Result mer_create_manager(
    const MER_ManagerConfig* config,
    const MER_CompiledLevel* c_level,  // **NEW**
    MER_ManagerHandle* out_handle) 
{
    // Convert C struct to C++ struct
    std::optional<CompiledLevel> compiled_level;
    if (c_level != nullptr) {
        compiled_level = CompiledLevel();
        compiled_level->num_tiles = c_level->num_tiles;
        compiled_level->max_entities = c_level->max_entities;
        compiled_level->width = c_level->width;
        compiled_level->height = c_level->height; 
        compiled_level->scale = c_level->scale;
        
        // Copy arrays (only up to num_tiles for efficiency)
        std::memcpy(compiled_level->tile_types, c_level->tile_types, 
                   sizeof(int32_t) * c_level->num_tiles);
        std::memcpy(compiled_level->tile_x, c_level->tile_x,
                   sizeof(float) * c_level->num_tiles);
        std::memcpy(compiled_level->tile_y, c_level->tile_y, 
                   sizeof(float) * c_level->num_tiles);
    }
    
    // Create Manager::Config from C struct
    Manager::Config mgr_config = convertCConfig(*config);
    mgr_config.compiledLevel = compiled_level;  // **NEW FIELD**
    
    // Create manager with compiled level
    Manager* mgr = new Manager(mgr_config);
    *out_handle = reinterpret_cast<MER_ManagerHandle>(mgr);
    return MER_SUCCESS;
}

MER_Result mer_validate_compiled_level(const MER_CompiledLevel* level) {
    if (!level) return MER_ERROR_NULL_POINTER;
    if (level->num_tiles < 0 || level->num_tiles > 256) return MER_ERROR_INVALID_PARAMETER;
    if (level->max_entities < 0) return MER_ERROR_INVALID_PARAMETER;
    if (level->width <= 0 || level->height <= 0) return MER_ERROR_INVALID_PARAMETER;
    if (level->scale <= 0.0f) return MER_ERROR_INVALID_PARAMETER;
    return MER_SUCCESS;
}
```

#### 3.3 Manager::Config Extension
```cpp
// src/mgr.hpp
struct Config {
    // ... existing fields ...
    std::optional<CompiledLevel> compiledLevel = std::nullopt;  // **NEW**
};

// Sim::Config extension  
struct Config {
    // ... existing fields ...
    std::optional<CompiledLevel> compiledLevel = std::nullopt;  // **NEW**
};
```

#### 3.4 Manager Integration
```cpp
// src/mgr.cpp - Manager::Impl::init() enhancement
Manager::Impl * Manager::Impl::init(const Manager::Config &mgr_cfg) {
    // Create sim config
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);
    sim_cfg.compiledLevel = mgr_cfg.compiledLevel;  // **NEW: Pass compiled level**
    
    // ... rest of initialization unchanged ...
}
```

#### 3.5 Sim Constructor Enhancement  
```cpp
// src/sim.cpp - Sim::Sim() constructor
Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &) : WorldBase(ctx) {
    CompiledLevel &compiled_level = ctx.singleton<CompiledLevel>();
    
    // **NEW: Use config data if available**
    if (cfg.compiledLevel.has_value()) {
        compiled_level = cfg.compiledLevel.value();
    } else {
        // Phase 2 fallback - hardcoded values
        compiled_level.num_tiles = 0;
        compiled_level.width = 0; 
        compiled_level.height = 0;
        compiled_level.scale = 1.0f;
        compiled_level.max_entities = 300;  // Compiler sets this value
    }
    
    // Use dynamic BVH sizing from compiled level
    CountT max_total_entities = compiled_level.max_entities;
    
    // ... rest unchanged ...
}
```

#### 3.6 Python ctypes Bindings
```python
# madrona_escape_room/ctypes_bindings.py (or similar)

import ctypes
from typing import Optional, Dict, List

# Add CompiledLevel struct to ctypes
class MER_CompiledLevel(ctypes.Structure):
    _fields_ = [
        ("num_tiles", ctypes.c_int32),
        ("max_entities", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("tile_types", ctypes.c_int32 * 256),
        ("tile_x", ctypes.c_float * 256),
        ("tile_y", ctypes.c_float * 256),
    ]

# Update function signatures
_lib.mer_create_manager.argtypes = [
    ctypes.POINTER(MER_ManagerConfig),
    ctypes.POINTER(MER_CompiledLevel),  # **NEW**
    ctypes.POINTER(ctypes.c_void_p)
]
_lib.mer_create_manager.restype = MER_Result

_lib.mer_validate_compiled_level.argtypes = [ctypes.POINTER(MER_CompiledLevel)]
_lib.mer_validate_compiled_level.restype = MER_Result

def create_manager(config_dict: dict, compiled_level_data: Optional[dict] = None):
    """Create manager with optional compiled level data"""
    config = MER_ManagerConfig()
    # ... populate config from dict ...
    
    c_level = None
    if compiled_level_data is not None:
        c_level = MER_CompiledLevel()
        c_level.num_tiles = compiled_level_data['num_tiles']
        c_level.max_entities = compiled_level_data['max_entities']
        c_level.width = compiled_level_data['width'] 
        c_level.height = compiled_level_data['height']
        c_level.scale = compiled_level_data['scale']
        
        # Copy arrays - pad with zeros
        tile_types = compiled_level_data['tile_types']
        tile_x = compiled_level_data['tile_x'] 
        tile_y = compiled_level_data['tile_y']
        
        for i in range(256):
            if i < len(tile_types):
                c_level.tile_types[i] = tile_types[i]
                c_level.tile_x[i] = tile_x[i] 
                c_level.tile_y[i] = tile_y[i]
            else:
                c_level.tile_types[i] = 0
                c_level.tile_x[i] = 0.0
                c_level.tile_y[i] = 0.0
        
        # Validate before passing to C++
        result = _lib.mer_validate_compiled_level(ctypes.byref(c_level))
        if result != MER_SUCCESS:
            raise ValueError(f"Invalid compiled level: {result}")
    
    handle = ctypes.c_void_p()
    result = _lib.mer_create_manager(
        ctypes.byref(config),
        ctypes.byref(c_level) if c_level else None,  # **NEW**
        ctypes.byref(handle)
    )
    
    if result != MER_SUCCESS:
        raise RuntimeError(f"Manager creation failed: {result}")
    return handle
```

#### 3.7 Python Level Compiler
```python
# madrona_escape_room/level_compiler.py

from typing import Dict, List, Tuple
import math

# Tile type constants (must match C++ TileType enum)
TILE_EMPTY = 0
TILE_WALL = 1  
TILE_CUBE = 2
TILE_SPAWN = 3
TILE_DOOR = 4    # Future
TILE_BUTTON = 5  # Future
TILE_GOAL = 6    # Future

CHAR_MAP = {
    '.': TILE_EMPTY,
    ' ': TILE_EMPTY,
    '#': TILE_WALL,
    'C': TILE_CUBE, 
    'S': TILE_SPAWN,
    # Future: 'D': TILE_DOOR, 'B': TILE_BUTTON, 'G': TILE_GOAL
}

def compile_level(ascii_str: str, scale: float = 2.0) -> Dict:
    """
    Compile ASCII level string to dict for ctypes.
    
    Args:
        ascii_str: Multi-line ASCII level definition
        scale: World units per ASCII character
        
    Returns:
        Dict matching MER_CompiledLevel struct fields
    """
    lines = [line.rstrip() for line in ascii_str.strip().split('\n')]
    if not lines:
        raise ValueError("Empty level string")
    
    height = len(lines)
    width = max(len(line) for line in lines) if lines else 0
    
    tiles = []
    spawns = []
    entity_count = 0  # Count entities that need physics bodies
    
    # Parse ASCII to tiles
    for y, line in enumerate(lines):
        for x, char in enumerate(line):
            if char in CHAR_MAP:
                tile_type = CHAR_MAP[char]
                
                # Convert grid coordinates to world coordinates (center at origin)
                world_x = (x - width/2.0) * scale
                world_y = (y - height/2.0) * scale
                
                if tile_type == TILE_SPAWN:
                    spawns.append((world_x, world_y))
                    # Don't add spawn tiles to tile array - handled by agent placement
                elif tile_type != TILE_EMPTY:
                    tiles.append((world_x, world_y, tile_type))
                    
                    # Count entities that need physics bodies
                    if tile_type in [TILE_WALL, TILE_CUBE]:
                        entity_count += 1
            else:
                raise ValueError(f"Unknown character '{char}' at ({x}, {y})")
    
    # Calculate max_entities needed (entities + buffer for agents, floor, etc.)  
    max_entities = entity_count + 50  # Buffer for persistent entities
    
    # Validate constraints
    if len(tiles) > 256:
        raise ValueError(f"Too many tiles: {len(tiles)} > 256")
    if len(spawns) == 0:
        raise ValueError("No spawn points (S) found in level")
    
    # Return dict matching MER_CompiledLevel struct
    return {
        'num_tiles': len(tiles),
        'max_entities': max_entities,
        'width': width,
        'height': height, 
        'scale': scale,
        'tile_types': [tile[2] for tile in tiles] + [0] * (256 - len(tiles)),
        'tile_x': [tile[0] for tile in tiles] + [0.0] * (256 - len(tiles)),
        'tile_y': [tile[1] for tile in tiles] + [0.0] * (256 - len(tiles)),
        # Future: spawn point data for multi-agent support
    }

# Example usage
if __name__ == "__main__":
    test_level = """
    ##########
    #S.......#
    #..####..#
    #........#
    ##########
    """
    
    compiled = compile_level(test_level)
    print(f"Compiled level: {compiled['num_tiles']} tiles, max_entities={compiled['max_entities']}")
```

#### 3.8 SimManager Integration
```python
# madrona_escape_room/__init__.py

from .level_compiler import compile_level
from .ctypes_bindings import create_manager

class SimManager:
    def __init__(self, 
                 exec_mode=ExecMode.CPU,
                 gpu_id=0,
                 num_worlds=1,
                 rand_seed=42,
                 auto_reset=True,
                 level_ascii: Optional[str] = None,  # **NEW**
                 **kwargs):
        
        # Compile level if provided
        compiled_level = None
        if level_ascii is not None:
            compiled_level = compile_level(level_ascii)
        
        # Create manager config
        config = {
            'exec_mode': exec_mode,
            'gpu_id': gpu_id,
            'num_worlds': num_worlds,
            'rand_seed': rand_seed,
            'auto_reset': auto_reset,
            **kwargs
        }
        
        # Create manager with optional compiled level
        self._handle = create_manager(config, compiled_level)
        self._num_worlds = num_worlds
```

### Phase 4: Test Migration

#### 4.1 Test Fixtures
```python
# conftest.py
@pytest.fixture
def manager_with_level():
    """Factory fixture for creating managers with custom levels"""
    managers = []
    
    def _create(level_ascii, **kwargs):
        mgr = SimManager(
            exec_mode="CPU",
            num_worlds=4,
            level_ascii=level_ascii,
            seed=42,  # Fixed for determinism
            **kwargs
        )
        managers.append(mgr)
        return mgr
    
    yield _create
    
    # Cleanup
    for mgr in managers:
        del mgr
```

#### 4.2 Example Test Conversions
```python
# test_movement_system.py
def test_forward_movement_clear_path(manager_with_level):
    """Test forward movement with no obstacles"""
    level = """
    ##########
    #S.......#
    #........#
    #........#
    ##########
    """
    mgr = manager_with_level(level)
    
    # Agent starts at S (1,1) in world coordinates (2,2)
    obs = mgr.self_observation_tensor().to_torch()
    initial_y = obs[0, 1]  # Y position
    
    # Move forward
    actions = mgr.action_tensor().to_torch()
    actions[:, 0] = MoveAmount.FAST
    actions[:, 1] = MoveAngle.FORWARD
    
    for _ in range(10):
        mgr.step()
    
    # Should have moved forward
    obs = mgr.self_observation_tensor().to_torch()
    assert obs[0, 1] > initial_y

def test_obstacle_collision(manager_with_level):
    """Test that agent can't walk through walls"""
    level = """
    ##########
    #S#......#
    #.#......#
    #........#
    ##########
    """
    mgr = manager_with_level(level)
    
    # Agent at S, wall immediately to the right
    # Try to move right - should be blocked
    # ... test continues
```

## Benefits

### For Testing
1. **Visual Clarity**: ASCII art makes test environment immediately clear
2. **Self-Contained**: Test defines its own environment
3. **Deterministic**: Same ASCII always produces same level
4. **Versioned**: Level changes tracked with test changes

### For Development
1. **Test Stability**: Can change default levels without breaking tests
2. **Rapid Iteration**: New level designs without recompilation
3. **Debugging**: Can reproduce exact test conditions easily
4. **Documentation**: Level IS the documentation

### Technical Benefits
1. **GPU Compatible**: Fixed-size arrays, no dynamic allocation
2. **Efficient**: Simple array iteration on GPU
3. **Extensible**: Easy to add new tile types
4. **Backwards Compatible**: Existing tests continue to work

## Success Criteria

1. **Functionality**
   - [ ] Tests can define levels as ASCII strings
   - [ ] Levels compile correctly to GPU format
   - [ ] GPU generation produces expected geometry
   - [ ] All existing tests still pass

2. **Performance**
   - [ ] No measurable performance regression
   - [ ] Level compilation < 1ms
   - [ ] GPU generation time unchanged

3. **Usability**
   - [ ] Clear documentation and examples
   - [ ] Helpful error messages for invalid ASCII
   - [ ] Easy to add new tile types

## Risks and Mitigations

### Risk 1: GPU Memory Overhead
**Risk**: Fixed-size buffers waste memory for simple levels
**Mitigation**: Use reasonable MAX_TILES (256 = 1KB per world)

### Risk 2: ASCII Parsing Complexity
**Risk**: Complex parsing logic for advanced features
**Mitigation**: Start simple, iterate based on needs

### Risk 3: Test Migration Effort
**Risk**: Many tests to update
**Mitigation**: Backwards compatibility, gradual migration

## Timeline

**IMPORTANT**: Always verify you are on the `feature/test-driven-levels` branch before making changes:
```bash
git branch  # Should show * feature/test-driven-levels
```

### Completed
- [x] Screenshot capture tool with PNG support
- [x] Hide-menu option for clean screenshots
- [x] Visual verification infrastructure
- [x] Phase 1.1: Hardcoded 16x16 room in generateLevel()
- [x] Test on CPU and GPU modes
- [x] Visual verification with screenshot tool
- [x] Phase 2: CompiledLevel struct and TileType enum
- [x] Phase 2: generateFromCompiled function with GPU-friendly array iteration
- [x] Phase 2: Conditional level generation (compiled vs hardcoded fallback)
- [x] Phase 2: Singleton registration and initialization

### Remaining Tasks

#### Phase 2: Core Infrastructure (COMPLETED ✅)
- [x] Implement CompiledLevel struct
- [x] GPU generateFromCompiled function

#### Phase 3: Python Integration (ctypes/C API)
- [ ] C API enhancement with MER_CompiledLevel struct
- [ ] Manager::Config and Sim::Config extension  
- [ ] Python ctypes bindings update
- [ ] Python level compiler implementation
- [ ] SimManager integration with level_ascii parameter

#### Phase 4: Test Migration
- [ ] First test using ASCII level
- [ ] Migrate key tests
- [ ] Documentation
- [ ] Performance validation

## Branch Management

**Critical Reminder**: This feature is being developed on the `feature/test-driven-levels` branch. 

### Before Starting Any Work:
1. **Always check current branch**: `git branch`
2. **Switch to correct branch if needed**: `git checkout feature/test-driven-levels`
3. **Verify you're on the right branch**: Look for `* feature/test-driven-levels` in output

### Why This Matters:
- Phase 1.1 camera positioning fixes are ONLY on `feature/test-driven-levels`
- MCP viewer server changes belong on `feature/mcp-viewer-server`
- Working on the wrong branch causes merge conflicts and regressions

## Appendix: ASCII Level Examples

### Empty Room
```
##########
#........#
#........#
#........#
##########
```

### Corridor
```
##########
#S.......#
##.#######
#........#
##########
```

### Maze
```
############
#S.#.......#
#..#.####..#
##.#....#..#
#..####.##.#
#.......#..#
########..##
```

### Multi-Agent
```
##########
#S1......#
#........#
#......S2#
##########
```