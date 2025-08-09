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

Start with the simplest possible implementation: a hardcoded 16x16 empty room that generates directly on GPU.

#### Visual Verification During Development
Use the screenshot capture tool to verify level generation:

```bash
# Capture a screenshot of the initial level state
python scripts/capture_level_screenshot.py -n 1 -o level_test.bmp

# View the screenshot using the Read tool to verify:
# - Walls are placed correctly
# - Agent spawn positions are correct  
# - Room dimensions match expectations
```

#### 1.1 Add makeWall helper and modify generateLevel() with Hardcoded Room
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

#### 1.2 Test Both CPU and GPU Execution
- Build and run on CPU mode
- Build and run on GPU mode  
- Use screenshot tool to verify visual output
- Ensure identical behavior between CPU and GPU

### Phase 2: Core Infrastructure

#### 1.1 GPU Data Structure (C++)
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

#### 1.2 Python Compiler
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

#### 1.3 Manager Integration
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

#### 1.4 GPU Level Generation
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

### Phase 2: Python Bindings

#### 2.1 Update SimManager
```python
# __init__.py
class SimManager:
    def __init__(self, 
                 exec_mode="CPU",
                 num_worlds=1,
                 level_ascii=None,  # NEW
                 **kwargs):
        
        # Compile level if provided
        level_buffer = None
        if level_ascii:
            compiler = LevelCompiler()
            level_buffer = compiler.compile(level_ascii)
        
        # Pass to C API
        self._handle = lib.mer_create_manager(
            exec_mode=exec_mode,
            num_worlds=num_worlds,
            compiled_level=level_buffer,  # NEW
            **kwargs
        )
```

#### 2.2 C API Updates
```cpp
// madrona_escape_room_c_api.cpp
mer_manager* mer_create_manager(
    const char* exec_mode,
    int32_t num_worlds,
    const void* compiled_level,  // NEW
    size_t level_size,           // NEW
    // ... other params
) {
    Manager::Config cfg;
    // ... existing config ...
    
    if (compiled_level != nullptr) {
        cfg.use_compiled_level = true;
        cfg.compiled_level_data = compiled_level;
        cfg.compiled_level_size = level_size;
    }
    
    return new Manager(cfg);
}
```

### Phase 3: Test Migration

#### 3.1 Test Fixtures
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

#### 3.2 Example Test Conversions
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

### Week 1
- [ ] Implement CompiledLevel struct
- [ ] Basic LevelCompiler in Python
- [ ] Manager integration

### Week 2  
- [ ] GPU generateFromCompiled function
- [ ] Python bindings updates
- [ ] First test using ASCII level

### Week 3
- [ ] Migrate key tests
- [ ] Documentation
- [ ] Performance validation

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