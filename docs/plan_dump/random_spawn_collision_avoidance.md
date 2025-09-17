# Random Spawn Location with Collision Avoidance Plan

## Overview
Implement a streamlined random spawn system that only needs to check the room entities array (which includes both persistent and dynamic objects).

## Problem Statement
The current agent spawning system uses fixed positions from CompiledLevel spawn data. We need to implement random spawn locations that avoid collisions with all objects using a fixed exclusion radius, and this needs to work efficiently on GPU with NVRTC compilation.

## Current System Analysis

**Current Spawn System:**
- Uses **fixed spawn positions** from `CompiledLevel.spawn_x/y[]` arrays
- Falls back to center position with slight offset if no spawn data
- No collision avoidance with objects

**Current Object Placement:**
- Objects placed from `CompiledLevel.tile_x/y/z[]` arrays
- Has **randomization support** via `tile_rand_x/y/z[]` ranges
- Objects can be persistent (survive resets) or dynamic (regenerated each episode)

**Execution Order (Critical for Implementation):**
1. `resetPersistentEntities()` - re-registers persistent entities
2. `generateLevel()` → `generateFromCompiled()` - creates all the dynamic objects
3. Inside `generateFromCompiled()`, at the end it calls `resetAgentPhysics()`

So when `resetAgentPhysics()` is called, all objects have been created and are available in `level_state.rooms[0].entities[]`.

## Simplified Implementation

### Key Insights
1. **No need to check other agents** - we only have 1 agent per world (consts::numAgents = 1)
2. **Persistent entities ARE in room entities array** - line 473 in generateFromCompiled() adds them
3. **Single loop suffices** - check `level_state.rooms[0].entities[]` which contains everything

### 1. Add Spawn Function
Create this function in `src/level_gen.cpp` before `resetAgentPhysics()`:

```cpp
static inline Vector2 findValidSpawnPosition(Engine &ctx, float exclusion_radius)
{
    CompiledLevel& level = ctx.singleton<CompiledLevel>();
    LevelState& level_state = ctx.singleton<LevelState>();

    // Use world episode for RNG variation
    RandKey spawn_key = rand::split_i(ctx.data().rng.randKey(),
                                       ctx.data().curWorldEpisode);

    const float WALL_MARGIN = 2.0f;
    const int MAX_ATTEMPTS = 30;
    float exclusion_sq = exclusion_radius * exclusion_radius;

    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        // Generate candidate position
        RandKey attempt_key = rand::split_i(spawn_key, attempt);
        Vector2 candidate = {
            rand::sampleUniform(attempt_key) *
                (level.world_max_x - level.world_min_x - 2*WALL_MARGIN) +
                level.world_min_x + WALL_MARGIN,
            rand::sampleUniform(rand::split_i(attempt_key, 1)) *
                (level.world_max_y - level.world_min_y - 2*WALL_MARGIN) +
                level.world_min_y + WALL_MARGIN
        };

        bool valid = true;

        // Single loop - checks ALL entities (persistent + dynamic)
        for (CountT i = 0; i < CompiledLevel::MAX_TILES; i++) {
            Entity e = level_state.rooms[0].entities[i];
            if (e == Entity::none()) continue;

            Position entity_pos = ctx.get<Position>(e);

            // Skip floor (z == 0)
            if (entity_pos.z < 0.1f) continue;

            float dx = candidate.x - entity_pos.x;
            float dy = candidate.y - entity_pos.y;
            float dist_sq = dx*dx + dy*dy;

            if (dist_sq < exclusion_sq) {
                valid = false;
                break;
            }
        }

        if (valid) {
            return candidate;
        }
    }

    // Fallback: Center position with small offset
    return Vector2{0.0f, 0.0f};
}
```

### 2. Modify resetAgentPhysics()
Replace lines 104-120 in `src/level_gen.cpp`:

```cpp
// Comment out old spawn logic
// Vector3 pos;
// if (i < level.num_spawns) {
//     pos = Vector3 {
//         level.spawn_x[i],
//         level.spawn_y[i],
//         1.0f
//     };
// } else {
//     pos = Vector3 {
//         i * consts::rendering::agentSpacing - 1.0f,
//         0.0f,
//         1.0f,
//     };
// }

// New: Random spawn with collision avoidance
const float EXCLUSION_RADIUS = 3.0f;
Vector2 spawn_2d = findValidSpawnPosition(ctx, EXCLUSION_RADIUS);
Vector3 pos = Vector3{spawn_2d.x, spawn_2d.y, 1.0f};

ctx.get<Position>(agent_entity) = pos;

// Comment out fixed facing angle
// float facing_angle = 0.0f;
// if (i < level.num_spawns) {
//     facing_angle = level.spawn_facing[i];
// }

// Random facing
float facing_angle = ctx.data().rng.sampleUniform() * 2.0f * math::pi;
```

## Key Simplifications

1. **Single Loop**: Only check `rooms[0].entities[]` which contains everything
2. **No Agent Checking**: Only 1 agent per world
3. **Floor Skip**: Simple z-check to skip floor entity
4. **Direct Access**: No queries, just array iteration (GPU-efficient)
5. **Simple Fallback**: Just return center if no valid spot found

## Why This Works on GPU

1. **Direct Array Access**: No dynamic queries, just iterate through fixed-size arrays
2. **Per-Episode RNG**: Uses episode counter for variation via `split_i()`
3. **No Branches in Inner Loop**: Simple distance checks with early exit
4. **Fixed Memory**: All arrays are statically sized (MAX_TILES)
5. **Coalesced Access**: Sequential array iteration is GPU-friendly
6. **NVRTC Compatible**: No STL, no dynamic allocation, uses Madrona RNG

## Benefits
- Works with NVRTC constraints (no STL, no dynamic allocation)
- Checks against ALL actual entities (not just theoretical positions)
- Handles randomized object positions correctly
- Guaranteed termination with simple fallback
- Efficient GPU execution with fixed-size loops
- Uses Madrona's high-quality Threefry RNG system

## Testing Plan
1. **Build**: `./build.sh`
2. **Visual verification**: `./build/viewer` - verify agent spawns in different positions each reset
3. **Performance test**: `./build/headless --num-worlds 1000`
4. **Python tests**: `uv run python tests/test_tracker.py --dry-run`

## Files to Modify
- `src/level_gen.cpp`: Add spawn position function and modify `resetAgentPhysics()`

## Parameters (Hardcoded for Initial Implementation)
- `EXCLUSION_RADIUS = 3.0f` - 3 units clearance around all objects
- `WALL_MARGIN = 2.0f` - Keep away from world boundaries
- `MAX_ATTEMPTS = 30` - Rejection sampling attempts before fallback
- Random facing angle instead of fixed level data