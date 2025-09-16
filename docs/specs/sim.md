# Simulation Core Specification

## Overview
The simulation core (`sim.cpp`) is the heart of the Madrona Escape Room environment, implementing the Entity Component System (ECS) architecture that drives all game logic. It manages agent movement, physics, collision detection, observations, rewards, and episode resets in a highly parallel, GPU-optimized manner.

## Key Files

### Source Code
Primary implementation files:

- `src/sim.cpp` - Core ECS simulation with system registration and task graph setup
- `src/sim.hpp` - Simulation class and enum definitions (TaskGraphID, ExportID, SimObject)
- `src/sim.inl` - Template implementations for renderable entity creation/destruction
- `src/types.hpp` - ECS component definitions and archetype structures
- `src/level_gen.cpp` - World generation from compiled level data
- `src/consts.hpp` - Game constants and action value definitions

### Test Files

#### C++ Tests
- `tests/cpp/test_asset_registry.cpp` - Asset loading and registry tests
- `tests/cpp/test_viewer_core.cpp` - Viewer core functionality tests

#### Python Tests
- `tests/python/test_bindings.py` - Core simulation behavior validation
- `tests/python/test_reward_termination_system.py` - Reward calculation and termination tests
- `tests/python/test_movement_system.py` - Movement system tests
- `tests/python/test_compass_tensor.py` - Compass observation system tests

## Architecture

### System Integration
The simulation integrates with:
- **Madrona Engine**: Base ECS framework providing entity management and parallelization
- **Physics System**: XPBD-based physics with collision detection
- **Rendering System**: Optional visualization pipeline
- **Manager Layer**: CPU-side interface for Python bindings and tensor exports

### GPU/CPU Code Separation
- **GPU (NVRTC) Code**: `sim.cpp`, `level_gen.cpp`, `types.hpp` - Compiled for GPU execution
- **CPU-Only Code**: `mgr.cpp` - Manager layer that orchestrates GPU simulation
- **Shared Headers**: `sim.hpp`, `consts.hpp` - Common definitions used by both

## Implementation

### Data Structures

#### Entity-Component Matrix
Overview of which components belong to which entity types:

| Component | Description | Agent | PhysicsEntity | RenderOnlyEntity | LidarRayEntity |
|-----------|-------------|-------|---------------|------------------|----------------|
| Position | World position | ✓ | ✓ | ✓ | ✓ |
| Rotation | Orientation quaternion | ✓ | ✓ | ✓ | ✓ |
| Scale | Entity scale | ✓ | ✓ | ✓ | ✓ |
| ObjectID | Rendering object ID | ✓ | ✓ | ✓ | ✓ |
| Velocity | Linear and angular velocity | ✓ | ✓ | | |
| ResponseType | Physics response (Dynamic/Static) | ✓ | ✓ | | |
| ExternalForce | Applied forces | ✓ | | | |
| ExternalTorque | Applied torques | ✓ | | | |
| Action | Agent control input | ✓ | | | |
| Reward | RL reward signal | ✓ | | | |
| Done | Episode termination flag | ✓ | | | |
| CollisionDeath | Collision termination tracker | ✓ | | | |
| TerminationReason | Termination code | ✓ | | | |
| SelfObservation | Agent position/progress obs | ✓ | | | |
| CompassObservation | 128-bucket direction encoding | ✓ | | | |
| Lidar | 128-sample depth array | ✓ | | | |
| Progress | Forward progress tracking | ✓ | | | |
| StepsTaken | Episode step counter | ✓ | | | |
| EntityType | Entity classification | ✓ | ✓ | | |
| DoneOnCollide | Triggers episode end on collision | | ✓ | | |

#### Archetype Definitions
```cpp
struct Agent : madrona::Archetype<
    Position, Rotation, Scale, ObjectID,
    Velocity, ResponseType,
    ExternalForce, ExternalTorque,
    Action, Reward, Done, CollisionDeath, TerminationReason,
    SelfObservation, CompassObservation, Lidar,
    Progress, StepsTaken, EntityType
> {};

struct PhysicsEntity : madrona::Archetype<
    Position, Rotation, Scale, ObjectID,
    Velocity, ResponseType,
    EntityType, DoneOnCollide
> {};

struct RenderOnlyEntity : madrona::Archetype<
    Position, Rotation, Scale, ObjectID
> {};

struct LidarRayEntity : madrona::Archetype<
    Position, Rotation, Scale, ObjectID
> {};
```

#### Component Definitions
```cpp
// Action component - discrete agent controls
struct Action {
    int32_t moveAmount;   // 0-3: stop/slow/medium/fast
    int32_t moveAngle;    // 0-7: 8 directional movement
    int32_t rotate;       // 0-4: turn left/right speeds
};

// Progress tracking for reward calculation
struct Progress {
    float maxY;      // Maximum Y position achieved
    float initialY;  // Starting Y position for normalization
};

// Termination reason codes
struct TerminationReason {
    int32_t code;  // -1: not terminated, 0: steps, 1: goal, 2: collision
};
```

### Core Systems

#### movementSystem
- **Purpose**: Converts discrete actions to physics forces and torques
- **Components Used**: Reads: `Action`, `Rotation`; Writes: `ExternalForce`, `ExternalTorque`
- **Task Graph Dependencies**: First system in step, feeds into physics
- **Specifications**:
  - **Move force**: 0-1000N mapped from 4 buckets (STOP=0, SLOW=333, MEDIUM=666, FAST=1000)
  - **Move angle**: 8 directions at 45° increments (FORWARD=0°, FORWARD_RIGHT=45°, RIGHT=90°, etc.)
  - **Turn torque**: ±640 Nm from 5 buckets (FAST_LEFT=-640, SLOW_LEFT=-320, NONE=0, SLOW_RIGHT=320, FAST_RIGHT=640)
  - **Forces in local space**: Applied in agent coordinates then rotated to world space
  - **Strafing**: LEFT/RIGHT move angles produce perpendicular movement relative to facing
  - **Combined actions**: Move and rotate can execute simultaneously

#### agentCollisionSystem
- **Purpose**: Detects agent collisions and triggers termination for fatal obstacles
- **Components Used**: Reads: `EntityType`, `DoneOnCollide`, `ContactConstraint`; Writes: `Done`, `CollisionDeath`, `TerminationReason`
- **Task Graph Dependencies**: Runs within physics substep loop, after narrowphase
- **Specifications**:
  - **Floor immunity**: Floor collisions always ignored
  - **Selective termination**: Only DoneOnCollide=true entities terminate episode
  - **Per-tile config**: Each level tile can specify collision behavior independently
  - **Collision death termination**:
    - Occurs immediately upon contact with DoneOnCollide=true entity
    - Sets done=1, collision_death=1, and termination_code=2
    - Reward is overridden to -0.1 (penalty)
    - Different objects can have different collision behaviors per level
    - No partial collision - either terminates or doesn't (no health system)

#### agentZeroVelSystem
- **Purpose**: Improves agent controllability by zeroing velocity after physics
- **Components Used**: Reads: `Action`; Writes: `Velocity`
- **Task Graph Dependencies**: After physics integration, before cleanup
- **Specifications**:
  - **Linear velocity**: X and Y components zeroed each frame
  - **Z velocity**: Clamped to <= 0 (allows falling but not upward motion)
  - **Angular velocity**: Completely zeroed to prevent rotation drift
  - **Control improvement**: Prevents sliding and makes movement more responsive

#### stepTrackerSystem
- **Purpose**: Tracks episode steps and triggers timeout termination
- **Components Used**: Writes: `StepsTaken`, `Done`, `TerminationReason`
- **Task Graph Dependencies**: After physics cleanup, before reward system
- **Specifications**:
  - **Step increment**: StepsTaken.t++ each frame
  - **Episode length**: Exactly 200 steps (consts::episodeLen)
  - **Step limit termination**:
    - Occurs exactly at step 200
    - Sets done=1 and termination_code=0
    - Reward is based on progress achieved (can be 0 if stationary)
    - Auto-reset triggers on next step if enabled
  - **Reset behavior**: Counter resets to 0 on episode reset

#### rewardSystem
- **Purpose**: Calculates incremental rewards based on forward progress
- **Components Used**: Reads: `Position`, `Progress`, `CollisionDeath`, `CompiledLevel`; Writes: `Reward`, `Done`, `TerminationReason`
- **Task Graph Dependencies**: After stepTrackerSystem, before resetSystem
- **Specifications**:
  - **Step 0**: Always 0.0 reward (no reward on reset)
  - **Forward only**: Only Y-axis forward movement gives rewards
  - **Incremental**: Reward = (new_maxY - old_maxY) / total_possible_progress
  - **High-water mark**: Progress tracked as maxY, never decreases
  - **Backward/lateral movement**: No reward (only forward progress counts)
  - **Stationary agent**: No reward (must move forward to earn rewards)
  - **Normalization**: Total rewards sum to ~1.0 for complete traversal
  - **Collision override**: Death penalty -0.1 overrides any progress reward
  - **Goal achievement termination**:
    - Occurs when normalized progress >= 1.0
    - Agent has reached or exceeded world_max_y
    - Sets done=1 and termination_code=1
    - Total accumulated rewards ≈ 1.0 for complete traversal
    - Represents successful episode completion

#### resetSystem
- **Purpose**: Manages episode resets when agents complete or external reset is triggered
- **Components Used**: Reads: `Done`, `WorldReset`; Writes: `WorldReset`
- **Task Graph Dependencies**: After reward system, triggers world regeneration
- **Specifications**:
  - **Deferred reset**: Waits one step after done=1 to allow observation
  - **Auto-reset behavior**:
    - When auto_reset=True and done=1, sets reset flag for next step
    - Episode resets automatically one step after termination
    - Python code can observe final state before reset
    - Step 0 after reset always has reward=0.0
    - Agent returns to spawn position defined in level
    - Episode counter increments
    - Progress tracking reinitializes
  - **Entity cleanup**: Preserves persistent entities (floor) while removing level entities
  - **World regeneration**: Calls generateWorld() with new RNG seed
  - **Episode counter**: Increments curWorldEpisode on each reset

#### initProgressAfterReset
- **Purpose**: Initializes progress tracking after physics settles post-reset
- **Components Used**: Reads: `Position`; Writes: `Progress`, `TerminationReason`
- **Task Graph Dependencies**: After post-reset BVH build, before observations
- **Specifications**:
  - **Sentinel detection**: Checks for -999999.0 to identify uninitialized state
  - **Initial values**: Sets maxY and initialY to current Y position
  - **Termination codes**:
    - Code -1: Not terminated (episode still running) - set on initialization
    - Code 0: Episode steps reached (hit 200-step limit) - set by stepTrackerSystem
    - Code 1: Goal achieved (reached world_max_y) - set by rewardSystem
    - Code 2: Collision death (hit DoneOnCollide=true entity) - set by agentCollisionSystem
  - **Physics settled**: Runs after physics to get stable position
  - **Export safety**: Ensures valid Progress values for tensor export

#### collectObservationsSystem
- **Purpose**: Packages agent state into normalized observations for policy
- **Components Used**: Reads: `Position`, `Rotation`, `Progress`, `CompiledLevel`; Writes: `SelfObservation`
- **Task Graph Dependencies**: After initProgressAfterReset, feeds tensor export
- **Specifications**:
  - **Position normalization**: Maps world coordinates to [0,1] range
  - **Progress observation**: maxY = (progress.maxY - initialY) / (world_max_y - initialY)
  - **Rotation encoding**: theta normalized to [-1,1] from [-π,π]
  - **Boundary awareness**: Uses CompiledLevel world boundaries
  - **Uninitialized handling**: Returns 0.0 for maxY if progress not initialized

#### compassSystem
- **Purpose**: Computes one-hot encoding of agent facing direction
- **Components Used**: Reads: `Rotation`; Writes: `CompassObservation`
- **Task Graph Dependencies**: After collectObservations, parallel with lidar
- **Specifications**:
  - **128 buckets**: Full 360° coverage with 2.8125° per bucket
  - **Encoding formula**: bucket = (64 - int(theta_radians / 2π * 128)) % 128
  - **One-hot**: Single 1.0 value, rest 0.0
  - **Angle wrapping**: Handles -π to π range correctly
  - **North alignment**: Bucket 64 represents forward (0 radians)

#### lidarSystem
- **Purpose**: Casts 128 rays for depth perception observations
- **Components Used**: Reads: `Position`, `Rotation`, BVH; Writes: `Lidar`
- **Task Graph Dependencies**: After post-reset BVH, parallel with compass
- **Specifications**:
  - **Field of view**: 120° arc (-60° to +60° from forward)
  - **Ray count**: 128 evenly distributed samples
  - **Max range**: 200 units (consts::lidarMaxRange)
  - **Depth normalization**: Returns depth/maxRange, clamped to [0,1]
  - **No hit**: Returns 0.0 when ray doesn't hit anything
  - **GPU parallelism**: 128 threads (4 warps) trace rays simultaneously
  - **Visualization**: Optional display of every 8th ray (16 total) when enabled

## Performance Considerations

### GPU Optimization
- **Warp-Level Parallelism**: Lidar system uses 128 threads (4 warps) for ray tracing
- **Coalesced Memory Access**: Components stored in structure-of-arrays layout
- **Entity Sorting**: Periodic sorting by WorldID for better cache locality
- **Contact Sorting**: Sorts collision contacts for efficient constraint solving
- **BVH Caching**: Broadphase acceleration structure rebuilt only on reset

## CompiledLevel Changes
The simulation relies on CompiledLevel singleton for world boundaries and entity limits:
```cpp
struct CompiledLevel {
    // World boundaries for normalization
    float world_min_x, world_max_x;
    float world_min_y, world_max_y;
    float world_min_z, world_max_z;

    // Entity management
    uint32_t max_entities;  // For BVH sizing

    // Level data arrays...
};
```

## Build Configuration

### CMake Changes
```cmake
# SIMULATOR_SRCS includes files compiled by NVRTC for GPU
set(SIMULATOR_SRCS
    sim.cpp
    level_gen.cpp
)
```