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

struct TargetEntity : madrona::Archetype<
    Position, Rotation, Scale, ObjectID,
    Velocity, MotionParams, TargetTag,
    madrona::render::Renderable
> {};
```

#### Determinism Invariants
- **Single PRNG Chain**: All randomness derives from the initial seed through a single deterministic PRNG
- **No External Entropy**: No time-based seeds, system randomness, or other non-deterministic inputs
- **Reproducible Resets**: Episode resets must occur at identical steps given the same seed + actions
- **Deterministic Spawns**: Random spawn positions must use the simulation's PRNG exclusively
- **Replay Guarantee**: Given the same seed and action sequence, the simulation MUST produce identical results

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

// Target entity identification
struct TargetTag {
    int32_t id;  // Target identifier (0 = primary target)
};

// Custom motion parameters
struct MotionParams {
    float omega_x, omega_y;    // Harmonic frequencies
    float center_x, center_y, center_z;  // Equilibrium position
    float mass;                // Mass for dynamics
    int32_t motion_type;       // 0=static, 1=harmonic
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
- **Purpose**: Calculates completion-based rewards when agent reaches level goal
- **Components Used**: Reads: `Position`, `Progress`, `CollisionDeath`, `CompiledLevel`; Writes: `Reward`, `Done`, `TerminationReason`
- **Task Graph Dependencies**: After stepTrackerSystem, before resetSystem
- **Specifications**:
  - **Step 0**: Always 0.0 reward (no reward on reset)
  - **Completion only**: Only completion gives rewards (no incremental progress rewards)
  - **Completion condition**: Reward = 1.0 when agent Y position >= world_max_y
  - **Non-completion**: 0.0 reward for all other steps
  - **High-water mark**: Progress tracked as maxY for termination detection
  - **Movement tracking**: No reward for forward movement until completion
  - **Collision override**: Death penalty -0.1 overrides any completion reward
  - **Goal achievement termination**:
    - Occurs when agent Y position >= world_max_y
    - Sets done=1 and termination_code=1
    - Reward = 1.0 for successful completion
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
- **Purpose**: Computes one-hot encoding pointing toward target entity
- **Components Used**: Reads: `Entity`, `Position`, `TargetTag`; Writes: `CompassObservation`
- **Task Graph Dependencies**: After collectObservations, parallel with lidar
- **Specifications**:
  - **Target tracking**: Points toward primary target (TargetTag.id == 0)
  - **Fallback behavior**: Uses agent rotation if no target found
  - **Angle calculation**: `atan2f(target.y - agent.y, target.x - agent.x)`
  - **128 buckets**: Full 360° coverage with 2.8125° per bucket
  - **Encoding formula**: bucket = (64 - int(theta_radians / 2π * 128)) % 128
  - **One-hot**: Single 1.0 value, rest 0.0

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

#### customMotionSystem
- **Purpose**: Applies custom equations of motion to target entities
- **Components Used**: Reads: `MotionParams`; Writes: `Position`, `Velocity`
- **Task Graph Dependencies**: After movementSystem, before physics broadphase
- **Specifications**:
  - **Motion types**: 0=static, 1=harmonic oscillator (extensible via templates)
  - **Timestep**: `dt = consts::deltaT / consts::numPhysicsSubsteps`
  - **Physics isolation**: Target entities not registered with physics system
  - **NVRTC compatibility**: Template specializations with runtime switch

## Performance Considerations

### GPU Optimization
- **Warp-Level Parallelism**: Lidar system uses 128 threads (4 warps) for ray tracing
- **Coalesced Memory Access**: Components stored in structure-of-arrays layout
- **Entity Sorting**: Periodic sorting by WorldID for better cache locality
- **Contact Sorting**: Sorts collision contacts for efficient constraint solving
- **BVH Caching**: Broadphase acceleration structure rebuilt only on reset

# Initialization Sequence

## Overview
The initialization sequence creates and prepares the Madrona Escape Room simulation, establishing the Manager interface, loading assets, registering ECS types, constructing worlds, and executing the first simulation step to populate initial observations.

```
Manager::Manager() Constructor
├─ Manager::Impl::init()
│  ├─ Step 1: Simulation Configuration
│  │  └─ Setup physics loader & RNG
│  ├─ Step 2: loadPhysicsObjects()
│  │  ├─ Load collision meshes (cube, wall, agent)
│  │  └─ Configure physics properties
│  ├─ Step 3: loadRenderObjects() [if rendering]
│  │  ├─ Load visual meshes
│  │  └─ Configure materials & lighting
│  ├─ Step 4: Create Executor
│  │  ├─ TaskGraphExecutor (CPU) or MWCudaExecutor (GPU)
│  │  └─ Triggers Executor Initialization Phase ────┐
│  ├─ Step 5: getExported() calls                   │
│  │  └─ Retrieve component data pointers           │
│  └─ Step 6: Initial World Setup                   │
│     ├─ triggerReset() for all worlds              │
│     └─ step() to populate observations            │
│                                                    │
├─ Executor Initialization Phase ◄──────────────────┘
│  ├─ Step 1: ThreadPoolExecutor base setup
│  ├─ Step 2: Sim::registerTypes()
│  │  ├─ Register components & archetypes
│  │  └─ Export components for Python
│  ├─ Step 3: World Construction
│  │  └─ Per-World Sim Construction ─────────┐
│  ├─ Step 4: Sim::setupTasks()              │
│  │  ├─ Build task graph                    │
│  │  ├─ Define system execution order       │
│  │  └─ Configure dependencies              │
│  └─ Step 5: initExport()                   │
│     └─ Commit physical memory              │
│                                             │
├─ Per-World Sim Construction ◄──────────────┘
│  ├─ Step 1: Sim::Sim() Constructor
│  │  ├─ Calculate entity limits
│  │  └─ Initialize physics system
│  ├─ Step 2: createPersistentEntities()
│  │  ├─ Create floor & walls
│  │  └─ Create agents with cameras
│  └─ Step 3: initWorld()
│     ├─ PhysicsSystem::reset()
│     └─ generateWorld()
│        ├─ resetPersistentEntities()
│        └─ generateLevel()
│
└─ First Step Execution
   ├─ Step 1: triggerReset() all worlds
   ├─ Step 2: step() → task graph execution
   ├─ Step 3: RenderManager::readECS() [if rendering]
   └─ Step 4: collectObservationsSystem populates tensors
```

## Input

The initialization process receives configuration parameters from Python that determine simulation behavior.

### Input Sources
- **Python Configuration**: Execution mode, number of worlds, episode length, auto-reset flag
- **CompiledLevel Data**: Binary level structure containing tile placements, spawn positions, world boundaries
- **Asset Files**: OBJ files for physics meshes (cube.obj, wall.obj, agent.obj)
- **Render Assets**: Visual meshes and texture images (if rendering enabled)

### Input Data Format

#### Simple Values
- **exec_mode** (`ExecMode`): CPU or CUDA execution mode
- **num_worlds** (`int32_t`): Number of parallel simulation worlds
- **episode_len** (`int32_t`): Steps per episode (default: 200)
- **auto_reset** (`bool`): Whether to automatically reset completed episodes
- **render_width** (`int32_t`): Render resolution width (if rendering)
- **render_height** (`int32_t`): Render resolution height (if rendering)

#### Structured Data
**Manager::Config**
```cpp
struct Config {
    ExecMode execMode;           // CPU or CUDA execution
    int32_t numWorlds;            // Parallel world count
    int32_t episodeLen;           // Steps per episode
    bool autoReset;               // Auto-reset on done
    bool enableRendering;         // Enable visual rendering
    int32_t renderWidth;          // Render target width
    int32_t renderHeight;         // Render target height
};
```

**CompiledLevel** (Critical Input)
```cpp
struct CompiledLevel {
    // Header fields
    int32_t num_tiles;            // Actual tiles used in level
    int32_t max_entities;         // For BVH sizing - critical for physics
    int32_t width, height;        // Grid dimensions
    float world_scale;            // World scale factor
    char level_name[64];          // Level identification

    // World boundaries (for observation normalization)
    float world_min_x, world_max_x;
    float world_min_y, world_max_y;
    float world_min_z, world_max_z;

    // Spawn data arrays [MAX_SPAWNS]
    int32_t num_spawns;
    float spawn_x[8], spawn_y[8], spawn_z[8];
    float spawn_facing[8];        // Initial agent rotations
    bool spawn_random;            // Whether to use random spawn positions instead of fixed ones

    // Tile data arrays [MAX_TILES = 1024]
    uint32_t object_ids[1024];    // Asset IDs (cube, wall, etc)
    float tile_x[1024], tile_y[1024], tile_z[1024];  // Positions
    Quat tile_rotation[1024];     // Orientations
    float tile_scale_x[1024], tile_scale_y[1024], tile_scale_z[1024];

    // Tile properties
    bool tile_persistent[1024];   // Survives episode resets
    bool tile_render_only[1024];  // No physics collision
    bool tile_done_on_collide[1024];  // Triggers episode end
    int32_t tile_entity_type[1024];    // EntityType enum
    int32_t tile_response_type[1024];  // Static/Dynamic/Kinematic

    // Randomization ranges (for non-persistent tiles)
    float tile_rand_x[1024], tile_rand_y[1024], tile_rand_z[1024];
    float tile_rand_rot_z[1024];
    float tile_rand_scale_x[1024], tile_rand_scale_y[1024], tile_rand_scale_z[1024];
};
```

## Processing

The initialization transforms configuration parameters into a fully-initialized simulation ready for stepping.

### Processing Pipeline
```
Python Config → Manager Creation → Asset Loading → ECS Registration → World Construction → First Step → Ready State
```

### Detailed Sequence

#### Phase 1: Manager Creation Phase
`Manager::Manager()` constructor initiates the entire initialization sequence.

#### Step 1: Simulation Configuration
**Function:** `Manager::Impl::init()`
**Location:** `src/mgr.cpp`
**Purpose:** Initialize simulation configuration based on execution mode

**Details:**
- Creates PhysicsLoader for CPU mode
- Calls loadPhysicsObjects() to load collision geometry
- Sets up physics object manager and random number generator

#### Step 2: Load Physics Assets
**Function:** `loadPhysicsObjects()`
**Location:** `src/mgr.cpp`
**Purpose:** Load and configure collision geometry for physics simulation

**Details:**
- Loads OBJ files: cube.obj, wall.obj, agent.obj
- Processes meshes into optimized collision hulls
- Configures per-object physics properties (mass, friction)
- Constrains agent rotation to Z-axis only

#### Step 3: Load Render Assets
**Function:** `loadRenderObjects()`
**Location:** `src/mgr.cpp`
**Purpose:** Load visual assets and configure rendering pipeline

**Details:**
- Loads visual meshes from OBJ files
- Defines materials with colors
- Assigns materials to mesh parts
- Configures scene lighting parameters

#### Step 4: Create Executor
**Function:** `TaskGraphExecutor` or `MWCudaExecutor` constructor
**Location:** Madrona framework
**Purpose:** Create execution engine for simulation

**Details:**
- Configures parallel world count
- Sets up exported buffers for Python access
- Builds task graphs for system execution
- GPU mode: Specifies source files for JIT compilation

#### Step 5: Get Export Pointers
**Function:** `getExported()`
**Location:** `src/mgr.cpp`
**Purpose:** Retrieve pointers to exported component data

**Details:**
- Maps ExportID enum values to component data
- Provides Python access to tensors
- Exports: Reset, Action, Reward, Done, SelfObservation, StepsRemaining

#### Step 6: Initial World Setup
**Function:** `Manager::Manager()` Constructor
**Location:** `src/mgr.cpp`
**Purpose:** Complete manager initialization and prepare worlds

**Details:**
- Calls Impl::init() to set up internal state
- Forces initial reset via triggerReset() for all worlds
- Executes one simulation step to populate observations
- Handles CPU/CUDA memory operations appropriately

### Phase 2: Executor Initialization Phase
Executor construction triggers ECS registration and world creation.

#### Step 1: Base Setup
**Function:** `ThreadPoolExecutor` constructor
**Location:** Madrona framework
**Purpose:** Initialize base executor infrastructure

**Details:**
- Initializes thread pool for parallel execution
- Obtains ECS registry via getECSRegistry()

#### Step 2: ECS Registration
**Function:** `Sim::registerTypes()`
**Location:** `src/sim.cpp`
**Purpose:** Register all ECS types and configure exports

**Details:**
- Registers base and physics system types
- Registers rendering types (if enabled)
- Core components: Action, Reward, Done, CollisionDeath, TerminationReason
- Game-specific components: SelfObservation, CompassObservation, Lidar, Progress, StepsTaken, EntityType, DoneOnCollide
- Singletons: WorldReset, LidarVisControl, LevelState, CompiledLevel
- Archetypes: Agent, PhysicsEntity, RenderOnlyEntity, LidarRayEntity
- Exports components for Python access via ExportID enum

#### Step 3: World Construction
**Function:** World constructor loop
**Location:** Executor implementation
**Purpose:** Create simulation world instances

**Details:**
- Creates per-world contexts
- Initializes task graph managers
- Constructs WorldT objects

#### Step 4: Task Graph Setup
**Function:** `Sim::setupTasks()`
**Location:** `src/sim.cpp`
**Purpose:** Build static execution graph for systems

**Details:**
- **movementSystem**: Converts discrete actions to forces/torques
- **broadphase setup**: Builds BVH for collision detection
- **Physics substeps loop**: Runs 4 physics substeps
  - Rigid body integration (substepRigidBodies)
  - Narrowphase collision detection
  - **agentCollisionSystem**: Detects agent-object collisions
  - Position/velocity constraint solving (XPBD solver)
- **agentZeroVelSystem**: Zeros agent velocities for control
- **stepTrackerSystem**: Tracks episode steps and sets done flag
- **rewardSystem**: Computes incremental forward progress rewards
- **resetSystem**: Conditionally resets world on episode completion
- **initProgressAfterReset**: Initializes progress tracking after reset
- **collectObservationsSystem**: Normalizes agent observations
- **compassSystem**: Computes compass one-hot encoding
- **lidarSystem**: Traces 128 lidar rays with GPU optimization

#### Step 5: Export Initialization
**Function:** `initExport()`
**Location:** Executor implementation
**Purpose:** Initialize exported component memory

**Details:**
- Triggers copyOutExportedColumns()
- Commits physical pages as needed
- Enables getExported() calls

### Phase 3: Per-World Sim Construction Phase
Each world instance is initialized with entities and level layout.

#### Step 1: Calculate Entity Limits
**Function:** `Sim::Sim()` Constructor
**Location:** `src/sim.cpp`
**Purpose:** Initialize world instance with entity limits

**Details:**
- **Initializes CompiledLevel singleton** from world_init data - this is THE critical data structure
- Initializes LidarVisControl singleton (disabled by default)
- **Gets max_total_entities from CompiledLevel** - used for BVH memory allocation
- Initializes physics system with:
  - deltaT = 0.04, numPhysicsSubsteps = 4
  - Gravity = -9.8 * up vector
  - **Max entities from CompiledLevel.max_entities** - critical for broadphase sizing
- Stores initRandKey, autoReset, customVerticalFov settings
- Initializes rendering system if enabled
- Calls createPersistentEntities()
- Calls initWorld()

#### Step 2: Create Persistent Entities
**Function:** `createPersistentEntities()`
**Location:** `src/level_gen.cpp`
**Purpose:** Create entities that persist across episodes

**Details:**
- **createFloorPlane()**: Creates static floor at origin
- **createOriginMarkerGizmo()**: Creates XYZ axis markers for visual reference
- **createLidarRayEntities()**: Creates 128 ray entities per agent for lidar visualization
- **createAgentEntities()**: Creates agent entities with render views
- **Persistent level tiles from CompiledLevel**:
  - Iterates through CompiledLevel.tile_persistent[] array
  - Creates entities for tiles marked as persistent
  - Stores in persistentLevelEntities[] array
  - Physics setup deferred to resetPersistentEntities()

#### Step 3: Initialize World
**Function:** `initWorld()`
**Location:** `src/sim.cpp`
**Purpose:** Prepare world for first/new episode

**Details:**
- **PhysicsSystem::reset()**: Clears collision pairs and physics state
- Creates new RNG state using episode counter and world ID
- **generateWorld()**: Orchestrates world generation using CompiledLevel data
  - **resetPersistentEntities()**: Sets up persistent tiles from CompiledLevel
    - Reads tile positions from CompiledLevel.tile_x/y/z[]
    - Applies scales from CompiledLevel.tile_scale_x/y/z[]
    - Sets physics properties from tile_response_type[] and tile_entity_type[]
    - Configures collision behavior from tile_done_on_collide[]
  - **resetAgentPhysics()**: Places agents at spawn positions
    - Uses CompiledLevel.spawn_x/y[] for positions
    - Uses CompiledLevel.spawn_facing[] for initial rotations
    - Falls back to default positions if spawn index exceeds num_spawns
  - **generateLevel()**: Creates non-persistent tiles
    - Iterates through all CompiledLevel.num_tiles
    - Skips persistent tiles (already created)
    - Applies randomization from tile_rand_* arrays
    - Creates physics or render-only entities based on tile_render_only[]

### Phase 4: First Step Execution Phase
Initial simulation step populates observations for Python. This step is critical because it ensures Python can observe the initial world state before any actions are taken.

#### Step 1: Trigger Resets
**Function:** `triggerReset()`
**Location:** `src/mgr.cpp`
**Purpose:** Mark all worlds for reset

**Details:**
- Sets reset flag in WorldReset buffer
- Handles CPU/CUDA memory operations

#### Step 2: Execute Simulation
**Function:** `step()`
**Location:** `src/mgr.cpp`
**Purpose:** Run one simulation timestep

**Details:**
- Calls impl->run() for execution
- Triggers task graph execution
- Updates all world states

#### Step 3: Render Updates
**Function:** `RenderManager::readECS()`
**Location:** Madrona framework
**Purpose:** Update rendering state from ECS

**Details:**
- Reads entity positions and orientations
- Updates visual representation

#### Step 4: Export Observations
**Function:** Multiple observation systems
**Location:** `src/sim.cpp`
**Purpose:** Populate observation tensors

**Details:**
- **collectObservationsSystem**:
  - **Uses CompiledLevel.world_min/max_x/y/z for normalization**
  - Normalizes agent positions to [0,1] range based on world boundaries
  - Computes normalized progress using world_max_y
- **compassSystem**: Computes 128-dim one-hot direction encoding
- **lidarSystem**: Traces 128 rays in 120-degree arc with GPU optimization
- **rewardSystem**:
  - **Uses CompiledLevel.world_max_y for progress normalization**
  - Computes incremental rewards based on forward movement
- Data automatically written to exported component buffers

## Output

The initialization sequence produces a fully-configured simulation ready for training or evaluation.

### Output Data

#### Direct Outputs
Data explicitly produced by the sequence:

- **Manager Instance**: Interface object for Python interaction
- **Tensor Pointers**: Direct memory access to simulation data via getExported()
- **Initial Observations**: SelfObservation, CompassObservation, Lidar samples
- **World States**: Agent entities at spawn positions, level tiles placed

#### Side Effects
State changes or resources modified:

- **GPU Memory**: Allocated buffers for parallel world execution
- **Physics State**: Broadphase BVH constructed, collision detection initialized
- **Rendering Pipeline**: Agent cameras attached, materials configured
- **Task Graph**: Compiled execution graph with dependency chains

### Exported Tensor Layout

| Export ID | Component/Singleton | Type | Dimensions | Description |
|-----------|-------------------|------|------------|-------------|
| Action | Agent::Action | int32_t* | [num_worlds][num_agents][3] | moveAmount, moveAngle, rotate |
| Reward | Agent::Reward | float* | [num_worlds][num_agents] | Incremental progress reward |
| Done | Agent::Done | uint8_t* | [num_worlds][num_agents] | Episode completion flag |
| TerminationReason | Agent::TerminationReason | int8_t* | [num_worlds][num_agents] | -1=running, 0=steps, 1=goal, 2=collision |
| SelfObservation | Agent::SelfObservation | float* | [num_worlds][num_agents][5] | globalX, globalY, globalZ, maxY, theta |
| CompassObservation | Agent::CompassObservation | float* | [num_worlds][num_agents][128] | One-hot direction encoding |
| Lidar | Agent::Lidar | float* | [num_worlds][num_agents][128] | Normalized depth samples |
| StepsTaken | Agent::StepsTaken | int32_t* | [num_worlds][num_agents] | Current episode step count |
| Progress | Agent::Progress | float* | [num_worlds][num_agents][2] | maxY, initialY positions |
| AgentPosition | Agent::Position | float* | [num_worlds][num_agents][3] | x, y, z coordinates |
| Reset | WorldReset | uint8_t* | [num_worlds] | Reset trigger flag singleton |
| LidarVisControl | LidarVisControl | uint8_t* | [num_worlds] | Lidar visualization enable singleton |
# Reset Sequence

## Overview
The reset sequence handles transitioning between episodes in the escape room environment. Resets can be triggered manually by external code or automatically when episodes complete. The sequence manages both persistent entities (agents, floor, lidar rays) and dynamic entities (level-specific objects) through the CompiledLevel system.

```
resetSystem() [Entry Point - runs every frame]
├─ Check Reset Triggers
│  ├─ Step 1: Check manual reset flag
│  │  └─ WorldReset.reset == 1
│  ├─ Step 2: Check auto-reset condition [if autoReset enabled]
│  │  ├─ Check each agent's Done flag
│  │  └─ If any Done.v == 1, set reset = 1 for next frame
│  └─ Step 3: Episode timeout
│     └─ stepTrackerSystem sets Done.v = 1 at episodeLen
│
├─ Process Deferred Reset [if reset flag = 1]
│  ├─ Step 1: Clear reset flag
│  │  └─ WorldReset.reset = 0
│  └─ Step 2: Execute world cleanup and init
│     ├─ cleanupWorld() ─────────────────┐
│     └─ initWorld() ◄──────────────────┘
│
├─ cleanupWorld() [Destroy dynamic entities]
│  ├─ Step 1: Access LevelState singleton
│  ├─ Step 2: Iterate through Room entities
│  │  ├─ Check if entity is persistent
│  │  └─ Destroy only non-persistent entities
│  └─ Step 3: Clear entity references
│
└─ initWorld() [Create new episode]
   ├─ Step 1: Reset physics system
   │  └─ PhysicsSystem::reset()
   ├─ Step 2: Initialize RNG
   │  └─ New seed from episode counter + worldID
   └─ Step 3: Generate world
      └─ generateWorld() ────────────────┐
         ├─ resetPersistentEntities() ◄──┘
         └─ generateLevel()
```

## Input

### Input Sources
- **Manual Reset Trigger**: External code via Manager::triggerReset() sets WorldReset singleton
- **Auto-Reset Trigger**: Done flags set by various systems (collision, goal reached, timeout)
- **Episode Timer**: stepTrackerSystem increments step counter each frame
- **CompiledLevel Data**: Per-world or shared compiled level data from Manager initialization

### Input Data Format

#### Simple Values
- **WorldReset.reset** (`int32_t`): Manual reset flag (1 = reset, 0 = normal)
- **Done.v** (`float`): Episode completion flag per agent (1.0 = done, 0.0 = active)
- **autoReset** (`bool`): Configuration flag to enable automatic resets
- **episodeLen** (`uint32_t`): Maximum steps per episode (default: 200)
- **curWorldEpisode** (`uint32_t`): Current episode counter for RNG seeding

#### Structured Data
**WorldReset Singleton**
```cpp
struct WorldReset {
    int32_t reset;                    // Manual reset trigger flag
};
```

**Done Component**
```cpp
struct Done {
    float v;                          // Episode completion flag (0.0 or 1.0)
};
```

**CompiledLevel Singleton**
```cpp
**CompiledLevel Singleton** (See Data Structures section for full definition)

```

## Processing

### Processing Pipeline
```
Reset Detection → Deferred Reset Flag → Cleanup Dynamic Entities → Reset Physics → Initialize RNG → Re-register Persistent → Generate Dynamic
```

### Detailed Sequence

#### Phase 1: Reset Detection and Deferral
Occurs every simulation step as part of the task graph

#### Step 1: Check Manual Reset
**Function:** `resetSystem()`
**Location:** `src/sim.cpp:159`
**Purpose:** [REQUIRED_INTERFACE] Check if external code triggered a reset

**Details:**
- Read WorldReset singleton reset flag
- Store in local variable `should_reset`
- Takes priority over auto-reset conditions

#### Step 2: Check Auto-Reset with Deferral
**Function:** `resetSystem()`
**Location:** `src/sim.cpp:162`
**Purpose:** Defer reset by one frame to allow observation of done state

**Details:**
- Only checked if autoReset configuration enabled
- Iterate through all agent Done components
- If any Done.v == 1:
  - Set reset.reset = 1 for NEXT frame
  - Return immediately without resetting
  - This allows Python to observe done=1 state and rewards
- Actual reset happens on following frame

#### Step 3: Check Episode Timeout
**Function:** `stepTrackerSystem()`
**Location:** `src/sim.cpp:581`
**Purpose:** Set Done flag when episode reaches maximum steps

**Details:**
- Increment step counter each frame
- Compare against episodeLen constant
- Set Done.v = 1.0 when limit reached
- Set termination_reason.code = 0 (episode_steps_reached)

### Phase 2: World Cleanup
Triggered when reset flag was set in previous frame

#### Step 1: Clear Reset Flag
**Function:** `resetSystem()`
**Location:** `src/sim.cpp:177`
**Purpose:** Prevent duplicate resets

**Details:**
- Set WorldReset.reset = 0
- Ensures single reset per trigger

#### Step 2: Destroy Dynamic Entities
**Function:** `cleanupWorld()`
**Location:** `src/sim.cpp:111`
**Purpose:** [GAME_SPECIFIC] Remove all non-persistent entities

**Details:**
- Access LevelState singleton for room entity list
- Iterate through all entities in room.entities array
- For each entity, check against persistentLevelEntities list
- Only destroy entities NOT in persistent list
- Call ctx.destroyRenderableEntity() for dynamic entities
- Persistent entities remain untouched

### Phase 3: World Initialization
Executes immediately after cleanup

#### Step 1: Reset Physics System
**Function:** `PhysicsSystem::reset()`
**Location:** `src/sim.cpp:141`
**Purpose:** [BOILERPLATE] Clear all physics state

**Details:**
- Clear collision pair cache
- Reset broad phase BVH
- Clear all constraints
- Required before entity re-registration

#### Step 2: Initialize RNG
**Function:** `initWorld()`
**Location:** `src/sim.cpp:144`
**Purpose:** [BOILERPLATE] Advance PRNG deterministically for episode

**Details:**
- Increment curWorldEpisode counter
- Combine episode counter with world ID
- Advance PRNG deterministically using episode counter and world ID (no external entropy)
- Ensures deterministic but varied generation across episodes
- **Determinism Requirement**: Uses only the global seed - no time-based or external randomness

#### Step 3: Generate World
**Function:** `generateWorld()`
**Location:** `src/level_gen.cpp:509`
**Purpose:** [GAME_SPECIFIC] Orchestrate world generation

**Details:**
- Call resetPersistentEntities() first
- Call generateLevel() for dynamic entities
- Uses CompiledLevel singleton data

### Phase 4: Persistent Entity Reset
Part of world generation process

#### Step 1: Re-register Floor and Persistent Tiles
**Function:** `resetPersistentEntities()`
**Location:** `src/level_gen.cpp:326`
**Purpose:** [BOILERPLATE] Re-register persistent entities with physics

**Details:**
- Register floor plane with physics system
- Iterate through persistent level entities (from CompiledLevel)
- For each persistent tile:
  - Set position from tile_x, tile_y, tile_z arrays
  - Set scale from tile_scale_x/y/z arrays
  - Set rotation from tile_rotation array
  - Register with physics if not render-only
  - Set DoneOnCollide component if specified

#### Step 2: Reset Agent Physics and Spawning
**Function:** `resetAgentPhysics()` and `applyRandomSpawnPositions()`
**Location:** `src/level_gen.cpp:97` and `src/level_gen.cpp:548`
**Purpose:** [GAME_SPECIFIC] Reset agent state for new episode

**Details:**
- Access CompiledLevel singleton for spawn data
- For each agent:
  - Register with physics system
  - **Spawn Position Logic**:
    - If `spawn_random = false`: Use fixed spawn position from level data (spawn_x, spawn_y)
    - If `spawn_random = true`: Generate random collision-free position using 3-unit exclusion radius
  - Set facing angle from spawn_facing array
  - Initialize Progress with sentinel values (-999999.0f)
  - Reset velocity, forces, and torques to zero
  - Reset action components to defaults
  - Reset steps_taken to 0
  - Reset done flag to 0
  - Reset collision_death flag to 0
  - Reset reward to 0.0f
  - Initialize compass observation to zeros

**Random Spawn Determinism Requirements** (when `spawn_random = true`):
- **PRNG Only**: Must use ONLY the simulation's PRNG (no std::random_device or time-based seeds)
- **Collision Avoidance**: Agent must not spawn inside obstacles
- **Boundary Constraints**: Agent must spawn within world boundaries
- **Seed Consistency**: Given the same global seed, spawn positions MUST be identical across runs
- **Unbroken Chain**: The PRNG sequence must not be broken or reseeded during episodes

**Replay Spawn Determinism**:
- **Bit-Identical Positions**: Replay runs must produce bit-identical spawn positions to the original
- **No External Randomness**: All spawn randomization must derive from the deterministic PRNG
- **Multiple Replays**: Multiple replay runs of the same recording must produce identical spawn positions
- **PRNG Sequence Integrity**: Spawn randomization must not disturb the deterministic PRNG sequence

### Phase 5: Dynamic Entity Generation
Final phase of reset sequence

#### Step 1: Generate Level from CompiledLevel
**Function:** `generateLevel()`
**Location:** `src/level_gen.cpp:488`
**Purpose:** [GAME_SPECIFIC] Generate dynamic entities from compiled data

**Details:**
- Access CompiledLevel singleton (fatal error if no data)
- Call generateFromCompiled() with level data

#### Step 2: Create Dynamic Entities
**Function:** `generateFromCompiled()`
**Location:** `src/level_gen.cpp:374`
**Purpose:** Create non-persistent entities for this episode

**Details:**
- Iterate through CompiledLevel tile arrays
- Skip persistent tiles (already handled)
- For each non-persistent tile:
  - Apply position randomization (tile_rand_x/y/z)
  - Apply rotation randomization (tile_rand_rot_z)
  - Apply scale randomization (tile_rand_scale_x/y/z)
  - Create entity (PhysicsEntity or RenderOnlyEntity)
  - Set up physics or render-only components
  - Store in LevelState room entities array

## Output

### Output Data

#### Direct Outputs
Data explicitly produced by the reset sequence:

- **Agent Position** (`Position`): From CompiledLevel spawn data (spawn_x, spawn_y)
- **Agent Rotation** (`Rotation`): From CompiledLevel spawn_facing
- **Progress Sentinels** (`Progress`): Set to -999999.0f for later initialization
- **Episode Counter** (`uint32_t`): Incremented for RNG seeding
- **Step Counter** (`uint32_t`): Reset to 0 for new episode
- **Dynamic Entities** (`Entity[]`): Generated from CompiledLevel with randomization

#### Side Effects
State changes or resources modified:

- **Physics System**: Completely reset with new entity registrations
- **RNG State**: New seed for procedural generation
- **Agent Components**: All reset to initial values
- **Persistent Entities**: Re-registered with physics but not destroyed
- **Dynamic Entities**: Old destroyed, new created with randomization
- **Done Flags**: Reset to 0.0 for all agents
- **Termination Reason**: Reset to -1 (unset)
- **Compass Observation**: Zeroed for all 128 buckets
- **Deferred Reset**: One-frame delay for done state observation

### Component State After Reset

| Component | Agent Value After Reset | Source/Calculation |
|-----------|------------------------|-------------------|
| Position | (spawn_x[i], spawn_y[i], 1.0f) | From CompiledLevel.spawn_x/y[] arrays |
| Rotation | Quat::angleAxis(spawn_facing[i], up) | From CompiledLevel.spawn_facing[] |
| Velocity | {Vector3::zero(), Vector3::zero()} | Zeroed |
| Action | {0, 0, numTurnBuckets/2} | Default: stop, forward, no turn |
| Progress | {-999999.0f, -999999.0f} | Sentinel values for later init |
| StepsTaken | 0 | Reset counter |
| Done | 0.0f | Not terminated |
| Reward | 0.0f | No reward on reset |
| CollisionDeath | 0 | No collision |
| TerminationReason | -1 (after physics settles) | Not terminated |
| CompassObservation | All 128 values = 0.0f | Zeroed array |

### World State After Reset

| State Element | Value After Reset | Location |
|---------------|------------------|----------|
| curWorldEpisode | Incremented | Sim::curWorldEpisode |
| RNG | New seed(initRandKey, episode, worldID) | Sim::rng |
| Persistent Entities | Re-registered with physics | persistentLevelEntities[] |
| Dynamic Entities | Newly created from CompiledLevel | LevelState.rooms[0].entities[] |
| Physics BVH | Rebuilt | PhysicsSystem internal |# Simulation Step Sequence

## Overview
The simulation step sequence executes the Entity Component System (ECS) task graph to advance the Madrona Escape Room by one timestep, processing agent actions, physics, collisions, observations, rewards, and episode resets.

```
Manager::step()
├─ Simulation Execution Phase
│  ├─ Step 1: Polymorphic dispatch (CPU/GPU)
│  │  ├─ CPU: ThreadPoolExecutor::run()
│  │  └─ GPU: MWCudaExecutor::run()
│  └─ Step 2: Task graph execution
│     └─ Executes Sim::setupTasks() graph
│
├─ Task Graph Execution Phase
│  ├─ Movement & Physics
│  │  ├─ movementSystem
│  │  ├─ BVH broadphase setup
│  │  └─ Physics substeps (x4)
│  │     ├─ Rigid body integration
│  │     ├─ Narrowphase collision
│  │     ├─ agentCollisionSystem ◄─── Collision detection
│  │     └─ Constraint solving
│  ├─ Post-Physics
│  │  ├─ agentZeroVelSystem
│  │  ├─ stepTrackerSystem
│  │  └─ rewardSystem
│  ├─ Episode Management
│  │  ├─ resetSystem
│  │  └─ Post-reset BVH rebuild [if reset]
│  └─ Observations
│     ├─ initProgressAfterReset
│     ├─ collectObservationsSystem
│     ├─ compassSystem
│     └─ lidarSystem (128 parallel rays)
│
└─ Render Update Phase [if enabled]
   ├─ Step 1: RenderManager::readECS()
   └─ Step 2: RenderManager::batchRender()
```

## Input

### Input Sources
- **Action Tensors**: Agent discrete actions from policy via `mgr.actionTensor()`
- **Reset Flags**: Episode reset triggers via `mgr.resetTensor()`
- **CompiledLevel**: Level geometry, spawn positions, world boundaries from singleton
- **Physics State**: Entity positions, velocities, collision contacts from ECS

### Input Data Format

#### Simple Values
- **step_idx** (`int32_t`): Current simulation step within episode (0-199)
- **world_id** (`uint32_t`): Parallel world identifier
- **auto_reset** (`bool`): Whether to automatically reset on episode completion

#### Structured Data
**Action Structure**
```cpp
**Action Structure** (See Data Structures section)

```

**CompiledLevel Singleton** (Critical for world generation)
```cpp
**CompiledLevel Singleton** (Critical for world generation - see Data Structures section)

```

## Processing

### Processing Pipeline
```
Actions → Forces → Physics → Collisions → Rewards → Reset Check → Observations → Tensor Export
```

### Detailed Sequence

#### Phase 1: Movement and Physics
Converts discrete actions to physics forces and simulates dynamics

#### Step 1: Movement System
**Function:** `movementSystem()`
**Location:** `src/sim.cpp:186`
**Purpose:** Convert discrete actions to forces and torques

**Details:**
- Move force: 0-1000N mapped from 4 buckets
- Move angle: 8 directions at 45° increments
- Turn torque: ±320 Nm from 5 buckets
- Forces applied in agent's local coordinate frame
- Rotation uses negative sign to compensate for encoding

#### Step 2: Broadphase Setup
**Function:** `PhysicsSystem::setupBroadphaseTasks()`
**Location:** `src/sim.cpp:630`
**Purpose:** Build BVH acceleration structure for collision detection

**Details:**
- Constructs spatial hierarchy for efficient ray tracing
- Updates entity bounding boxes
- Required for both collision detection and lidar

#### Step 3: Physics Substeps Loop
**Function:** Loop at `src/sim.cpp:647`
**Purpose:** Run 4 XPBD physics substeps for stability

**Substep Components:**
1. **Rigid Body Integration** (`substepRigidBodies`) - Apply forces, update positions
2. **Narrowphase Detection** - Find collision contacts between entities
3. **Collision Detection** (`agentCollisionSystem`) at `src/sim.cpp:669` - Check for fatal collisions
4. **Position Solving** (`solvePositions`) - Resolve penetrations
5. **Velocity Update** (`setVelocities`) - Compute post-collision velocities
6. **Velocity Solving** (`solveVelocities`) - Apply friction and restitution

### Phase 2: Post-Physics Processing
Handles agent control, episode tracking, and rewards

#### Step 1: Zero Velocity System
**Function:** `agentZeroVelSystem()`
**Location:** `src/sim.cpp:224`
**Purpose:** Zero agent velocities for better control

**Details:**
- Zeros X and Y linear velocity
- Clamps Z velocity to <= 0 (allow falling)
- Zeros all angular velocity
- Prevents sliding and drift

#### Step 2: Step Tracker System
**Function:** `stepTrackerSystem()`
**Location:** `src/sim.cpp:581`
**Purpose:** Track episode steps and trigger timeout

**Details:**
- Increments `StepsTaken.t` counter
- Sets `done=1` at step 200
- Sets `termination_code=0` for step limit
- Counter resets to 0 on episode reset

#### Step 3: Reward System
**Function:** `rewardSystem()`
**Location:** `src/sim.cpp`
**Purpose:** Calculate completion-based rewards when agent reaches goal

**Details:**
- Implements completion-only reward system
- Tracks agent progress toward level goal
- Handles collision death penalty override

### Phase 3: Episode Management
Handles episode resets and world regeneration

#### Step 1: Reset System
**Function:** `resetSystem()`
**Location:** `src/sim.cpp:159`
**Purpose:** Manage episode resets with deferred execution

**Details:**
- Deferred reset: Waits one step after done=1
- Allows Python to observe final state
- Calls `cleanupWorld()` to remove non-persistent entities
- Calls `initWorld()` to regenerate level
- Increments episode counter

#### Step 2: World Regeneration
**Function:** `generateWorld()` via `initWorld()`
**Location:** `src/level_gen.cpp:509`
**Purpose:** Reset world to initial state

**Details:**
- `resetPersistentEntities()`: Re-register persistent tiles with physics
- `resetAgentPhysics()`: Place agents at spawn positions from CompiledLevel
- `generateLevel()`: Create non-persistent tiles with randomization
- Initialize Progress with sentinel values

### Phase 4: Observation Collection
Prepares normalized observations for policy network

#### Step 1: Initialize Progress After Reset
**Function:** `initProgressAfterReset()`
**Location:** `src/sim.cpp:280`
**Purpose:** Initialize progress tracking after physics settles

**Details:**
- Detects sentinel value -999999.0f
- Sets maxY and initialY to current position
- Sets termination_code to -1 (not terminated)
- Ensures valid values for tensor export

#### Step 2: Collect Observations
**Function:** `collectObservationsSystem()`
**Location:** `src/sim.cpp:298`
**Purpose:** Normalize agent state for policy

**Details:**
- Uses CompiledLevel world boundaries for normalization
- globalX/Y/Z: Position normalized to [0,1]
- maxY: Progress normalized by total possible distance
- theta: Rotation normalized to [-1,1]

#### Step 3: Compass System
**Function:** `compassSystem()`
**Location:** `src/sim.cpp:330`
**Purpose:** One-hot encoding of facing direction

**Details:**
- 128 buckets covering 360°
- Formula: `(64 - int(theta/2π * 128)) % 128`
- Single 1.0 value, rest 0.0
- Bucket 64 = forward facing

#### Step 4: Lidar System
**Function:** `lidarSystem()`
**Location:** `src/sim.cpp:364`
**Purpose:** Cast 128 depth rays for spatial awareness

**Details:**
- 120° field of view (-60° to +60°)
- GPU: 128 threads trace rays in parallel
- Max range: 200 units
- Returns normalized depth [0,1]
- Optional visualization of every 8th ray

## Output

### Output Data

#### Direct Outputs
Tensors available after step() returns:

- **Self Observations** (`float[5]`): globalX, globalY, globalZ, maxY, theta
- **Compass** (`float[128]`): One-hot direction encoding
- **Lidar** (`float[128]`): Normalized depth samples
- **Reward** (`float`): Incremental progress reward or penalty
- **Done** (`uint8_t`): Episode completion flag
- **Termination Reason** (`int8_t`): -1=running, 0=steps, 1=goal, 2=collision
- **Steps Taken** (`int32_t`): Current step count
- **Progress** (`float[2]`): maxY and initialY positions

#### Side Effects
State changes from step execution:

- **Entity Positions**: Updated based on physics simulation
- **Collision Contacts**: Resolved with XPBD solver
- **Episode State**: Reset triggered if done=1 and auto_reset=true
- **BVH Structure**: Rebuilt after reset for new level geometry
- **Lidar Ray Entities**: Positioned for visualization if enabled

### Component State After Step

| Component | Updated By System | Final Value Range/Type |
|-----------|------------------|----------------------|
| Position | Physics integration | World coordinates (float x,y,z) |
| Rotation | Physics + agentZeroVelSystem | Quaternion (normalized) |
| Velocity | agentZeroVelSystem | Linear: (0, 0, ≤0), Angular: (0, 0, 0) |
| ExternalForce | movementSystem | 0-1000N in world space |
| ExternalTorque | movementSystem | ±320 Nm around Z axis |
| Reward | rewardSystem | 0.0 to ~0.005 per step, or -0.1 on collision |
| Done | Multiple systems | 0.0 or 1.0 |
| TerminationReason | Multiple systems | -1, 0, 1, or 2 |
| SelfObservation | collectObservationsSystem | Normalized [0,1] for positions |
| CompassObservation | compassSystem | One-hot 128-dim vector |
| Lidar | lidarSystem | 128 values [0,1] normalized depths |
| Progress | rewardSystem | maxY updated if forward progress |
| StepsTaken | stepTrackerSystem | Incremented by 1 |

Note: These component values are directly accessible via the exported tensors listed in the Initialization Sequence section's Exported Tensor Layout table.