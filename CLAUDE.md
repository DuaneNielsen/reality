# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Madrona Escape Room - a high-performance 3D multi-agent reinforcement learning environment built on the Madrona Engine. It implements a cooperative puzzle game where agents must navigate rooms by stepping on buttons or pushing blocks to open doors.

**Tech Stack:**
- C++ (core simulation using Entity Component System pattern)
- Python (PyTorch-based PPO training)
- CMake build system
- CUDA (optional GPU acceleration)

## Code Classification System

The codebase uses a three-tier classification system to help developers understand what needs to be modified:

### [BOILERPLATE]
Pure Madrona framework code that should never be changed. This includes:
- CPU/GPU execution infrastructure
- Memory management systems
- Rendering pipeline setup
- Base class structures

### [REQUIRED_INTERFACE]
Methods and structures that every Madrona environment must implement:
- `loadPhysicsObjects()` - Load collision meshes and configure physics
- `loadRenderObjects()` - Load visual assets and materials
- `triggerReset()` - Reset episode state (for episodic environments)
- `setAction()` - Accept actions from the policy
- Tensor export methods - Define observation/action spaces
- Reset and action buffers - Required for episodic RL

### [GAME_SPECIFIC]
Implementation details unique to this escape room game:
- Action structure fields (moveAmount, moveAngle, rotate, grab)
- Observation tensor types and shapes
- Object types and their physics properties
- Material colors and textures
- Game constants (see Game-Specific Constants section below)

When creating a new environment:
1. Keep all `[BOILERPLATE]` code unchanged
2. Implement all `[REQUIRED_INTERFACE]` methods with your game's logic
3. Replace all `[GAME_SPECIFIC]` code with your game's details

## Game-Specific Constants

The following constants are defined in `src/consts.hpp` and used throughout the codebase:

### Core Game Parameters
- `numAgents` (2) - Number of agents per world
- `numRooms` (2) - Number of rooms in the level
- `maxEntitiesPerRoom` (15) - Maximum entities per room (walls, buttons, blocks)
- `worldWidth` (18) - Width of the world in units
- `worldLength` (10) - Length of the world in units

### Physics Parameters
- `deltaT` (0.04f) - Fixed timestep in seconds
- `numPhysicsSubsteps` (4) - Physics substeps per frame
- `agentSpeed` (8.f) - Agent movement speed
- `agentRotateSpeed` (5.f) - Agent rotation speed in radians/sec

### Rendering Parameters
- `numRows` (64) - Camera view height in pixels
- `numCols` (64) - Camera view width in pixels
- `verticalFOV` (60.f * M_PI / 180.f) - Camera field of view

### RL Parameters
- `episodeLen` (200) - Maximum steps per episode
- `rewardPerLevel` (1.f) - Reward for completing a room

### Enum Definitions (in `src/types.hpp`)
- `ExportID::NumExports` - Total number of exported tensors
- `TaskGraphID::NumTaskGraphs` - Number of task graphs (currently 1: Step)
- `SimObject::NumObjects` - Total number of object types

### Build Configuration (in CMakeLists.txt)
- `MADRONA_ESCAPE_ROOM_SRC_LIST` - List of source files for GPU compilation
- `GPU_HIDESEEK_SRC_LIST` - GPU-specific source file list
- `GPU_HIDESEEK_COMPILE_FLAGS` - GPU compilation flags

## Essential Commands

### Building the Project
```bash
# Initial setup (from repo root)
mkdir build
cd build
/opt/cmake/bin/cmake ..
make -j$(nproc)
cd ..

# Install Python package
pip install -e .

# Alternative headless executable (no visualization)
./build/headless
```

### Running the Simulation
```bash
# Interactive viewer
./build/viewer

# Benchmark performance
python scripts/sim_bench.py --num-worlds 1024 --num-steps 1000 --gpu-id 0
```

### Training
```bash
# Basic CPU training
python scripts/train.py --num-worlds 1024 --num-updates 100 --ckpt-dir build/checkpoints

# Full GPU training with optimizations
python scripts/train.py --num-worlds 8192 --num-updates 5000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints/
```

### Inference
```bash
# Run trained policy
python scripts/infer.py --num-worlds 1 --num-steps 1000 --fp16 --ckpt-path build/checkpoints/5000.pth --action-dump-path build/dumped_actions

# Replay in viewer
./build/viewer 1 --cpu build/dumped_actions
```

### Development
```bash
# Rebuild after C++ changes
cd build && make -j$(nproc) && cd ..

# No linting/testing commands provided - verify with training scripts
```

### Debugging with GDB

**Important**: Always use the MCP GDB server for debugging. The MCP server provides a clean interface for debugging sessions and handles GDB interactions properly.

```bash
# Build with debug symbols (CPU mode recommended for debugging)
mkdir build && cd build
/opt/cmake/bin/cmake -DCMAKE_BUILD_TYPE=Debug -DMADRONA_CUDA_SUPPORT=OFF ..
make -j$(nproc)

# Note: Ubuntu 20.04's default GDB (9.2) doesn't support DWARF 5 used by modern compilers
# If you see "DW_FORM_strx1" errors, upgrade to GDB 16.3+
```

#### Using MCP GDB Server in Claude Code

When debugging in Claude Code, use the following MCP tools:
- `mcp__gdb__gdb_start` - Start a new debugging session
- `mcp__gdb__gdb_load` - Load executable with arguments
- `mcp__gdb__gdb_set_breakpoint` - Set breakpoints using readable names
- `mcp__gdb__gdb_continue` - Continue execution
- `mcp__gdb__gdb_step` - Step through code
- `mcp__gdb__gdb_backtrace` - View call stack
- `mcp__gdb__gdb_terminate` - End debugging session

Example debugging workflow:
```python
# 1. Start GDB session
mcp__gdb__gdb_start(workingDir="/path/to/build")

# 2. Load program
mcp__gdb__gdb_load(sessionId="...", program="./headless", arguments=["CPU", "1", "10"])

# 3. Set breakpoints using readable C++ names
mcp__gdb__gdb_set_breakpoint(sessionId="...", location="madEscape::Manager::Manager")
mcp__gdb__gdb_set_breakpoint(sessionId="...", location="loadPhysicsObjects")

# 4. Run and debug
mcp__gdb__gdb_continue(sessionId="...")
```

#### Key Breakpoint Locations

For debugging initialization issues:
- `main` - Program entry point
- `madEscape::Manager::Manager` - Manager constructor
- `madEscape::Manager::Impl::init` - Core initialization
- `loadPhysicsObjects` - Physics asset loading
- `loadRenderObjects` - Render asset loading (if rendering enabled)
- `madEscape::Sim::Sim` - Per-world simulator construction
- `madEscape::Sim::setupTasks` - Task graph configuration

For debugging simulation issues:
- `madEscape::Sim::step` - Main simulation step
- `movementSystem` - Agent movement processing
- `physicsSystem` - Physics simulation
- `rewardSystem` - Reward calculation
- `resetSystem` - Episode reset logic

## Architecture

### Core Concepts
The simulator uses Madrona's Entity Component System (ECS) pattern:
- **Components**: Data containers (Position, Action, Reward, etc.)
- **Archetypes**: Entity templates grouping components (Agent, Wall, Button, etc.)
- **Systems**: Functions that operate on components (movementSystem, physicsSystem, etc.)
- **Task Graph**: Defines execution order of systems each simulation step

### Key Files
- `src/types.hpp`: All ECS component and archetype definitions
- `src/sim.cpp/hpp`: Core simulation logic and task graph setup
- `src/level_gen.cpp`: Procedural level generation
- `src/mgr.cpp/hpp`: Manager class handling Python/PyTorch integration
- `scripts/train.py`: PPO training entry point
- `scripts/policy.py`: Neural network policy definitions
- `src/consts.hpp`: Game parameters and constants

### System Execution Order
The task graph defines precise system dependencies:
1. **Input**: Action processing
2. **Physics Pipeline**: 
   - `movementSystem` → `broadphase_setup` → `grabSystem` → `physicsSystem` → `agentZeroVel`
3. **Game Logic Pipeline**:
   - `doorOpenSystem` → `rewardSystem` → `doneSystem` → `resetSystem`
4. **Output**: Observation collection

### Data Flow
1. **Actions** (Python) → `Action` components
2. **Forces** → Physics simulation → Position updates
3. **Game state** → Logic systems → Rewards/Done flags
4. **World state** → Observation systems → PyTorch tensors

### Python Integration
- Manager class exports PyTorch tensors directly mapped to ECS components
- Zero-copy data transfer between simulation and training
- Tensors shaped as (num_worlds × agents_per_world, ...)
- Components marked with `exportColumn` become accessible from Python

### GPU Optimization Patterns
- **Warp-level dispatch**: Lidar uses 32 threads per agent
- **Entity sorting**: Entities grouped by world ID for coalesced access
- **Conditional compilation**: `#ifdef MADRONA_GPU_MODE` for GPU-specific code
- **Memory layout**: Components packed for cache efficiency

### Observation Space Details
- **Egocentric coordinates**: All positions relative to agent
- **Normalization**: Distances by world size, angles by π
- **Fixed arrays**: Padded to `maxEntitiesPerRoom`
- **Lidar**: 30 samples in circle around agent

### Modifying the Simulator

**Important**: When adding entities, update `max_total_entities` calculation in `Sim::Sim()` to ensure the BVH has sufficient space. The physics system currently requires knowing the upper bound at initialization.

1. **Adding new observations**: 
   - Add component to `src/types.hpp`
   - Update `Agent` archetype
   - Populate in `collectObservationSystem`
   - Export in `src/mgr.cpp` with `exportColumn`

2. **Changing game logic**:
   - Modify systems in `src/sim.cpp`
   - Update level generation in `src/level_gen.cpp`
   - Adjust constants in `src/consts.hpp`

3. **Adjusting physics**:
   - Tune parameters in movement/physics systems
   - Modify collision responses
   - Note: Velocities zeroed each frame for controllability

4. **Adding new entity types**:
   - Define archetype in `src/types.hpp`
   - Add to `SimArchetypes` enum
   - Update observation collection if visible to agents

### Performance Considerations
- Batch simulation across thousands of worlds simultaneously
- GPU backend provides massive speedup for large batches (8192+ worlds)
- Use `--profile-report` flag to identify bottlenecks
- Adjust `--num-worlds` based on available GPU memory
- Fixed timestep: 0.04s with 4 physics substeps
- Component access patterns critical for cache performance

## Initialization Process

### Manager Creation Flow
The Manager constructor performs crucial initialization:
```cpp
Manager::Manager(const Config &cfg) {
    // 1. Initialize implementation via Impl::init()
    // 2. Force reset all worlds via triggerReset()
    // 3. Execute one step via step()
}
```

### Execution Mode Initialization

#### CPU Mode:
1. Creates `PhysicsLoader` for CPU execution
2. Calls `loadPhysicsObjects()` to load collision meshes
3. Initializes `ThreadPoolExecutor` with:
   - Auto-detected worker threads (0 = num CPU cores)
   - Exported component memory allocation via `getExported()`
   - Per-world initialization data
4. Maps exported buffer pointers

#### CUDA Mode:
1. Calls `MWCudaExecutor::initCUDA()` for GPU context
2. Creates GPU-based `PhysicsLoader`
3. Calls `loadPhysicsObjects()` to load collision meshes
4. Initializes `MWCudaExecutor` with:
   - JIT compilation of GPU kernels
   - Device memory allocation
   - CUDA graph optimization via `buildLaunchGraphAllTaskGraphs()`
5. Maps device pointers via `getExported()`

### Asset Loading

#### Physics Assets (`loadPhysicsObjects()`):
- **Collision Mesh Loading**: Calls `AssetImporter::importFromDisk()` for OBJ files
- **Rigid Body Processing Pipeline**:
  1. Import raw meshes as convex hulls
  2. Process with `RigidBodyAssets::processRigidBodyAssets()`:
     - Optimizes convex hulls for collision detection
     - Computes bounding volumes and centroids
     - Calculates mass properties (center of mass, inertia tensor)
     - Builds collision primitives (hulls, planes)
     - Allocates contiguous memory block for cache efficiency
  3. Configure physics properties via `setupHull()`:
     - Movable objects: Small inverse mass values
     - Static objects: Zero inverse mass
     - Controlled entities: Unit mass with rotation constraints
     - Friction coefficients: μs=0.5, μd=0.5-0.75
  4. Load processed data via `PhysicsLoader::loadRigidBodies()`

#### Render Assets (`loadRenderObjects()`):
- **Meshes**: Calls `AssetImporter::importFromDisk()` for visual assets
- **Materials**: Configured with RGB values and texture indices
- **Textures**: Loaded via `ImageImporter::importImages()`
- **Lighting**: Set via `RenderManager::configureLighting()`
- **Final Load**: `RenderManager::loadObjects()` uploads to GPU

### Memory Layout
1. **World Data**: Array of `Sim` instances
2. **Exported Tensors**:
   - Actions: `[numWorlds × numAgents × 4]`
   - Rewards: `[numWorlds × numAgents × 1]`
   - Multiple observation tensors
3. **Physics Data**: Collision geometry, rigid body metadata
4. **Render Buffers**: GPU memory for meshes, textures, outputs

## Initialization Sequence

### 1. **Manager Creation Phase** (`Manager::Manager()` → `Impl::init()`)
The Manager acts as the interface between Python and the simulation. During construction:

**Manager Creation - Step 1: Simulation Configuration**
```cpp
// In Manager::Impl::init() - src/mgr.cpp:443
Sim::Config sim_cfg;
sim_cfg.autoReset = mgr_cfg.autoReset;  // [REQUIRED_INTERFACE] Enable automatic episode restart
sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);  // [BOILERPLATE] Initialize RNG

switch (mgr_cfg.execMode) {
case ExecMode::CUDA: {
    // GPU-specific configuration...
    break;
}
case ExecMode::CPU: {
    // CPU-specific configuration...
    // Creates PhysicsLoader for CPU execution
    PhysicsLoader phys_loader(ExecMode::CPU, 10);  // 10 = max collision objects
    
    // Load physics assets
    loadPhysicsObjects(phys_loader);
    
    // Pass physics object manager to sim config
    sim_cfg.rigidBodyObjMgr = phys_loader.getObjectManager();
    break;
}
}
```

**Manager Creation - Step 2: Load Physics Assets** (`loadPhysicsObjects()`)
```cpp
// 1. Load collision meshes from OBJ files
AssetImporter::importFromDisk({
    "cube_collision.obj",    // Pushable blocks
    "wall_collision.obj",    // Static walls and doors
    "agent_collision_simplified.obj",  // Agent collision shape
});

// 2. Process meshes into collision hulls
RigidBodyAssets::processRigidBodyAssets() performs:
- Convex hull optimization
- Bounding volume calculation
- Mass and inertia tensor computation

// 3. Configure physics properties per object type
setupHull(SimObject::Cube, 0.075f, {.muS = 0.5f, .muD = 0.75f});  // Pushable
setupHull(SimObject::Wall, 0.f, {.muS = 0.5f, .muD = 0.5f});      // Static
setupHull(SimObject::Agent, 1.f, {.muS = 0.5f, .muD = 0.5f});     // Controlled

// 4. Special handling: Constrain agent rotation to Z-axis only
rigid_body_assets.metadatas[Agent].mass.invInertiaTensor.x = 0.f;  // No X rotation
rigid_body_assets.metadatas[Agent].mass.invInertiaTensor.y = 0.f;  // No Y rotation

// 5. Pass physics object manager to sim config
sim_cfg.rigidBodyObjMgr = phys_loader.getObjectManager();
```

**Manager Creation - Step 3: Load Render Assets** (`loadRenderObjects()` - if rendering enabled)
```cpp
// 1. Load visual meshes from OBJ files
AssetImporter::importFromDisk({
    "cube_render.obj",    // Visual cube mesh
    "wall_render.obj",    // Wall/door mesh
    "agent_render.obj",   // Multi-part agent mesh
});

// 2. Define materials (colors, textures)
materials[0] = { rgb(191, 108, 10), ...};  // Brown cube
materials[5] = { rgb(230, 20, 20), ...};   // Red door
materials[6] = { rgb(230, 230, 20), ...};  // Yellow button

// 3. Assign materials to mesh parts
render_assets->objects[SimObject::Agent].meshes[0].materialIDX = 2;  // Body
render_assets->objects[SimObject::Agent].meshes[1].materialIDX = 3;  // Eyes

// 4. Load textures
ImageImporter::importImages({"green_grid.png", "smile.png"});

// 5. Configure scene lighting
render_mgr.configureLighting({direction, color, intensity});
```

**Manager Creation - Step 4: Create Executor**
```cpp
// For CPU mode - src/mgr.cpp:525-540
TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit> cpu_exec {
    ThreadPoolExecutor::Config {
        .numWorlds = (uint32_t)mgr_cfg.numWorlds,
        .numExportedBuffers = (uint32_t)ExportID::NumExports,  // [GAME_SPECIFIC]
    },
    sim_cfg,
    {},  // Per-world init data
    (uint32_t)TaskGraphID::NumTaskGraphs  // [GAME_SPECIFIC]
};

// For CUDA mode - src/mgr.cpp:461-480
MWCudaExecutor gpu_exec({
    .worldDataSize = sizeof(Sim),
    .worldDataAlignment = alignof(Sim),
    .numWorldDataSlots = (uint32_t)mgr_cfg.numWorlds,
    .numTaskGraphs = (uint32_t)TaskGraphID::NumTaskGraphs,  // [GAME_SPECIFIC]
    .numExportedBuffers = (uint32_t)ExportID::NumExports,    // [GAME_SPECIFIC]
}, {
    { MADRONA_ESCAPE_ROOM_SRC_LIST },  // [GAME_SPECIFIC] Source files
}, sim_cfg, {});
```

**Manager Creation - Step 5: Get Export Pointers**
```cpp
// In Manager::Impl::init() - src/mgr.cpp:542-544 (CPU) or 483-485 (CUDA)
WorldReset *reset_buffer = (WorldReset *)cpu_exec.getExported(
    (uint32_t)ExportID::Reset);
    
Action *action_buffer = (Action *)cpu_exec.getExported(
    (uint32_t)ExportID::Action);

// These are game-specific exports defined in types.hpp:
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    SelfObservation,
    PartnerObservations,
    RoomEntityObservations,
    DoorObservation,
    Lidar,
    StepsRemaining,
    NumExports,  // Must be last
};
```

**Manager Creation - Step 6: Initial World Setup**
```cpp
// In Manager::Manager() constructor - src/mgr.cpp:576-592
Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))  // Calls Impl::init() to set up everything
{
    // [REQUIRED_INTERFACE] Force initial reset for all worlds
    // This ensures all worlds start in a valid state
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }

    // [BOILERPLATE] Execute one step to populate initial observations
    // This is required so Python can see the initial state
    step();
}

// triggerReset implementation - src/mgr.cpp:762-778
void Manager::triggerReset(int32_t world_idx)
{
    // [GAME_SPECIFIC] Set reset flag to trigger world regeneration
    WorldReset reset {
        1,  // reset flag
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *reset_ptr = reset;
    }
}
```
   
### 2. **Executor Initialization Phase** (inside TaskGraphExecutor/MWCudaExecutor constructor)

**Executor Init - Step 1: Base Setup**
- Base class `ThreadPoolExecutor` constructor runs first
- Call `getECSRegistry()` to obtain the ECS registry

**Executor Init - Step 2: ECS Registration** (`Sim::registerTypes()`)
```cpp
// In Sim::registerTypes() - src/sim.cpp:40-93
// [GAME_SPECIFIC] Register all game components
registry.registerComponent<Action>();
registry.registerComponent<Reward>();
registry.registerComponent<Done>();
// ... register all other game-specific component types

// [GAME_SPECIFIC] Register game archetypes (entity templates)
registry.registerArchetype<Agent>();        // Player-controlled entities
registry.registerArchetype<PhysicsEntity>(); // Movable objects
// ... register other entity types

// [GAME_SPECIFIC] Export components for Python access
// Pattern: registry.exportColumn<Archetype, Component>(ExportID)
registry.exportSingleton<WorldReset>((uint32_t)ExportID::Reset);
registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
// ... export all components that Python needs to access

// Note: Export IDs must match the ExportID enum in types.hpp
```
- **Memory allocation**: Virtual address space (1B × component size) reserved per exported component

**Executor Init - Step 3: World Construction**
- Create per-world contexts and task graph managers
- Construct world data instances (`WorldT` objects)

**Executor Init - Step 4: Task Graph Setup** (`Sim::setupTasks()`)
```cpp
// In Sim::setupTasks() - src/sim.cpp:574-720
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr,
                     const Config &cfg)
{
    TaskGraphBuilder &builder = taskgraph_mgr.init(TaskGraphID::Step);

    // ===== Phase 1: Input Processing =====
    // Convert actions from policy into forces/torques for physics
    // This must run first to prepare inputs for the physics simulation
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
        Action,           // Input from policy
        Rotation,         // Current orientation
        ExternalForce,    // Output: forces for physics
        ExternalTorque    // Output: torques for physics
    >>({/* no dependencies - runs first */});

    // ===== Phase 2: Pre-Physics Updates =====
    // Update kinematically-controlled objects (doors, platforms, etc.)
    // These objects move based on game state, not physics
    auto set_door_pos = builder.addToGraph<ParallelForNode<Engine,
        setDoorPositionSystem,
        Position,         // Output: updated position
        OpenState         // Input: whether door is open
    >>({move_sys});       // Must run after input processing

    // ===== Phase 3: Spatial Structure Build =====
    // Build BVH for efficient collision detection and raycasting
    // Must run after all position updates, before any spatial queries
    auto broadphase_setup = PhysicsSystem::setupBroadphaseTasks(
        builder, {set_door_pos});

    // ===== Phase 4: Spatial Query Systems =====
    // Systems that need raycasting or proximity queries
    // Example: grab system uses raycast to find objects to pick up
    auto grab_sys = builder.addToGraph<ParallelForNode<Engine,
        grabSystem,
        Entity,           // Self reference
        Position,         // Current position
        Rotation,         // Facing direction for raycast
        Action,           // Grab button state
        GrabState         // Output: what we're holding
    >>({broadphase_setup}); // Needs BVH for raycasting

    // ===== Phase 5: Physics Simulation =====
    // Run the actual physics simulation
    // This is a monolithic step - no other systems can run during it
    auto phys_sys = PhysicsSystem::setupPhysicsSimulationTasks(
        builder, cfg.rigidBodyObjMgr, {grab_sys});
    
    // Post-physics cleanup (clear temporary collision data)
    auto phys_cleanup = PhysicsSystem::setupCleanupTasks(
        builder, {phys_sys});

    // ===== Phase 6: Post-Physics Corrections =====
    // Zero velocities for direct control (common in RL environments)
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem,
        Velocity,         // Output: zeroed velocity
        Action            // Only for controlled agents
    >>({phys_cleanup});   // Must wait for physics to complete

    // ===== Phase 7: Game Logic =====
    // Check win conditions, calculate rewards, manage game state
    // These systems read physics results but don't modify positions
    
    auto door_open_sys = builder.addToGraph<ParallelForNode<Engine,
        doorOpenSystem,
        OpenState,        // Output: is door open?
        DoorProperties    // Input: which buttons open this door
    >>({agent_zero_vel}); // Runs after physics is finalized

    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
        rewardSystem,
        Position,         // Input: where is agent?
        Progress,         // Input/Output: level progress
        Reward            // Output: reward signal
    >>({door_open_sys});

    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        doneSystem,
        Progress,         // Input: did we complete level?
        StepsRemaining,   // Input: time limit check
        Done              // Output: episode over flag
    >>({reward_sys});

    // ===== Phase 8: Episode Management =====
    // Handle resets when episodes end
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
        WorldReset        // Input/Output: reset request/acknowledgment
    >>({done_sys});

    // ===== Phase 9: Post-Reset Spatial Rebuild =====
    // Rebuild BVH after reset (world geometry changed)
    // On GPU, this runs every frame (can't conditionally execute)
    auto post_reset_broadphase = PhysicsSystem::setupBroadphaseTasks(
        builder, {reset_sys});

    // ===== Phase 10: Observation Collection =====
    // Gather all observations for the policy
    // MUST run last after all state updates
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
        Position,                // Input: agent position
        Rotation,                // Input: agent orientation
        Progress,                // Input: game state
        GrabState,               // Input: what we're holding
        SelfObservation,         // Output: egocentric obs
        PartnerObservations,     // Output: teammate info
        RoomEntityObservations,  // Output: object positions
        DoorObservation          // Output: door states
    >>({post_reset_broadphase}); // Must run after everything
}
```

**Executor Init - Step 5: Export Initialization**
- Call `initExport()` which triggers initial `copyOutExportedColumns()`
- **Memory allocation**: Physical pages committed as needed during copy
- **Now `getExported()` can be called** to retrieve pointers to exported data

### 3. **Per-World Sim Construction Phase** (`Sim::Sim()` constructor)

**World Init - Step 1: Calculate Entity Limits**
```cpp
// In Sim::Sim() constructor - src/sim.cpp:729-761
Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    // [GAME_SPECIFIC] Calculate max entities based on game parameters
    constexpr CountT max_total_entities = consts::numAgents +
        consts::numRooms * (consts::maxEntitiesPerRoom + 3) +  // +3 for doors/walls per room
        4;  // side walls + floor

    // [BOILERPLATE] Initialize physics with entity limit
    PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities);

    // [BOILERPLATE] Store configuration
    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;

    // ... rendering initialization ...

    // [GAME_SPECIFIC] Create persistent entities and generate initial world
    createPersistentEntities(ctx);  // Creates agents, walls, floor
    initWorld(ctx);                 // Generates level layout
}
```

**World Init - Step 2: Create Persistent Entities** (`createPersistentEntities()`)

The `createPersistentEntities` function creates entities that persist across all episodes. These entities are created once during world initialization and are reused/reset for each episode rather than destroyed and recreated.

```cpp
// In createPersistentEntities() - src/level_gen.cpp:76-188
void createPersistentEntities(Engine &ctx)
{
    // [GAME_SPECIFIC] Create floor plane - a static physics entity
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },        // Position at origin
        Quat { 1, 0, 0, 0 },        // No rotation
        SimObject::Plane,           // Physics object type
        EntityType::None,           // Floor plane type never queried
        ResponseType::Static);      // Immovable

    // [GAME_SPECIFIC] Create outer boundary walls
    // Three walls are created (behind, right, left) - front is open
    
    // Behind wall (at y=0)
    ctx.data().borders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[0],
        Vector3 { 0, -consts::wallWidth / 2.f, 0 },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {  // Scale to span entire world width
            consts::worldWidth + consts::wallWidth * 2,
            consts::wallWidth,
            2.f
        });

    // Right and Left walls similarly positioned at world boundaries...

    // [GAME_SPECIFIC] Create agent entities
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] = 
            ctx.makeRenderableEntity<Agent>();

        // [GAME_SPECIFIC] Attach camera view to agent (if rendering enabled)
        if (ctx.data().enableRender) {
            render::RenderingSystem::attachEntityToView(ctx,
                agent,
                100.f,              // Far plane distance
                0.001f,             // Near plane distance  
                1.5f * math::up);   // Camera offset from agent
        }

        // [GAME_SPECIFIC] Initialize agent components
        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<GrabState>(agent).constraintEntity = Entity::none();
        ctx.get<EntityType>(agent) = EntityType::Agent;
    }

    // [GAME_SPECIFIC] Populate OtherAgents component
    // Each agent maintains references to all other agents for observations
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity cur_agent = ctx.data().agents[i];
        OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
        
        CountT out_idx = 0;
        for (CountT j = 0; j < consts::numAgents; j++) {
            if (i == j) continue;  // Skip self
            
            Entity other_agent = ctx.data().agents[j];
            other_agents.e[out_idx++] = other_agent;
        }
    }
}
```

**Key Points about Persistent Entities:**
- **Floor and Walls**: Static collision geometry that defines world boundaries
- **Agents**: Player-controlled entities with attached cameras and physics
- **Lifetime**: Created once at startup, persist across all episodes
- **Reset Behavior**: Positions/states reset each episode via `resetPersistentEntities()`
- **Component Initialization**: Only invariant components set here; episode-specific values set during reset

**World Init - Step 3: Initialize World** (`initWorld()`)

The `initWorld` function prepares each new episode through this call sequence:
1. `PhysicsSystem::reset()` - Clears all collision pairs, constraints, and physics state from the previous episode
2. Creates a new RNG state using `rand::split_i()` with the episode counter and world ID
3. `generateWorld()` - Orchestrates the world generation process

The `generateWorld` function then calls:
1. `resetPersistentEntities()` - Resets agents and re-registers all persistent entities
2. `generateLevel()` - Creates the procedural level layout

Inside `resetPersistentEntities()`:
- `registerRigidBodyEntity()` is called for each persistent entity (floor, walls, agents) to re-register them with the physics broadphase
- For each agent, the following game-specific state is reset:

```cpp
// [GAME_SPECIFIC] Agent spawn positioning
Vector3 pos {
    randInRangeCentered(ctx, consts::worldWidth / 2.f - 2.5f * consts::agentRadius),
    randBetween(ctx, consts::agentRadius * 1.1f, 2.f),  // Slightly above floor
    0.f
};

// Spread agents left/right alternately
if (i % 2 == 0) {
    pos.x += consts::worldWidth / 4.f;
} else {
    pos.x -= consts::worldWidth / 4.f;
}

// [GAME_SPECIFIC] Random initial rotation (±45 degrees)
ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
    randInRangeCentered(ctx, math::pi / 4.f), math::up);

// [GAME_SPECIFIC] Clear grab state from previous episode
auto &grab_state = ctx.get<GrabState>(agent_entity);
if (grab_state.constraintEntity != Entity::none()) {
    ctx.destroyEntity(grab_state.constraintEntity);
    grab_state.constraintEntity = Entity::none();
}

// [GAME_SPECIFIC] Reset gameplay components
ctx.get<Progress>(agent_entity).maxY = pos.y;  // Track forward progress
ctx.get<Action>(agent_entity) = Action {
    .moveAmount = 0,
    .moveAngle = 0,
    .rotate = consts::numTurnBuckets / 2,  // Center rotation bucket
    .grab = 0
};
ctx.get<StepsRemaining>(agent_entity).t = consts::episodeLen;  // Reset timer
```

### 4. **First Step Execution Phase**

**First Step - Step 1: Trigger Resets**
- Call `triggerReset()` for all worlds

**First Step - Step 2: Execute Simulation**
- Call `step()` → `impl->run()` → `gpuExec.run()`

**First Step - Step 3: Render Updates** (if enabled)
- Call `RenderManager::readECS()`

**First Step - Step 4: Export Observations**
- Observations are automatically populated during task graph execution
- The `collectObservationSystem` [REQUIRED_INTERFACE] writes to exported components
- No explicit export call needed - data is directly accessible via tensor methods

### Reset Sequence

The reset sequence handles transitioning between episodes in the escape room environment. Resets can be triggered manually by external code or automatically when episodes complete.

#### **Reset Entry Point** (`resetSystem()` - src/sim.cpp:138)

The `resetSystem` function is the main entry point for episode resets. It's executed every simulation step as part of the task graph.

```cpp
// [REQUIRED_INTERFACE] Reset system - every episodic environment needs this
// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    // [GAME_SPECIFIC] Check manual reset flag
    int32_t should_reset = reset.reset;
    
    // [GAME_SPECIFIC] Check auto-reset condition
    if (ctx.data().autoReset) {
        for (CountT i = 0; i < consts::numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            Done done = ctx.get<Done>(agent);
            if (done.v) {
                should_reset = 1;  // Any agent done triggers reset
            }
        }
    }

    if (should_reset != 0) {
        reset.reset = 0;  // Clear the reset flag
        
        cleanupWorld(ctx);  // Destroy dynamic entities
        initWorld(ctx);     // Create new episode
    }
}
```

**Reset Triggers:**
1. **Manual Reset**: External code sets `WorldReset.reset = 1` via `Manager::triggerReset()`
2. **Auto-Reset**: When any agent's `Done` flag is set (if `autoReset` enabled)
3. **Episode Timeout**: `stepTrackerSystem` sets `Done.v = 1` when steps reach limit

#### **World Cleanup** (`cleanupWorld()` - src/sim.cpp:100)

Destroys all dynamically created entities from the current episode:

```cpp
// [GAME_SPECIFIC] Helper to cleanup world before reset
static inline void cleanupWorld(Engine &ctx)
{
    // Destroy current level entities
    LevelState &level = ctx.singleton<LevelState>();
    for (CountT i = 0; i < consts::numRooms; i++) {
        Room &room = level.rooms[i];
        
        // Destroy room entities (buttons, blocks, etc.)
        for (CountT j = 0; j < consts::maxEntitiesPerRoom; j++) {
            if (room.entities[j] != Entity::none()) {
                ctx.destroyRenderableEntity(room.entities[j]);
            }
        }
        
        // Destroy walls and doors for this room
        ctx.destroyRenderableEntity(room.walls[0]);  // Left wall segment
        ctx.destroyRenderableEntity(room.walls[1]);  // Right wall segment
        ctx.destroyRenderableEntity(room.door);      // Door entity
    }
}
```

**Important**: Persistent entities (agents, floor, outer walls) are NOT destroyed during cleanup.

#### **World Initialization** (`initWorld()` - src/sim.cpp:119)

Prepares the world for a new episode:

```cpp
// [GAME_SPECIFIC] Helper to initialize a new escape room world
static inline void initWorld(Engine &ctx)
{
    // [BOILERPLATE] Always reset physics first
    phys::PhysicsSystem::reset(ctx);  // Clears collision pairs, constraints, BVH
    
    // [BOILERPLATE] Assign a new episode ID and RNG
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        ctx.data().curWorldEpisode++, (uint32_t)ctx.worldID().idx));
    
    // [GAME_SPECIFIC] Generate new world layout
    generateWorld(ctx);  // Defined in src/level_gen.cpp
}
```

**Call Sequence:**
1. `PhysicsSystem::reset()` - [BOILERPLATE] Clears all physics state
2. RNG initialization - [BOILERPLATE] New random seed for this episode
3. `generateWorld()` - [GAME_SPECIFIC] Creates new level layout

#### **World Generation** (`generateWorld()` - src/level_gen.cpp:522)

Orchestrates the world generation process:

```cpp
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);  // Reset agents and re-register entities
    generateLevel(ctx);            // Create new room layout
}
```

#### **Persistent Entity Reset** (`resetPersistentEntities()` - src/level_gen.cpp:193)

Resets entities that persist across episodes:

```cpp
static void resetPersistentEntities(Engine &ctx)
{
    // [BOILERPLATE] Re-register persistent entities with physics
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);
    
    // Re-register boundary walls
    for (CountT i = 0; i < 3; i++) {
        Entity wall_entity = ctx.data().borders[i];
        registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
    }
    
    // [GAME_SPECIFIC] Reset each agent
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity agent_entity = ctx.data().agents[i];
        
        // [BOILERPLATE] Re-register with physics
        registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);
        
        // [GAME_SPECIFIC] Random spawn position near starting wall
        Vector3 pos {
            randInRangeCentered(ctx, 
                consts::worldWidth / 2.f - 2.5f * consts::agentRadius),
            randBetween(ctx, consts::agentRadius * 1.1f, 2.f),  // Slightly above floor
            0.f,
        };
        
        // Spread agents left/right alternately
        if (i % 2 == 0) {
            pos.x += consts::worldWidth / 4.f;
        } else {
            pos.x -= consts::worldWidth / 4.f;
        }
        
        ctx.get<Position>(agent_entity) = pos;
        
        // [GAME_SPECIFIC] Random initial rotation (±45 degrees)
        ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
            randInRangeCentered(ctx, math::pi / 4.f), math::up);
        
        // [GAME_SPECIFIC] Clear grab constraint from previous episode
        auto &grab_state = ctx.get<GrabState>(agent_entity);
        if (grab_state.constraintEntity != Entity::none()) {
            ctx.destroyEntity(grab_state.constraintEntity);
            grab_state.constraintEntity = Entity::none();
        }
        
        // [GAME_SPECIFIC] Reset progress tracking
        ctx.get<Progress>(agent_entity).maxY = pos.y;
        
        // [BOILERPLATE] Clear physics state
        ctx.get<Velocity>(agent_entity) = { Vector3::zero(), Vector3::zero() };
        ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
        ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
        
        // [GAME_SPECIFIC] Reset action and timer
        ctx.get<Action>(agent_entity) = Action {
            .moveAmount = 0,
            .moveAngle = 0,
            .rotate = consts::numTurnBuckets / 2,  // Center bucket
            .grab = 0,
        };
        ctx.get<StepsRemaining>(agent_entity).t = consts::episodeLen;
    }
}
```

#### **Level Generation** (`generateLevel()` - src/level_gen.cpp:486)

Creates the procedural room layout for the new episode:

```cpp
static void generateLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();
    
    // [GAME_SPECIFIC] Generate each room with random type
    for (CountT room_idx = 0; room_idx < consts::numRooms; room_idx++) {
        int room_type_idx = (int)randBetween(ctx, 0, (int)RoomType::NumTypes);
        RoomType room_type = (RoomType)room_type_idx;
        
        makeRoom(ctx, level, room_idx, room_type);
    }
}
```

Each room is generated with:
- Random room type (SingleButton, DoubleButton, Cube)
- End walls with doors at random positions
- Buttons, blocks, and other interactive elements
- All new entities registered with the physics system

#### **Key Points:**

- **Persistent vs Dynamic**: Agents, floor, and outer walls persist; rooms are recreated
- **Physics Reset**: Critical to clear collision state before world generation
- **Entity Registration**: All entities must be re-registered with physics after reset
- **Random Seed**: Each episode gets unique RNG state for procedural generation
- **Component Reset**: Agent components reset to valid initial states
- **Task Graph Integration**: Reset system runs every frame, checking conditions

### Step Sequence

The `Manager::step()` function is the main entry point for advancing the simulation by one timestep. It's called from the main loop after initialization and orchestrates all simulation updates.

#### **Manager::step() Implementation**

The step function executes three main phases:

1. **Simulation Update**
   - Calls `impl_->run()` which polymorphically dispatches to:
     - CPU Mode: `CPUImpl::run()` → `cpuExec.run()` → `ThreadPoolExecutor::run()`
     - CUDA Mode: `CUDAImpl::run()` → `gpuExec.run(stepGraph)` → `MWCudaExecutor::run()`
   - This executes the task graph defined in `Sim::setupTasks()`

2. **Render State Update** (if rendering enabled)
   - Calls `renderMgr->readECS()` to synchronize render state with ECS components
   - Copies transform data (Position, Rotation, Scale) to render buffers
   - Updates instance data for all visible entities

3. **Batch Rendering** (if batch renderer enabled)
   - Calls `renderMgr->batchRender()` to perform actual rendering
   - Renders all worlds in a single batch operation
   - Outputs to configured render targets

**Key Points:**
- **Zero-Copy Design**: Simulation data is directly accessible to Python without copying
- **Polymorphic Execution**: Same interface for CPU and GPU execution modes
- **Optional Rendering**: Render updates only occur when visualization is enabled
- **Synchronous Execution**: Each step completes before returning to caller

**Usage Example:**
```cpp
// Main simulation loop
for (int64_t i = 0; i < num_steps; i++) {
    mgr.step();  // Advance simulation by one timestep
    
    // Python can now read updated observations via tensor methods
    // e.g., mgr.rewardTensor(), mgr.doneTensor(), etc.
}
```

### Key Configuration Parameters
- `execMode`: CPU or CUDA execution
- `gpuID`: Target GPU device
- `numWorlds`: Parallel simulation count
- `randSeed`: RNG initialization
- `autoReset`: Automatic episode restart
- `enableBatchRenderer`: GPU rendering toggle

### Thread/GPU Parallelism
- **CPU**: Thread pool with automatic core detection
- **CUDA**: One thread block per world, warp-level optimizations
- Task graph ensures correct system execution order
- Zero-copy memory mapping for Python integration

## ECS System

### Adding a Component

To add a new component to the Madrona ECS:

1. **Define the Component** in `src/types.hpp`:
   ```cpp
   struct MyNewComponent {
       float value1;
       int32_t value2;
   };
   ```

2. **Register the Component** in `Sim::registerTypes()`:
   ```cpp
   registry.registerComponent<MyNewComponent>();
   ```

3. **Add to Archetype** if needed:
   ```cpp
   struct MyEntity : public madrona::Archetype<
       Position,
       Rotation,
       MyNewComponent  // Add here
   > {};
   ```

4. **Export for Python Access** (optional):
   - In `Sim::registerTypes()`, add export column:
     ```cpp
     registry.exportColumn<MyEntity, MyNewComponent>(
         (uint32_t)ExportID::MyNewComponent);
     ```
   - Add to `ExportID` enum in `src/types.hpp`
   - Map tensor in `src/mgr.cpp`:
     ```cpp
     exported.myNewComponent = gpu_exec.getExported((uint32_t)ExportID::MyNewComponent);
     ```

5. **Initialize Component Values**:
   - Set initial values when creating entities
   - Update in reset systems if component should reset

### Adding a System

To add a new system to process components:

1. **Write the System Function** in `src/sim.cpp`:
   ```cpp
   inline void myNewSystem(Engine &ctx,
                          Position &pos,
                          MyNewComponent &my_comp)
   {
       // System logic here
       my_comp.value1 += pos.x;
   }
   ```

2. **Register System** in `Sim::setupTasks()`:
   ```cpp
   TaskGraphNodeID my_new_sys = builder.addToGraph<ParallelForNode<Engine,
       myNewSystem,
       Position,
       MyNewComponent
   >>({optional_dependencies});
   ```

3. **Define Dependencies**:
   - Systems execute in dependency order
   - Add node ID to dependency array of later systems:
     ```cpp
     TaskGraphNodeID later_sys = builder.addToGraph<...>({
         my_new_sys,  // This system depends on myNewSystem
         other_dep
     });
     ```

4. **Considerations**:
   - **Query Scope**: Systems automatically iterate over all entities with required components
   - **Context Access**: Use `ctx` to access world state, entity references
   - **Performance**: Keep systems focused, avoid random memory access
   - **GPU Compatibility**: Use `#ifdef MADRONA_GPU_MODE` for GPU-specific code
   - **Parallelism**: Systems run in parallel across worlds and entities

### GPU Execution of Systems

Madrona automatically compiles and executes systems on GPU without requiring manual CUDA code:

#### How GPU Compilation Works

1. **Automatic Translation**: System functions written in standard C++ are automatically compiled for GPU via NVRTC (NVIDIA Runtime Compilation)
2. **JIT Compilation**: At runtime, Madrona:
   - Compiles all system functions into PTX code
   - Generates a "megakernel" containing all systems
   - Creates dispatch logic to route execution
3. **Execution Model**:
   - **CPU**: One thread iterates through entities sequentially
   - **GPU**: Multiple threads process entities in parallel per world

#### Supported C++ Features in Systems

**✅ Can Use:**
- Control flow (`if/else`, loops, `switch`)
- Math functions (`fminf`, `sqrt`, `sin`, etc.)
- Local variables and fixed-size arrays
- Function calls to other inline functions
- Component read/write access
- Ternary operators and all arithmetic operations

**❌ Cannot Use:**
- Dynamic memory allocation (`new`, `malloc`)
- STL containers (`std::vector`, `std::map`)
- Virtual functions or RTTI
- Exceptions or `try/catch`
- File I/O or system calls
- Global/static variables
- Recursive functions

#### GPU-Specific Optimizations

1. **Warp-Level Systems** using `CustomParallelForNode`:
   ```cpp
   #ifdef MADRONA_GPU_MODE
   auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
       lidarSystem, 32, 1,  // 32 threads per entity
       Entity, Lidar
   >>({dependencies});
   #endif
   ```

2. **Entity Sorting** for memory coalescing:
   ```cpp
   #ifdef MADRONA_GPU_MODE
   auto sort_agents = queueSortByWorld<Agent>(builder, {deps});
   #endif
   ```

3. **Thread Indexing** in warp-level systems:
   ```cpp
   inline void lidarSystem(Engine &ctx, Entity e, Lidar &lidar) {
   #ifdef MADRONA_GPU_MODE
       int thread_id = threadIdx.x % 32;  // Thread's position in warp
       // Each thread traces a different ray
   #endif
   }
   ```

#### Key GPU Architecture Details

- **Megakernel Design**: All systems run in a single CUDA kernel to minimize launch overhead
- **One Thread Block Per World**: Each simulation world gets dedicated threads
- **Work Stealing**: Dynamic load balancing across thread blocks
- **Zero-Copy Memory**: Direct mapping between GPU memory and Python tensors
- **Compilation Cache**: Compiled kernels cached to avoid recompilation

### Task Graph Setup (setupTasks)

The `setupTasks` function builds a static execution graph defining the order and dependencies of all systems that run each simulation step. Understanding this is crucial for modifying the simulation.

#### Data Flow Phases

1. **Input Processing Phase**
   - Transforms external actions into physics forces/torques
   - Must run first to prepare physics inputs

2. **Pre-Physics State Updates**
   - Updates positions of kinematically-controlled entities (doors)
   - Must complete before spatial structure build

3. **Spatial Structure Build (Phase 1)**
   - Builds BVH (Bounding Volume Hierarchy) for efficient spatial queries
   - Depends on all entity positions being finalized
   - Required by any system doing raycasting or proximity queries

4. **Spatial Query Systems**
   - Systems like grab that need raycasting
   - Must run after BVH build, before physics

5. **Physics Simulation Pipeline**
   - Broadphase collision detection (uses BVH)
   - Narrowphase collision detection  
   - Constraint solving (multiple substeps)
   - Position/velocity integration
   - **Monolithic step** - no other systems can run during physics

6. **Post-Physics Corrections**
   - Modifications to physics output (e.g., zeroing velocities)
   - Must run after physics, before cleanup

7. **Physics Cleanup**
   - Finalizes physics state
   - Clears temporary collision data
   - Must run after all physics modifications

8. **Game Logic Phase**
   - Reward calculation, door logic, episode management
   - Reads physics state but doesn't modify it
   - Safe to run in any order within this phase

9. **Episode Management**
   - Reset detection and world regeneration
   - May invalidate entire world state

10. **Spatial Structure Rebuild (Phase 2)**
    - Required after reset (world geometry may have changed)
    - Runs every frame due to GPU constraints (no conditional execution)

11. **Observation Collection**
    - Gathers all entity states for external consumers
    - Must run last, after all state updates

#### Key Principles

**Physics Data Access Rules:**
- **Pre-physics systems**: Can write positions/forces, no physics data exists yet
- **During physics**: No other systems can execute
- **Post-physics systems**: Can read physics results, limited modification allowed
- **Logic systems**: Read-only access to physics state

**Spatial Query Dependencies:**
- Any system using raycasting must run after BVH build
- Systems modifying positions invalidate the BVH
- Two BVH builds needed: pre-physics and post-reset

**GPU Constraints:**
- No dynamic graph modification - all nodes run every frame
- Conditional logic implemented via no-op execution
- Entity recycling handled separately on GPU backend

**Common Patterns:**
```cpp
// System with dependencies
auto my_sys = builder.addToGraph<ParallelForNode<Engine,
    mySystem,
    Component1,
    Component2
>>({dependency1, dependency2});

// Physics-dependent system  
auto post_phys_sys = builder.addToGraph<...>({phys_cleanup});

// Observation system (runs last)
auto obs_sys = builder.addToGraph<...>({post_reset_broadphase});
```
