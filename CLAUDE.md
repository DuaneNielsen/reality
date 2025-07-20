# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Madrona Escape Room - a high-performance 3D multi-agent reinforcement learning environment built on the Madrona Engine. It implements a cooperative puzzle game where agents must navigate rooms by stepping on buttons or pushing blocks to open doors.

**Tech Stack:**
- C++ (core simulation using Entity Component System pattern)
- Python (PyTorch-based PPO training)
- CMake build system
- CUDA (optional GPU acceleration)

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
   - `doorOpenSystem` → `rewardSystem` → `bonusRewardSystem` → `doneSystem` → `resetSystem`
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

**Manager Creation - Step 1: Boilerplate Setup**
- Creates execution context (CPU threads or CUDA)
- Allocates memory for parallel worlds
- Sets up data export infrastructure

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

**Manager Creation - Step 4: Final Setup**
- Creates executor (CPU/GPU) with loaded assets
- Retrieves pointers to exported buffers (actions, observations)
- Triggers initial reset and steps once to populate observations
   
### 2. **Executor Initialization Phase** (inside TaskGraphExecutor/MWCudaExecutor constructor)

**Executor Init - Step 1: Base Setup**
- Base class `ThreadPoolExecutor` constructor runs first
- Call `getECSRegistry()` to obtain the ECS registry

**Executor Init - Step 2: ECS Registration** (`Sim::registerTypes()`)
- Registers all components with `registry.registerComponent<T>()`
- Registers all archetypes with `registry.registerArchetype<T>()` 
- Exports components for Python access with `registry.exportColumn<T>()`
- **Memory allocation**: Virtual address space (1B × component size) reserved per exported component

**Executor Init - Step 3: World Construction**
- Create per-world contexts and task graph managers
- Construct world data instances (`WorldT` objects)

**Executor Init - Step 4: Task Graph Setup** (`Sim::setupTasks()`)
- Build task graphs defining system execution order
- Configure dependencies between systems

**Executor Init - Step 5: Export Initialization**
- Call `initExport()` which triggers initial `copyOutExportedColumns()`
- **Memory allocation**: Physical pages committed as needed during copy
- **Now `getExported()` can be called** to retrieve pointers to exported data

### 3. **Per-World Sim Construction Phase** (`Sim::Sim()` constructor)

**World Init - Step 1: Calculate Entity Limits**
```cpp
max_total_entities = numAgents + numRooms * (maxEntitiesPerRoom + 3) + 4
// Accounts for: agents + room entities + doors/walls + floor/boundaries
```

**World Init - Step 2: Initialize Subsystems**
- Initialize physics via `PhysicsSystem::init()` with entity count
- Initialize rendering via `RenderingSystem::init()` if enabled

**World Init - Step 3: Create Persistent Entities**
- Call `createPersistentEntities()` for agents, walls, floor

**World Init - Step 4: Generate Initial World**
- Call `initWorld()` → `generateWorld()` → `generateLevel()`

### 4. **First Step Execution Phase**

**First Step - Step 1: Trigger Resets**
- Call `triggerReset()` for all worlds

**First Step - Step 2: Execute Simulation**
- Call `step()` → `impl->run()` → `gpuExec.run()`

**First Step - Step 3: Render Updates** (if enabled)
- Call `RenderManager::readECS()`

**First Step - Step 4: Export Observations**
- Populate initial observations for Python

### Reset Sequence

todo: we need to read through the reset sequence and map out documentation for each function that needs to be called and each important step

### Step Sequence

todo: we need to map out the step sequence and document it, naming each function that needs to be called and each important step

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
