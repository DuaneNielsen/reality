# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Madrona Escape Room - a high-performance 3D multi-agent reinforcement learning environment built on the Madrona Engine. It implements a navigation environment where agents explore and try to maximize their forward progress through the world.

**Tech Stack:**
- C++ (core simulation using Entity Component System pattern)
- Python (PyTorch-based PPO training)
- CMake build system


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

## Python Package Management

**IMPORTANT**: This project uses `uv` for all Python package management. Always use `uv` instead of `pip` or plain `python` commands.

## Essential Commands

### Building the Project
```bash
# Initial setup (from repo root)
mkdir build
cd build
/opt/cmake/bin/cmake ..
make -j$(nproc)
cd ..

# Install Python package (ALWAYS use uv)
uv pip install -e .

# Alternative headless executable (no visualization)
./build/headless
```

### Running the Simulation
```bash
# Interactive viewer
./build/viewer

# Benchmark performance
uv run python scripts/sim_bench.py --num-worlds 1024 --num-steps 1000 --gpu-id 0
```

### Training
```bash
# Basic CPU training
uv run python scripts/train.py --num-worlds 1024 --num-updates 100 --ckpt-dir build/checkpoints

# Full GPU training with optimizations
uv run python scripts/train.py --num-worlds 8192 --num-updates 5000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints/
```

### Inference
```bash
# Run trained policy
uv run python scripts/infer.py --num-worlds 1 --num-steps 1000 --fp16 --ckpt-path build/checkpoints/5000.pth --action-dump-path build/dumped_actions

# Replay in viewer
./build/viewer 1 --cpu build/dumped_actions
```

### Development
```bash
# Rebuild after C++ changes
cd build && make -j$(nproc) && cd ..

# Run tests - ALWAYS run CPU tests first, then GPU tests
# Run all CPU tests (default)
uv run --extra test pytest tests/python/ -v --no-gpu

# Only after CPU tests pass, run GPU tests
uv run --extra test pytest tests/python/ -v -k "gpu"

# Run specific test file
uv run --extra test pytest tests/python/test_bindings.py -v --tb=short

# Run reward system tests
uv run --extra test pytest tests/python/test_reward_system.py -v

# Run tests with action recording (saves to test_recordings/)
uv run --extra test pytest tests/python/test_reward_system.py -v --record-actions

# Run tests with recording and automatic visualization
uv run --extra test pytest tests/python/test_reward_system.py -v --record-actions --visualize

# View recorded actions from tests in the interactive viewer
# Tests record 4 worlds, so viewer must use 4 worlds to replay correctly
./build/viewer 4 --cpu tests/python/test_recordings/test_name/actions.bin
```

#### Testing Configuration

The project uses pytest for testing with the following configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["."]
norecursedirs = ["external", "build", ".venv", "*.egg", "dist"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

This configuration:
- Excludes the `external` directory (containing Madrona framework code) from test discovery
- Skips build artifacts and virtual environment directories
- Follows standard pytest naming conventions for test discovery

##### Testing Flags

The test suite supports several custom flags:

- `--no-gpu`: Skip all tests that require GPU. This is the default way to run tests.
- `--record-actions`: Record agent actions during test execution for viewer replay
- `--visualize`: Automatically launch the viewer after tests complete (requires --record-actions)

##### Testing Order

**IMPORTANT**: Always run tests in this order:
1. **CPU tests first**: `uv run --extra test pytest tests/python/ -v --no-gpu`
2. **GPU tests only after CPU tests pass**: `uv run --extra test pytest tests/python/ -v -k "gpu"`

This ensures that basic functionality is validated before testing GPU-specific features.

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

When the users says "debug the code" or "debug this function" or makes other references to "debugging" interpret this as the user requesting you to 

1. read docs/GDB_GUIDE.md
2. use the MCP GDB server to trace through the code and diagnose the cause of issues, check the values of variables during execution etc

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
- `src/viewer.cpp'` : Renders the environment, and allows user to switch to first person view to control the agent

### System Execution Order
The task graph defines precise system dependencies:
1. **Input Processing**: 
   - `movementSystem` - Converts actions to forces/torques
2. **Physics Pipeline**: 
   - `broadphase_setup` - Builds BVH for collision detection
   - `physicsStepTasks` - Runs physics simulation (multiple substeps)
   - `agentZeroVelSystem` - Zeros agent velocities for direct control
   - `physicsCleanupTasks` - Finalizes physics state
3. **Game Logic Pipeline**:
   - `stepTrackerSystem` - Decrements steps remaining, sets done flag
   - `rewardSystem` - Computes rewards (only at episode end)
   - `resetSystem` - Handles episode resets
4. **GPU-specific Systems** (when enabled):
   - `RecycleEntitiesNode` - Reclaims deleted entity IDs
   - `ResetTmpAllocNode` - Clears temporary allocations
   - Entity sorting by world ID for cache efficiency
5. **Post-Reset**:
   - Second `broadphase_setup` - Rebuilds BVH after reset
6. **Output**: 
   - `collectObservationsSystem` - Gathers observations for policy
7. **Rendering** (if enabled):
   - `RenderingSystem::setupTasks` - Updates render state

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
- **Self Observation**: Contains agent's global position (x, y, z), maximum Y reached (maxY), and rotation angle (theta)
- **Normalization**: Positions normalized by world size, angles normalized by π
- **Reward**: Progress-based reward given only at episode end (normalized max Y position reached)

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

**Function: `Manager::Manager()`**
1. Initialize implementation via `Impl::init()`
2. Force reset all worlds via `triggerReset()`
3. Execute one step via `step()`

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
   - Source file list and compilation flags for JIT compilation
   - Configuration for world count, task graphs, and exported buffers
   - Device memory allocation happens during construction
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

`Manager::Impl::init()` (src/mgr.cpp:443) - Initializes the simulation configuration based on execution mode (CPU or CUDA). For CPU mode, creates a PhysicsLoader and calls loadPhysicsObjects() to load collision geometry. Sets up the physics object manager and random number generator.

**Manager Creation - Step 2: Load Physics Assets**

`loadPhysicsObjects()` - Loads collision meshes from OBJ files (cube, wall, agent) and processes them into optimized collision hulls. Configures physics properties for each object type including mass, friction coefficients, and movement constraints. Specifically constrains agent rotation to Z-axis only by setting X and Y inverse inertia to zero.

**Manager Creation - Step 3: Load Render Assets**

`loadRenderObjects()` (if rendering enabled) - Loads visual meshes from OBJ files for rendering. Defines materials with colors and assigns them to mesh parts. Loads texture images and configures scene lighting parameters.

**Manager Creation - Step 4: Create Executor**

`TaskGraphExecutor` (CPU) or `MWCudaExecutor` (GPU) construction - Creates the execution engine for running the simulation. Configures the number of parallel worlds, exported buffers for Python access, and task graphs. For GPU mode, also specifies source files for JIT compilation.

**Manager Creation - Step 5: Get Export Pointers**

`getExported()` calls - Retrieves pointers to the exported component data that will be accessible from Python. Each export ID corresponds to a component registered in `Sim::registerTypes()`. The ExportID enum [GAME_SPECIFIC] defines all the tensors that will be exposed to Python: Reset, Action, Reward, Done, SelfObservation, StepsRemaining.

**Manager Creation - Step 6: Initial World Setup**

**Function: `Manager::Manager()` Constructor**
The Manager constructor:
- Calls `Impl::init()` to set up all the internal state
- [REQUIRED_INTERFACE] Forces an initial reset for all worlds by calling `triggerReset()` for each world to ensure they start in a valid state
- [BOILERPLATE] Executes one simulation step to populate initial observations so Python can see the initial state

**Function: `triggerReset()`**
[GAME_SPECIFIC] Sets the reset flag in the WorldReset buffer to trigger world regeneration. Handles both CPU (direct memory write) and CUDA (cudaMemcpy) execution modes.
   
### 2. **Executor Initialization Phase** (inside TaskGraphExecutor/MWCudaExecutor constructor)

**Executor Init - Step 1: Base Setup**
- Base class `ThreadPoolExecutor` constructor runs first
- Call `getECSRegistry()` to obtain the ECS registry

**Executor Init - Step 2: ECS Registration** (`Sim::registerTypes()`)

**Function: `Sim::registerTypes()`**
[GAME_SPECIFIC] Registers all ECS types with the registry:
- Registers all game components (Action, Reward, Done, etc.)
- Registers game archetypes/entity templates (Agent for player-controlled entities, PhysicsEntity for movable objects, etc.)
- Exports components for Python access using the pattern `registry.exportColumn<Archetype, Component>(ExportID)`
- Exports singletons like WorldReset and per-entity components like Action, Reward, observations
- Export IDs must match the ExportID enum in types.hpp
- **Memory allocation**: Virtual address space (1B × component size) reserved per exported component

**Executor Init - Step 3: World Construction**
- Create per-world contexts and task graph managers
- Construct world data instances (`WorldT` objects)

**Executor Init - Step 4: Task Graph Setup** (`Sim::setupTasks()`)

**Function: `Sim::setupTasks()`**
Builds the static execution graph defining system execution order. The function creates a task graph with the following phases:

1. **Input Processing**: `movementSystem` converts actions from the policy into forces/torques for physics
2. **Spatial Structure Build**: `PhysicsSystem::setupBroadphaseTasks` builds the BVH for collision detection
3. **Physics Simulation**: `PhysicsSystem::setupPhysicsStepTasks` runs physics collision detection and solver (multiple substeps)
4. **Post-Physics Corrections**: `agentZeroVelSystem` zeros velocities for direct control
5. **Physics Cleanup**: `PhysicsSystem::setupCleanupTasks` finalizes physics subsystem work
6. **Episode Management**: 
   - `stepTrackerSystem` decrements steps remaining and sets done flag
   - `rewardSystem` computes rewards (only given at episode end)
   - `resetSystem` conditionally resets the world if episode is over
7. **GPU Backend Tasks** (when enabled):
   - `ResetTmpAllocNode` clears temporary allocations
   - `RecycleEntitiesNode` reclaims deleted entity IDs
8. **Post-Reset Spatial Rebuild**: Second BVH build (required due to current API limitation)
9. **Observation Collection**: `collectObservationsSystem` gathers all observations for the policy
10. **Rendering Setup**: `RenderingSystem::setupTasks` (if rendering enabled)
11. **Entity Sorting** (GPU only): Sorts entities by world ID for cache efficiency

**Executor Init - Step 5: Export Initialization**
- Call `initExport()` which triggers initial `copyOutExportedColumns()`
- **Memory allocation**: Physical pages committed as needed during copy
- **Now `getExported()` can be called** to retrieve pointers to exported data

### 3. **Per-World Sim Construction Phase** (`Sim::Sim()` constructor)

**World Init - Step 1: Calculate Entity Limits**

**Function: `Sim::Sim()` Constructor**
The Sim constructor initializes each simulation world by:
- [GAME_SPECIFIC] Calculating the maximum total entities based on game parameters (agents + floor + walls)
- [BOILERPLATE] Initializing the physics system with the entity limit, time step, substeps, gravity, and rigid body object manager
- [BOILERPLATE] Storing the initial random key and auto-reset configuration
- [GAME_SPECIFIC] Calling `createPersistentEntities()` to create agents, walls, and floor
- [GAME_SPECIFIC] Calling `initWorld()` to generate the initial level layout

**World Init - Step 2: Create Persistent Entities** (`createPersistentEntities()`)

The `createPersistentEntities` function creates entities that persist across all episodes. These entities are created once during world initialization and are reused/reset for each episode rather than destroyed and recreated.

**Function: `createPersistentEntities()`**
[GAME_SPECIFIC] Creates all persistent entities by:
- Creating the floor plane as a static physics entity at the origin
- Creating three outer boundary walls (behind, right, left - front is open) scaled to span the world width
- For each agent:
  - Creating an Agent entity
  - Attaching a camera view (if rendering enabled) with specified near/far planes and offset
  - Initializing components: scale, object ID, response type as Dynamic, empty grab state, and entity type
- Populating the OtherAgents component for each agent with references to all other agents (for teammate observations)

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

**Function: `resetPersistentEntities()` Agent Reset Logic**
Inside `resetPersistentEntities()`, each persistent entity is re-registered with the physics broadphase, and for each agent:
- [GAME_SPECIFIC] Sets a random spawn position near the starting wall, slightly above the floor
- [GAME_SPECIFIC] Spreads agents left/right alternately by adding/subtracting from X position
- [GAME_SPECIFIC] Applies a random initial rotation (±45 degrees around the up axis)
- [GAME_SPECIFIC] Clears any existing grab constraints from the previous episode
- [GAME_SPECIFIC] Resets progress tracking to the spawn Y position
- [GAME_SPECIFIC] Resets action to default values (no movement, center rotation bucket, no grab)
- [GAME_SPECIFIC] Resets steps remaining timer to episode length

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

**Function: `resetSystem()`**
[REQUIRED_INTERFACE] Reset system that runs each frame to check if the current episode is complete or if external code has triggered a reset. The function:
- Checks the manual reset flag from the WorldReset singleton
- If auto-reset is enabled, checks if any agent's Done flag is set
- When reset is needed, clears the reset flag, calls `cleanupWorld()` to destroy dynamic entities, and calls `initWorld()` to create a new episode

**Reset Triggers:**
1. **Manual Reset**: External code sets `WorldReset.reset = 1` via `Manager::triggerReset()`
2. **Auto-Reset**: When any agent's `Done` flag is set (if `autoReset` enabled)
3. **Episode Timeout**: `stepTrackerSystem` sets `Done.v = 1` when steps reach limit

#### **World Cleanup** (`cleanupWorld()` - src/sim.cpp:100)

Destroys all dynamically created entities from the current episode:

**Function: `cleanupWorld()`**
[GAME_SPECIFIC] Helper function that destroys all dynamic entities before a reset. This should:
- Iterate through all dynamically created entities (those created during level generation)
- Call `ctx.destroyRenderableEntity()` for each dynamic entity
- Clear any references to destroyed entities
- Leave persistent entities (agents, static world geometry) untouched

**Important**: Persistent entities (agents, floor, outer walls) are NOT destroyed during cleanup.

#### **World Initialization** (`initWorld()` - src/sim.cpp:119)

Prepares the world for a new episode:

**Function: `initWorld()`**
[GAME_SPECIFIC] Helper function that initializes a new escape room world by:
- Calling `PhysicsSystem::reset()` [BOILERPLATE] to clear all collision pairs, constraints, and the BVH
- Creating a new RNG instance [BOILERPLATE] with a unique seed based on the episode counter and world ID
- Calling `generateWorld()` [GAME_SPECIFIC] to create the new level layout

**Call Sequence:**
1. `PhysicsSystem::reset()` - [BOILERPLATE] Clears all physics state
2. RNG initialization - [BOILERPLATE] New random seed for this episode
3. `generateWorld()` - [GAME_SPECIFIC] Creates new level layout

#### **World Generation** (`generateWorld()` - src/level_gen.cpp:522)

Orchestrates the world generation process:

**Function: `generateWorld()`**
Orchestrates the world generation by calling:
- `resetPersistentEntities()` to reset agents and re-register persistent entities with the physics system
- `generateLevel()` to create the new procedural room layout

#### **Persistent Entity Reset** (`resetPersistentEntities()` - src/level_gen.cpp:193)

Resets entities that persist across episodes:

**Function: `resetPersistentEntities()`**
Resets all persistent entities (entities that survive across episodes) by:
- Re-registering the floor plane and boundary walls with the physics system [BOILERPLATE]
- For each agent [GAME_SPECIFIC]:
  - Re-registering with the physics system
  - Setting a random spawn position near the starting wall, spread left/right alternately
  - Applying a random initial rotation (±45 degrees)
  - Clearing any existing grab constraints from the previous episode
  - Resetting progress tracking to the spawn Y position
  - Clearing all physics state (velocity, forces, torques) [BOILERPLATE]
  - Resetting the action to default values and timer to episode length

#### **Level Generation** (`generateLevel()` - src/level_gen.cpp)

Creates the procedural room layout for the new episode:

**Function: `generateLevel()`**
[GAME_SPECIFIC] Generates the procedural level layout. This function should:
- Create any dynamic entities needed for the current episode
- Set up level geometry, obstacles, goals, or interactive elements
- Register all new entities with the physics system using `registerRigidBodyEntity()`
- Initialize entity positions, rotations, and other component values
- Store references to dynamic entities for cleanup during reset

#### **Key Points:**

- **Persistent vs Dynamic**: Some entities persist across episodes (agents, static world geometry) while others are recreated each episode
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

## Procedures for common tasks

are documented in the docs folder, the filenames should be self explanatory

- ADD_COMPONENT.md
- ADD_SYSTEM.md

## Useful documents in the docs folder

- README_BINDINGS.md : python bindings
- ECS_ARCHITECTURE.md : the madrona ECS system


### GPU Execution of Systems

Madrona automatically compiles and executes systems on GPU using [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)

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
