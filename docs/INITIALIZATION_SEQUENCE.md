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
