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
