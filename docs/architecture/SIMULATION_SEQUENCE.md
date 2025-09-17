# Simulation Step Sequence

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
struct Action {
    int32_t moveAmount;   // 0-3: stop/slow/medium/fast
    int32_t moveAngle;    // 0-7: 8 directional movement
    int32_t rotate;       // 0-4: turn left/right speeds
};
```

**CompiledLevel Singleton** (Critical for world generation)
```cpp
struct CompiledLevel {
    int32_t max_entities;        // For BVH sizing
    float world_min_x, world_max_x;  // World boundaries
    float world_min_y, world_max_y;  // For observation normalization
    int32_t num_spawns;          // Agent spawn count
    float spawn_x[8], spawn_y[8]; // Spawn positions
    float spawn_facing[8];        // Initial rotations
    // Tile arrays for level geometry...
};
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
**Location:** `src/sim.cpp:476`
**Purpose:** Calculate incremental forward progress rewards

**Details:**
- Skip if progress not initialized (sentinel -999999.0f)
- Reward = (new_maxY - old_maxY) / total_possible_progress
- Only forward Y movement gives positive reward
- Goal reached when normalized_progress >= 1.0
- Collision death overrides with -0.1 penalty

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

### Output Format
```cpp
// Tensor layout for Python access
struct StepOutput {
    // Per agent, per world
    float self_obs[num_worlds][num_agents][5];
    float compass[num_worlds][num_agents][128];
    float lidar[num_worlds][num_agents][128];
    float reward[num_worlds][num_agents];
    uint8_t done[num_worlds][num_agents];
    int8_t termination[num_worlds][num_agents];

    // Render output if enabled
    uint8_t rgb[num_worlds][height][width][4];
    float depth[num_worlds][height][width];
};
```