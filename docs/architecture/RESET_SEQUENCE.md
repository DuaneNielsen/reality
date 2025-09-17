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
struct CompiledLevel {
    static constexpr int MAX_TILES = 1024;
    static constexpr int MAX_SPAWNS = 8;

    // World boundaries
    float world_min_x, world_max_x;
    float world_min_y, world_max_y;
    float world_min_z, world_max_z;

    // Tile data arrays
    uint32_t object_ids[MAX_TILES];
    float tile_x[MAX_TILES], tile_y[MAX_TILES], tile_z[MAX_TILES];
    float tile_scale_x[MAX_TILES], tile_scale_y[MAX_TILES], tile_scale_z[MAX_TILES];
    Quat tile_rotation[MAX_TILES];
    bool tile_persistent[MAX_TILES];
    bool tile_render_only[MAX_TILES];
    bool tile_done_on_collide[MAX_TILES];

    // Spawn data
    float spawn_x[MAX_SPAWNS], spawn_y[MAX_SPAWNS];
    float spawn_facing[MAX_SPAWNS];

    uint32_t num_tiles;
    uint32_t num_spawns;
    uint32_t max_entities;
};
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
**Purpose:** [BOILERPLATE] Create unique random seed for episode

**Details:**
- Increment curWorldEpisode counter
- Combine episode counter with world ID
- Create new RNG instance with unique seed
- Ensures deterministic but varied generation

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
**Function:** `resetAgentPhysics()`
**Location:** `src/level_gen.cpp:97`
**Purpose:** [GAME_SPECIFIC] Reset agent state for new episode

**Details:**
- Access CompiledLevel singleton for spawn data
- For each agent:
  - Register with physics system
  - Use spawn position from level data (spawn_x, spawn_y)
  - Set facing angle from spawn_facing array
  - Initialize Progress with sentinel values (-999999.0f)
  - Reset velocity, forces, and torques to zero
  - Reset action components to defaults
  - Reset steps_taken to 0
  - Reset done flag to 0
  - Reset collision_death flag to 0
  - Reset reward to 0.0f
  - Initialize compass observation to zeros

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

### Output Format
```cpp
// Agent state after reset
struct AgentResetState {
    Position pos;          // From CompiledLevel spawn data
    Rotation rot;          // From CompiledLevel spawn_facing
    Velocity vel;          // Zero velocity
    Action action;         // Default action values
    Progress progress;     // Sentinel values (-999999.0f)
    float timer;           // Not used (steps_taken used instead)
    Done done;            // Reset to 0
    Reward reward;        // Reset to 0.0f
};

// World state after reset
struct WorldResetState {
    Entity persistentEntities[];  // Re-registered, not recreated
    Entity dynamicEntities[];     // Newly created from CompiledLevel
    RNG rng;                     // New seed for episode
    uint32_t curWorldEpisode;    // Incremented
};
```