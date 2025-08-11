# Madrona Escape Room - Collision Detection and Handling System

## Overview

The Madrona escape room uses a sophisticated two-phase collision detection system with XPBD (Extended Position Based Dynamics) for physics resolution. The system is designed for high-performance batch simulation, supporting thousands of parallel worlds on GPU.

## Collision Detection Pipeline

### 1. Two-Phase Detection

**Broadphase (Spatial Culling)**:
```cpp
// Built every frame after position updates
auto broadphase_setup = PhysicsSystem::setupBroadphaseTasks(
    builder, {movement_sys, set_door_pos});
```
- Uses a BVH (Bounding Volume Hierarchy) for spatial acceleration
- Quickly identifies potential collision pairs
- Must rebuild after any position changes

**Narrowphase (Exact Testing)**:
- Tests candidate pairs from broadphase
- Generates exact contact points and normals
- Supports multiple collision primitive types:
  - Sphere-Sphere
  - Hull-Hull (convex hulls using SAT algorithm)
  - Sphere-Hull
  - Plane collisions (floor/walls)

### 2. Collision Response Types

Each entity has a `ResponseType` that determines behavior:

```cpp
enum class ResponseType : uint32_t {
    Dynamic,    // Fully simulated (agents, pushable blocks)
    Kinematic,  // Position-controlled but affects others
    Static,     // Immovable (walls, floor)
};
```

### 3. Physics Properties Configuration

Objects are configured with specific physics parameters:

```cpp
// Example: Setting up a pushable cube
setupRigidBodyEntity(
    ctx,
    cube_entity,
    position,
    rotation,
    SimObject::Cube,        // Object type for collision geometry
    EntityType::Cube,       // Game logic type
    ResponseType::Dynamic,  // Can be pushed
    scale                   // Optional scaling
);

// In physics loader:
setupHull(SimObject::Cube, 
    0.075f,  // Inverse mass (~13.3kg cube)
    {.muS = 0.5f, .muD = 0.75f}  // Friction coefficients
);
```

### 4. Object Types and Properties

**Cube (pushable blocks)**:
- Mass: ~13.3kg (inverse mass = 0.075)
- Static friction: μs = 0.5
- Dynamic friction: μd = 0.75
- Response: Dynamic

**Wall/Door**:
- Mass: Infinite (inverse mass = 0.0)
- Friction: μs = μd = 0.5
- Response: Static

**Agent**:
- Mass: 1kg (unit mass for direct control)
- Friction: μs = μd = 0.5
- Response: Dynamic
- Special: Rotation constrained to Z-axis only

**Floor Plane**:
- Mass: Infinite
- Friction: μs = μd = 0.5
- Response: Static

### 5. Contact Constraint Solver

The system uses XPBD (Extended Position Based Dynamics):

```cpp
struct ContactConstraint {
    Loc ref;              // Reference entity
    Loc alt;              // Other entity  
    Vector4 points[4];    // Contact points (xyz + penetration depth w)
    int32_t numPoints;    // Number of contact points
    Vector3 normal;       // Contact normal
};
```

**Solver Configuration**:
- Fixed timestep: 0.04s (25Hz)
- Physics substeps: 4 per frame
- Gravity: -9.8 m/s² in Z direction
- Restitution threshold: Based on gravity and timestep

### 6. Collision Handling Pipeline

The collision pipeline executes in this order:

1. **Movement System** - Converts actions to forces
2. **Broadphase Setup** - Builds spatial structure
3. **Narrowphase Detection** - Finds exact contacts
4. **Constraint Preparation** - Sets up contact constraints
5. **Solver Iterations** - Resolves collisions (substeps)
6. **Position Integration** - Updates final positions
7. **Cleanup** - Clears temporary collision data

### 7. Special Cases

**Agent Collisions**:
```cpp
// Agents have rotation locked to prevent tipping
rigid_body_assets.metadatas[Agent].mass.invInertiaTensor.x = 0.f;  // No X rotation
rigid_body_assets.metadatas[Agent].mass.invInertiaTensor.y = 0.f;  // No Y rotation
// Only Z rotation allowed
```

**Velocity Zeroing**:
```cpp
// Agents have velocities zeroed each frame for direct control
inline void agentZeroVelSystem(Engine &ctx, Velocity &vel, Action &)
{
    vel.linear = Vector3::zero();
    vel.angular = Vector3::zero();
}
```

### 8. Key Implementation Details

**Contact Generation**:
- SAT algorithm tests all face normals and edge pairs
- Clips incident face against reference face for contact manifold
- Handles edge-edge collisions for stable box stacking

**Friction Model**:
- Static friction prevents sliding below threshold
- Dynamic friction applies when sliding occurs
- Friction forces computed in tangent plane to contact normal

### 9. Performance Optimizations

- **GPU Support**: Parallel collision detection across worlds
- **Entity Sorting**: Groups entities by world for cache efficiency
- **Warp-Level Parallelism**: GPU narrowphase uses thread cooperation
- **Monolithic Solver**: All constraints solved in single kernel
- **Zero-Copy Design**: Direct memory mapping to Python tensors

### 10. Collision Events

While the system defines collision event structures:
```cpp
struct CollisionEvent {
    Entity a;
    Entity b;
};
```

The escape room implementation doesn't currently use collision callbacks. Interactions are handled through:
- Spatial queries (removed grab system used raycasting)
- Proximity checks in game logic
- Direct component queries

## Key Features

- **Deterministic**: Essential for RL training reproducibility
- **Batch Processing**: Thousands of worlds in parallel
- **GPU Accelerated**: Collision detection runs on GPU
- **No Callbacks**: Interactions handled through game logic systems
- **Continuous Detection**: Prevents tunneling with velocity-based AABB expansion

The system is designed for high-performance batch simulation while maintaining physical accuracy for multi-agent reinforcement learning training.

## Creating Collision Handlers

While the escape room doesn't use collision callbacks directly, here are several methods to handle collision interactions:

### 1. Collision Events (Currently Unused)

While Madrona defines collision event structures, the escape room doesn't use them. But here's how you could implement collision event handling:

```cpp
// First, register to receive collision events in your system
inline void myCollisionHandlerSystem(Engine &ctx)
{
    // Query for collision events (would need to be exported)
    ctx.query<CollisionEvent>().forEach([&](CollisionEvent &event) {
        Entity a = event.a;
        Entity b = event.b;
        
        // Check entity types
        EntityType typeA = ctx.get<EntityType>(a);
        EntityType typeB = ctx.get<EntityType>(b);
        
        // Handle specific collision types
        if (typeA == EntityType::Agent && typeB == EntityType::Cube) {
            // Handle agent-cube collision
        }
    });
}
```

### 2. Proximity-Based Detection (Recommended)

This is how the escape room currently handles interactions:

```cpp
// Example: Check if agent is near a button
inline void buttonPressSystem(Engine &ctx,
                             Entity agent_entity,
                             Position agent_pos)
{
    float button_press_radius = 1.5f;
    
    // Query all buttons in the level
    LevelState &level = ctx.singleton<LevelState>();
    for (Room &room : level.rooms) {
        for (Entity button : room.entities) {
            if (ctx.get<EntityType>(button) != EntityType::Button) continue;
            
            Position button_pos = ctx.get<Position>(button);
            float dist = (agent_pos - button_pos).length();
            
            if (dist < button_press_radius) {
                // Agent is pressing this button!
                ctx.get<ButtonState>(button).isPressed = true;
            }
        }
    }
}
```

### 3. Spatial Queries (Most Flexible)

Use the physics system's spatial queries:

```cpp
// Example: Find all entities within a radius
inline void explosionSystem(Engine &ctx,
                           Position explosion_pos)
{
    float explosion_radius = 5.0f;
    AABB explosion_bounds {
        explosion_pos - Vector3::all(explosion_radius),
        explosion_pos + Vector3::all(explosion_radius)
    };
    
    // Use BVH to find nearby entities
    ctx.getSpatialAABBTree().findIntersecting(explosion_bounds, 
        [&](Entity e) {
            Position e_pos = ctx.get<Position>(e);
            float dist = (e_pos - explosion_pos).length();
            
            if (dist < explosion_radius) {
                // Apply explosion force/damage
                ctx.get<Health>(e).damage(100.0f * (1.0f - dist/explosion_radius));
            }
        });
}
```

### 4. Post-Physics Collision Detection

Create a system that runs after physics to check for contacts:

```cpp
// Add this system after physics in setupTasks()
inline void collisionResponseSystem(Engine &ctx,
                                  Entity entity,
                                  Position pos,
                                  EntityType type)
{
    // Check if this entity is touching specific types
    if (type == EntityType::Agent) {
        // Check ground contact for jumping
        AABB ground_check = {
            pos - Vector3{0.5f, 0.5f, 0.1f},
            pos + Vector3{0.5f, 0.5f, -0.01f}
        };
        
        bool on_ground = false;
        ctx.getSpatialAABBTree().findIntersecting(ground_check,
            [&](Entity other) {
                if (ctx.get<EntityType>(other) == EntityType::Wall ||
                    ctx.get<EntityType>(other) == EntityType::Floor) {
                    on_ground = true;
                }
            });
        
        ctx.get<AgentState>(entity).canJump = on_ground;
    }
}
```

### 5. Custom Trigger Volumes

Create invisible trigger entities:

```cpp
// Create a trigger zone
Entity trigger = ctx.makeEntity<PhysicsEntity>();
ctx.get<Position>(trigger) = trigger_pos;
ctx.get<EntityType>(trigger) = EntityType::Trigger;
ctx.get<ResponseType>(trigger) = ResponseType::Static;
// Use a larger collision shape for the trigger zone

// Check for entities entering the trigger
inline void triggerSystem(Engine &ctx)
{
    ctx.query<Position, EntityType>().forEach([&](Position pos, EntityType type) {
        if (type == EntityType::Trigger) {
            // Check for agents in trigger zone
            AABB trigger_bounds = /* compute from trigger size */;
            
            ctx.getSpatialAABBTree().findIntersecting(trigger_bounds,
                [&](Entity other) {
                    if (ctx.get<EntityType>(other) == EntityType::Agent) {
                        // Agent entered trigger zone!
                        // Fire event, open door, etc.
                    }
                });
        }
    });
}
```

### Implementation Steps

1. **Add your collision handler system to the task graph**:
```cpp
// In Sim::setupTasks()
auto collision_handler = builder.addToGraph<ParallelForNode<Engine,
    collisionResponseSystem,
    Entity,
    Position,
    EntityType
>>({phys_cleanup}); // Run after physics
```

2. **Choose the appropriate method**:
   - **Proximity checks**: Simple, efficient for small radius checks
   - **Spatial queries**: Best for area effects, triggers
   - **Post-physics checks**: For physics-dependent states (grounded, wall contact)

3. **Performance considerations**:
   - Proximity checks scale O(n²) - use sparingly
   - Spatial queries use BVH - efficient for many checks
   - Consider caching collision state in components

The escape room uses proximity-based detection because it's simple and sufficient for button presses and goal zones. For more complex interactions, spatial queries provide the best balance of flexibility and performance.

## High-Performance Agent Collision Detection

For detecting if an agent hit another entity with geometry, the best performance approach depends on your specific needs:

### Best Performance: Use Existing Physics Contact Data

The most efficient approach is to leverage the collision data that the physics system already generates:

#### Option 1: Contact Constraint Queries (Most Efficient)

The physics system already detects all collisions during narrowphase. You can query these contacts directly:

```cpp
// This would run after physics narrowphase
inline void agentCollisionSystem(Engine &ctx, 
                                Entity agent_entity,
                                EntityType agent_type)
{
    if (agent_type != EntityType::Agent) return;
    
    // Query contact constraints involving this agent
    ctx.query<ContactConstraint>().forEach([&](ContactConstraint &contact) {
        // Check if this agent is involved in the contact
        Entity ref_entity = ctx.get<Entity>(contact.ref);
        Entity alt_entity = ctx.get<Entity>(contact.alt);
        
        if (ref_entity == agent_entity || alt_entity == agent_entity) {
            // Agent is colliding!
            Entity other = (ref_entity == agent_entity) ? alt_entity : ref_entity;
            EntityType other_type = ctx.get<EntityType>(other);
            
            // Handle collision based on what we hit
            handleAgentCollision(ctx, agent_entity, other, other_type);
        }
    });
}
```

**Performance**: O(active contacts) - typically very few per frame

#### Option 2: Tight AABB Overlap Check (Good Performance)

If you need to detect "near misses" or slightly larger collision zones:

```cpp
inline void agentHitDetection(Engine &ctx,
                             Entity agent_entity,
                             Position agent_pos,
                             ObjectID agent_obj_id)
{
    // Get agent's actual collision bounds from physics
    const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
    AABB agent_bounds = obj_mgr.rigidBodies[agent_obj_id.idx].aabb;
    
    // Transform to world space
    agent_bounds.pMin += agent_pos;
    agent_bounds.pMax += agent_pos;
    
    // Slightly expand for "hit" detection if needed
    float hit_margin = 0.1f;
    agent_bounds.pMin -= Vector3::all(hit_margin);
    agent_bounds.pMax += Vector3::all(hit_margin);
    
    // Use BVH for efficient spatial query
    ctx.singleton<broadphase::BVH>().findIntersecting(agent_bounds,
        [&](Entity other) {
            if (other == agent_entity) return;
            
            // Do precise hit check if needed
            Position other_pos = ctx.get<Position>(other);
            ObjectID other_obj_id = ctx.get<ObjectID>(other);
            
            // Could do more precise shape-vs-shape test here
            // But AABB overlap is usually sufficient
            
            handleAgentHit(ctx, agent_entity, other);
        });
}
```

**Performance**: O(log n) BVH traversal + O(local entities)

#### Option 3: Tagged Contact Points (Best for Specific Interactions)

Add a component to track recent collisions:

```cpp
struct CollisionInfo {
    Entity lastHitEntity;
    float hitTime;
    Vector3 hitNormal;
    float hitForce;
};

// In your physics post-processing
inline void trackAgentCollisions(Engine &ctx,
                                Entity agent,
                                CollisionInfo &col_info)
{
    // Reset old collision
    col_info.lastHitEntity = Entity::none();
    
    // Check narrowphase results
    ctx.query<ContactConstraint>().forEach([&](ContactConstraint &contact) {
        Entity ref = ctx.get<Entity>(contact.ref);
        Entity alt = ctx.get<Entity>(contact.alt);
        
        if (ref == agent || alt == agent) {
            Entity other = (ref == agent) ? alt : ref;
            
            // Calculate impact force from penetration depth
            float max_penetration = 0.f;
            for (int i = 0; i < contact.numPoints; i++) {
                max_penetration = fmaxf(max_penetration, -contact.points[i].w);
            }
            
            col_info.lastHitEntity = other;
            col_info.hitTime = ctx.data().curWorldEpisode;
            col_info.hitNormal = contact.normal;
            col_info.hitForce = max_penetration * 100.f; // Approximate
        }
    });
}
```

**Performance**: O(active contacts) with caching

### Recommendation

**For best performance**, use **Option 1 (Contact Constraint Queries)** because:

1. **No extra collision detection** - reuses physics results
2. **Exact collision data** - includes contact points, normals, penetration
3. **Already filtered** - only active contacts exist
4. **Cache friendly** - contacts are temporary entities, grouped together

**Implementation in setupTasks():**
```cpp
// After narrowphase, before solver
auto collision_detection = builder.addToGraph<ParallelForNode<Engine,
    agentCollisionSystem,
    Entity,
    EntityType
>>({narrowphase_node}); // Run right after narrowphase detection
```

This approach gives you:
- Real collision detection (not approximations)
- Contact normals for hit direction
- Penetration depth for hit strength
- Multiple contact points if needed
- Near-zero additional computational cost

The physics system has already done the hard work - just read its results!

## BVH Entity Allocation and Management

The Bounding Volume Hierarchy (BVH) is the core spatial acceleration structure that enables efficient collision detection. Understanding how it allocates and manages entities is crucial for proper `max_entities` configuration.

### BVH Initialization and Entity Limits

The BVH is initialized during physics system setup with a fixed maximum entity count:

```cpp
// In Sim::Sim() constructor
CountT max_total_entities = compiled_level.max_entities;

phys::PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
    consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
    max_total_entities);
```

The BVH constructor allocates fixed-size arrays based on this limit:

```cpp
BVH::BVH(const ObjectManager *obj_mgr,
         CountT max_leaves,
         float leaf_velocity_expansion,
         float leaf_accel_expansion)
    : leaf_entities_((Entity *)rawAlloc(sizeof(Entity) * max_leaves)),
      leaf_obj_ids_((ObjectID *)rawAlloc(sizeof(ObjectID) * max_leaves)),
      leaf_aabbs_((AABB *)rawAlloc(sizeof(AABB) * max_leaves)),
      // ... other arrays allocated with max_leaves size
      num_allocated_leaves_(max_leaves)
```

**Critical Constraint**: Once allocated, the BVH cannot grow beyond `max_leaves`. The assertion that failed in our tests:
```cpp
assert(leaf_idx < num_allocated_leaves_);  // In BVH::reserveLeaf()
```

### What Counts as a "Physics Entity" for BVH

Not all Madrona entities require BVH leaves. Only entities with physics bodies that participate in collision detection consume BVH slots:

#### **Entities that DO consume BVH leaves:**
1. **Agents** (2 per world in escape room)
   - Have physics bodies with Dynamic response
   - Participate in collision detection
   - Each agent = 1 BVH leaf

2. **Physical Level Geometry** (variable per compiled level)
   - **Walls**: Static physics bodies from compiled level tiles
   - **Cubes**: Dynamic pushable blocks from compiled level tiles  
   - **Floor**: Single static plane entity
   - Each physical entity = 1 BVH leaf

3. **Physics-Enabled Game Objects** (when implemented)
   - Buttons, doors, movable objects with collision
   - Only if they have physics bodies

#### **Entities that do NOT consume BVH leaves:**
1. **Render-Only Entities** 
   - Origin marker boxes (3 per world) - visual debugging aids
   - UI elements, particle effects, etc.
   - These use `makeRenderableEntity<RenderOnlyEntity>()`

2. **Pure Logic Entities**
   - AI state machines, timers, score counters
   - Game logic entities without spatial representation

3. **Non-Physical Components**
   - Observation data, action buffers, metadata
   - These are components, not spatial entities

### Entity Creation Breakdown in Escape Room

Based on `createPersistentEntities()` and level generation:

**Persistent Physics Entities (per world):**
- 1 floor plane (physics body)
- 2 agents (physics bodies)  
- **Total persistent: 3 physics entities**

**Persistent Render-Only Entities (per world):**
- 3 origin marker boxes (render-only)
- **Total render-only: 3 entities (DO NOT consume BVH leaves)**

**Variable Level Entities (from compiled level):**
- N walls (each tile with `TILE_WALL` type)
- M cubes (each tile with `TILE_CUBE` type)
- **Total level entities: N + M (varies by compiled level)**

### Correct max_entities Calculation

```cpp
// In compiled level generation
int32_t physics_entity_count = 0;

// Count level tiles that create physics entities
for (tile in compiled_level_tiles) {
    if (tile.type == TILE_WALL || tile.type == TILE_CUBE) {
        physics_entity_count++;
    }
    // TILE_SPAWN, TILE_EMPTY don't create physics entities
}

// Add persistent physics entities
physics_entity_count += 3;  // 1 floor + 2 agents

// Add safety buffer for temporary physics entities
int32_t safety_buffer = 20-50;  // Collision contacts, constraints, etc.

compiled_level.max_entities = physics_entity_count + safety_buffer;
```

### Performance Implications

**BVH Size Impact:**
- **Memory**: O(max_entities) arrays allocated upfront
- **Rebuild Cost**: O(max_entities log max_entities) when geometry changes
- **Query Cost**: O(log max_entities) for spatial queries

**Best Practices:**
1. **Minimize max_entities**: Only count actual physics bodies
2. **Exclude render-only entities**: They don't need collision detection  
3. **Add reasonable buffer**: 20-50 entities for contacts/constraints
4. **Per-world optimization**: Use different max_entities per world type

### Debugging BVH Allocation Issues

When you get `leaf_idx < num_allocated_leaves_` assertion failures:

1. **Count actual physics entities**: Check `createPersistentEntities()` and level generation
2. **Verify max_entities calculation**: Ensure it matches actual entity creation
3. **Check for entity leaks**: Old entities not properly cleaned up during resets
4. **Monitor BVH usage**: Add debug output for `num_leaves_` vs `num_allocated_leaves_`

```cpp
// Debug BVH usage
printf("BVH: %d/%d leaves allocated\n", 
       bvh.num_leaves_.load(), bvh.num_allocated_leaves_);
```

The BVH is a high-performance spatial acceleration structure, but it requires accurate upfront sizing to prevent allocation failures during simulation.

## ⚠️ Critical BVH Design Issues

### Issue 1: ObjectID Auto-Registration Problem

**Problem**: Any entity with an `ObjectID` component automatically consumes a BVH leaf slot, regardless of whether it actually needs collision detection.

**Impact**: 
- Render-only entities (like visual markers) waste BVH slots
- Impossible to have entities with visual assets that don't participate in physics
- Forces oversized BVH allocation for simple visual elements

**Current Workaround**: Include ALL entities with `ObjectID` in max_entities calculation:

```cpp
// CURRENT REALITY (suboptimal)
int32_t bvh_slot_count = 0;

// Count ALL entities that will get ObjectID (not just physics entities)
bvh_slot_count += physics_entities;      // Actual collision participants  
bvh_slot_count += visual_marker_count;   // Render-only but have ObjectID
bvh_slot_count += ui_elements_with_obj;  // UI that needs visual assets

compiled_level.max_entities = bvh_slot_count + buffer;
```

**Root Cause**: The physics system calls `registerEntity(entity, obj_id)` for any entity with `ObjectID`, without checking if the entity actually needs collision detection.

**Future Fix Needed**: 
- Decouple visual asset IDs from physics registration
- Add separate `RenderObjectID` component for render-only entities
- Only register entities that have both `ObjectID` AND physics components

### Issue 2: Fixed BVH Size Limitation

**Problem**: BVH size cannot be changed after initialization, even when world geometry changes dramatically.

**Impact**:
- Must pre-allocate for worst-case scenario across all worlds
- Memory waste when some worlds have simple geometry  
- Difficult to support dynamic level generation

**Current Workaround**: Allocate for maximum possible entities across all world types:

```cpp
// CURRENT REALITY (wasteful)
int32_t max_across_all_worlds = 0;
for (world_type in world_types) {
    max_across_all_worlds = max(max_across_all_worlds, 
                               world_type.entity_count);
}
compiled_level.max_entities = max_across_all_worlds + large_buffer;
```

**Future Fix Needed**:
- Dynamic BVH resizing during world resets
- Per-world BVH size optimization
- Streaming BVH allocation for procedural content

### Issue 3: Silent BVH Overflow

**Problem**: BVH overflow causes assertion failures with cryptic error messages, making debugging difficult.

**Manifestation**: 
```
Assertion failed: leaf_idx < num_allocated_leaves_
```

**Impact**: No clear indication of which entities caused overflow or how to fix sizing.

**Current Workaround**: Add debug logging to track BVH usage:

```cpp
// DEBUG: Monitor BVH allocation
printf("BVH Usage: %d/%d leaves (%.1f%% full)\n", 
       bvh.num_leaves_.load(), 
       bvh.num_allocated_leaves_,
       100.0f * bvh.num_leaves_.load() / bvh.num_allocated_leaves_);

// Add safety checks
if (bvh.num_leaves_.load() > bvh.num_allocated_leaves_ * 0.9f) {
    printf("WARNING: BVH approaching capacity limit!\n");
}
```

**Future Fix Needed**:
- Better error messages indicating entity types causing overflow
- Automatic capacity warnings before hitting limits
- Graceful degradation instead of assertion failures

## Recommended Immediate Actions

1. **Audit ObjectID Usage**: Review all entities receiving `ObjectID` components
2. **Minimize Render-Only ObjectIDs**: Remove `ObjectID` from pure visual elements where possible
3. **Conservative Sizing**: Always overestimate max_entities by 20-30%
4. **Add Monitoring**: Include BVH usage logging in debug builds

## Long-Term Architecture Improvements Needed

1. **Separate Render and Physics IDs**: 
   - `RenderObjectID` for visual assets only
   - `PhysicsObjectID` for collision participants only

2. **Dynamic BVH Sizing**:
   - Resize BVH during world resets
   - Per-world BVH optimization

3. **Better Error Handling**:
   - Graceful capacity warnings
   - Detailed overflow diagnostics  
   - Runtime capacity adjustment

4. **Memory Optimization**:
   - Optional BVH participation for entities
   - Streaming allocation for large worlds

This is a fundamental limitation that affects all Madrona applications using both rendering and physics systems.