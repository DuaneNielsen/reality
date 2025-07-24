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