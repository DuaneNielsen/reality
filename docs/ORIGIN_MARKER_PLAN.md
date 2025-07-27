# Origin Marker Implementation Plan

## Overview
Add a visual indicator at the world origin (0, 0, 0) using a render-only entity that doesn't participate in physics collision.

## Implementation Steps

### 1. Define RenderOnlyEntity Archetype
**File: `src/types.hpp`**
Add after the PhysicsEntity definition:
```cpp
// Archetype for entities that only need rendering, no physics
struct RenderOnlyEntity : public madrona::Archetype<
    Position, Rotation, Scale, ObjectID,
    madrona::render::Renderable
> {};
```

### 2. Register the New Archetype
**File: `src/sim.cpp`**
In `Sim::registerTypes()`, add after `registry.registerArchetype<PhysicsEntity>();`:
```cpp
registry.registerArchetype<RenderOnlyEntity>();
```

### 3. Add Origin Marker to Sim Structure
**File: `src/sim.hpp`**
Add to the `Sim` struct after the `agents` array:
```cpp
// Origin marker entity to visualize (0, 0, 0) position
Entity originMarker;
```

### 4. Create Origin Marker in createPersistentEntities
**File: `src/level_gen.cpp`**
At the end of `createPersistentEntities()` function, add:
```cpp
// Create origin marker - a medium cube at (0, 0, 0)
ctx.data().originMarker = ctx.makeRenderableEntity<RenderOnlyEntity>();
ctx.get<Position>(ctx.data().originMarker) = Vector3{0, 0, 0};
ctx.get<Rotation>(ctx.data().originMarker) = Quat{1, 0, 0, 0};
ctx.get<Scale>(ctx.data().originMarker) = Diag3x3{0.5f, 0.5f, 0.5f};
ctx.get<ObjectID>(ctx.data().originMarker) = ObjectID{(int32_t)SimObject::Cube};
```

## Key Design Decisions

### Why Render-Only Archetype?
- No Ghost ResponseType exists in Madrona (only Dynamic, Kinematic, Static)
- All PhysicsEntity objects participate in collision detection
- Creating a separate archetype allows true non-collidable rendering

### What We DON'T Need to Do
- No `setupRigidBodyEntity()` call
- No `registerRigidBodyEntity()` call  
- No inclusion in `resetPersistentEntities()`
- No addition to `max_total_entities` count
- No physics properties configuration

## Result
- Medium-sized (0.5x0.5x0.5) brown cube at world origin
- Visible in viewer but agents pass through it
- No physics overhead or collision detection
- Persistent across episode resets

## Future Enhancements
- Add axis indicators (X=red, Y=green, Z=blue lines)
- Create distinct material/color for origin marker
- Add toggle visibility option in viewer