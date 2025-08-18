# Madrona ECS Architecture

This document explains how the Madrona Engine's Entity Component System (ECS) determines which components are attached to entities.

## Overview

Madrona uses an **Archetype-based ECS** system, not an entity mask system. This design prioritizes performance over runtime flexibility by defining component sets at compile time.

## Key Concepts

### 1. Archetypes Define Component Sets

Each entity type is defined as an archetype that specifies its exact component set at compile time:

```cpp
// Example archetype definition
struct Agent : public madrona::Archetype<
    Position,
    Rotation, 
    Action,
    Reward,
    Done
> {};
```

### 2. Component Storage Structure

Components are stored using a Structure of Arrays (SoA) pattern for optimal cache efficiency:

- Each archetype has its own `Table` with columns for each component type
- The `ArchetypeStore` contains:
  - `TableStorage tblStorage` - the actual component data
  - `ColumnMap columnLookup` - maps component IDs to column indices using perfect hashing
  - `componentOffset` and `numComponents` - metadata about the archetype

### 3. Entity-Component Association

Each entity stores minimal data:
- `id`: unique identifier
- `gen`: generation counter for safe recycling

Entities map to a location (`Loc`) containing:
- `archetype`: index of the entity's archetype
- `row`: position in that archetype's table

### 4. Component Access Flow

When accessing a component via `ctx.get<ComponentT>(entity)`:

```cpp
// Simplified access flow
1. Get entity's location (archetype + row)
   Loc loc = entity_store_.getLoc(entity);

2. Look up the archetype
   ArchetypeStore &archetype = archetype_stores_[loc.archetype];

3. Check if archetype has the component via columnLookup
   auto col_idx = archetype.columnLookup.lookup(componentID<ComponentT>().id);

4. If found, access the component data
   if (col_idx.has_value()) {
       ComponentT* col = archetype.tblStorage.column<ComponentT>(col_idx);
       return col[loc.row];
   }
```

### 5. The Column Lookup System

- `StaticIntegerMap` provides O(1) lookup from component ID to column index
- Built during archetype registration with perfect hashing
- Allows fast checking if an archetype contains a specific component
- No runtime bitmasks needed

## Advantages of Archetype-Based Design

1. **Cache Efficiency**: All entities of the same archetype have components laid out contiguously in memory
2. **Fast Iteration**: Systems can iterate over all entities with specific components efficiently
3. **Type Safety**: Component sets are known at compile time, enabling compiler optimizations
4. **Zero Overhead**: No runtime masks or indirection for component access
5. **Memory Density**: Perfect memory layout with no wasted space

## Comparison with Mask-Based Systems

Traditional ECS systems often use bitmasks where each bit represents whether an entity has a specific component. Madrona's archetype approach trades:

**Flexibility Lost:**
- Cannot add/remove components from entities at runtime
- All entities of a type must have exactly the same components

**Performance Gained:**
- Direct memory access without indirection
- Perfect memory layout for SIMD and cache optimization
- Compile-time knowledge enables aggressive optimizations
- No memory overhead for component masks

## Implementation Details

### Archetype Registration

During `Sim::registerTypes()`:
```cpp
registry.registerArchetype<Agent>();
registry.registerArchetype<PhysicsEntity>();
```

### Component Export

Components can be exported for external access (e.g., Python):
```cpp
registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
```

### Query System

Systems automatically iterate over entities with matching component requirements:
```cpp
inline void movementSystem(Engine &ctx,
                          Position &pos,      // Only entities with Position
                          Velocity &vel)      // AND Velocity are processed
{
    // System logic
}
```

## Memory Layout and System Access Optimization

### Column-Oriented Component Storage

Madrona uses a **column-oriented storage** approach where each component type is stored in separate contiguous buffers:

- Each component type (Position, Velocity, etc.) gets its own dedicated memory allocation
- All Position components for an archetype are stored contiguously in one buffer
- This Structure of Arrays (SoA) layout maximizes cache efficiency

Example memory layout for an archetype with Position and Velocity:
```
Position buffer: [P0][P1][P2][P3]...
Velocity buffer: [V0][V1][V2][V3]...
Entity buffer:   [E0][E1][E2][E3]...
```

### System Execution and Component Retrieval

When a system runs in the task graph:

1. **Query Matching**: The system queries ALL archetypes that contain the required components
   - A system requiring `(Position, Velocity)` will process entities from ANY archetype containing both
   - Example: Both `Agent` and `Projectile` archetypes would be processed if they have the required components

2. **Direct Pointer Access**: Systems receive direct pointers to component arrays
   ```cpp
   // System gets pointers to the start of each component array
   Position* positions = archetype.getColumn<Position>();
   Velocity* velocities = archetype.getColumn<Velocity>();
   ```

3. **Linear Iteration**: Simple array indexing for maximum performance
   ```cpp
   for (int i = 0; i < num_entities; i++) {
       positions[i].x += velocities[i].x * dt;  // Direct array access
   }
   ```

### Key Performance Characteristics

**Systems operate on ALL entities with matching components:**
- The archetype doesn't matter from a system's perspective
- Only component presence determines which entities are processed
- Pre-computed queries avoid runtime type checking

**Cache-Optimized Access Patterns:**
- **Spatial Locality**: Components of the same type are contiguous
- **Temporal Locality**: Systems process all entities of an archetype before moving to the next
- **Predictable Access**: Linear traversal enables hardware prefetching
- **No Indirection**: Direct array access without pointer chasing
- **SIMD Potential**: Contiguous same-type data enables vectorization

### Multi-World Memory Layout

For multi-world simulations, Madrona offers two modes:

**Dynamic Mode**:
- Each world has separate tables
- Good for variable entity counts

**Fixed Mode**:
- Single large buffer for all worlds
- Components for world N start at offset `world_id * max_entities_per_world`
- Better for GPU execution with coalesced memory access

### Query Optimization

Queries are pre-computed and cached:
- Which archetypes match each query signature
- Component column indices within each archetype
- Eliminates runtime type checking during iteration

## Summary

Madrona's archetype-based ECS achieves high performance by:
- Defining component sets at compile time via archetypes
- Using perfect hashing for O(1) component lookups
- Storing components in cache-friendly SoA layout with separate buffers per component type
- Systems operating on all entities with matching components regardless of archetype
- Direct pointer access to component arrays for zero-overhead iteration
- Eliminating runtime overhead of component masks

This design is ideal for simulations where entity types are known upfront and maximum performance is critical.

## Entity Reference Safety

### How Entity References Work

Madrona uses a **generation-based system** to handle entity references safely when entities are destroyed:

#### Entity Structure
```cpp
struct Entity {
    uint32_t gen;  // Generation counter
    int32_t id;    // Entity ID
};
```

#### Generation Counter Mechanism

1. **When an entity is destroyed**: Its generation counter is incremented
2. **When an ID is reused**: The new entity gets the current generation number
3. **When accessing an entity**: The generation is validated

```cpp
// Entity validation during lookup
if (stored_generation != entity.gen) {
    return invalid_location;  // Access fails for stale references
}
```

### What Happens to Stale References

When entity A holds a reference to entity B and B is destroyed:

1. **Reference remains unchanged**: A still holds B's ID and old generation
2. **Access attempts fail gracefully**:
   - `getLoc(stale_entity)` returns `Loc::none()`
   - `getSafe<Component>(stale_entity)` returns invalid `ResultRef`
   - `get<Component>(stale_entity)` fails assertion in debug builds
3. **ID reuse is safe**: If B's ID is reused for entity C:
   - C gets generation = old_generation + 1
   - A's reference still has old generation
   - Generation mismatch prevents accessing wrong entity

### Safe Access Patterns

**Pattern 1: Check for Entity::none()**
```cpp
if (entity != Entity::none()) {
    // Safe to use entity
}
```

**Pattern 2: Use safe accessors**
```cpp
ResultRef<ComponentT> result = ctx.getSafe<ComponentT>(entity);
if (result.valid()) {
    ComponentT& component = result.value();
}
```

**Pattern 3: Clear references after destruction**
```cpp
// From escape room grab system
if (grab_state.constraintEntity != Entity::none()) {
    ctx.destroyEntity(grab_state.constraintEntity);
    grab_state.constraintEntity = Entity::none();  // Clear reference
}
```

### Key Safety Features

1. **Generation Validation**: Every entity access validates the generation counter
2. **No Dangling Pointers**: Uses IDs instead of pointers, preventing memory corruption
3. **Explicit Invalid State**: `Entity::none()` (gen=0xFFFFFFFF, id=0xFFFFFFFF)
4. **Thread Safety**: Generation updates use atomic operations
5. **Fail-Safe Design**: Invalid entities simply fail validation, no crashes

This system ensures that stale entity references never cause undefined behavior - they simply fail validation checks and return invalid results that can be handled gracefully.

## Related Documentation

- [Asset Management](ASSET_MANAGEMENT.md) - How physics and rendering assets are stored and accessed
- [Collision System](COLLISION_SYSTEM.md) - Physics system implementation
- [Initialization Sequence](INITIALIZATION_SEQUENCE.md) - World and entity setup process
- [Step Sequence](STEP_SEQUENCE.md) - Simulation update cycle