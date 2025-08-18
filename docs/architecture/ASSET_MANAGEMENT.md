# Asset Management Architecture

This document explains how physics and rendering assets are managed in the Madrona Escape Room, including loading, storage, and runtime access patterns.

## Overview

The asset system provides efficient, GPU-optimized access to shared physics and rendering data across all simulation worlds. Assets are loaded once at startup and shared across all parallel worlds via direct array indexing.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CPU/HOST MEMORY                           │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │              PhysicsLoader (temporary)                     │     │
│  │  • Loads .obj files from disk                             │     │
│  │  • Builds asset arrays on CPU                             │     │
│  │  • Uploads to GPU via cudaMemcpy                          │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                   │                                 │
│                                   │ cudaMemcpy                      │
│                                   ▼                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                            GPU MEMORY                               │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   ObjectManager (SHARED)                     │   │
│  │                                                              │   │
│  │  collisionPrimitives[]    primitiveAABBs[]    metadata[]    │   │
│  │  ┌───┬───┬───┬───┐       ┌───┬───┬───┬───┐   ┌─────────┐  │   │
│  │  │ 0 │ 1 │ 2 │ 3 │       │ 0 │ 1 │ 2 │ 3 │   │ mass    │  │   │
│  │  ├───┼───┼───┼───┤       ├───┼───┼───┼───┤   │ friction│  │   │
│  │  │ C │ W │ A │ P │       │AAB│AAB│AAB│AAB│   │ per obj │  │   │
│  │  └───┴───┴───┴───┘       └───┴───┴───┴───┘   └─────────┘  │   │
│  │    ▲   ▲   ▲   ▲                                           │   │
│  │    │   │   │   └─── Plane    (SimObject::Plane = 3)       │   │
│  │    │   │   └─────── Agent    (SimObject::Agent = 2)       │   │
│  │    │   └─────────── Wall     (SimObject::Wall  = 1)       │   │
│  │    └─────────────── Cube     (SimObject::Cube  = 0)       │   │
│  │                                                              │   │
│  │  rigidBodyAABBs[]         rigidBodyPrimitiveOffsets[]      │   │
│  │  ┌───┬───┬───┬───┐       ┌───┬───┬───┬───┐                │   │
│  │  │ 0 │ 1 │ 2 │ 3 │       │ 0 │ 5 │ 8 │12 │                │   │
│  │  └───┴───┴───┴───┘       └───┴───┴───┴───┘                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                 ▲                                   │
│                                 │                                   │
│  ┌──────────────────────────────┼────────────────────────────────┐ │
│  │                              │                                 │ │
│  │         ┌────────────────────┼────────────────────┐           │ │
│  │         │                    │                     │           │ │
│  │  ┌──────▼─────┐      ┌───────▼──────┐     ┌──────▼─────┐     │ │
│  │  │  World 0   │      │   World 1    │     │  World 2   │     │ │
│  │  │            │      │              │     │            │     │ │
│  │  │ ObjectData │      │ ObjectData   │     │ ObjectData │     │ │
│  │  │ ┌────────┐ │      │ ┌────────┐   │     │ ┌────────┐ │     │ │
│  │  │ │ mgr* ──┼─┼──────┼─┤ mgr* ──┼───┼─────┼─┤ mgr* ──┼─┤     │ │
│  │  │ └────────┘ │      │ └────────┘   │     │ └────────┘ │     │ │
│  │  │            │      │              │     │            │     │ │
│  │  │ Entities:  │      │ Entities:    │     │ Entities:  │     │ │
│  │  │ ┌────────┐ │      │ ┌────────┐   │     │ ┌────────┐ │     │ │
│  │  │ │ Wall   │ │      │ │ Cube   │   │     │ │ Agent  │ │     │ │
│  │  │ │ obj.idx│ │      │ │ obj.idx│   │     │ │ obj.idx│ │     │ │
│  │  │ │  = 1   │ │      │ │  = 0   │   │     │ │  = 2   │ │     │ │
│  │  │ └────────┘ │      │ └────────┘   │     │ └────────┘ │     │ │
│  │  └────────────┘      └──────────────┘     └────────────┘     │ │
│  │                                                               │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  GPU Kernel Access Pattern:                                        │
│  ─────────────────────────                                         │
│  // Direct array indexing - no searches or hash maps               │
│  RigidBodyMetadata &meta = obj_mgr.metadata[obj_id.idx];          │
│  AABB &aabb = obj_mgr.rigidBodyAABBs[obj_id.idx];                 │
│  uint32_t offset = obj_mgr.rigidBodyPrimitiveOffsets[obj_id.idx];  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Asset Descriptors (CPU-side)

Static compile-time tables define asset properties:

```cpp
// asset_descriptors.cpp
const PhysicsAssetDescriptor PHYSICS_ASSETS[] = {
    { .name = "cube", .objectId = SimObject::Cube, ... },
    { .name = "wall", .objectId = SimObject::Wall, ... },
    // ...
};
```

### 2. SimObject Enum

Serves as the index into asset arrays:

```cpp
enum class SimObject : uint32_t {
    Cube  = 0,  // Array index 0
    Wall  = 1,  // Array index 1  
    Agent = 2,  // Array index 2
    Plane = 3,  // Array index 3
};
```

### 3. ObjectManager Structure

Container for all physics asset data:

```cpp
struct ObjectManager {
    CollisionPrimitive *collisionPrimitives;  // Collision shapes
    AABB *primitiveAABBs;                     // Bounding boxes for primitives
    AABB *rigidBodyAABBs;                     // Bounding boxes for whole objects
    uint32_t *rigidBodyPrimitiveOffsets;      // Where each object's primitives start
    uint32_t *rigidBodyPrimitiveCounts;       // How many primitives per object
    RigidBodyMetadata *metadata;              // Mass, friction data per object
};
```

### 4. ObjectData Singleton

Per-world singleton that points to the shared ObjectManager:

```cpp
struct ObjectData {
    ObjectManager *mgr;  // Points to shared GPU memory
};
```

## Loading Process

### Phase 1: CPU Loading
1. `PhysicsLoader` reads .obj files from disk
2. Builds asset arrays in CPU memory
3. Calculates bounding boxes and collision primitives

### Phase 2: GPU Upload
```cpp
// For CUDA execution mode
PhysicsLoader phys_loader(ExecMode::CUDA, max_objects);
loadPhysicsObjects(phys_loader);  // Populates arrays

// PhysicsLoader internally calls cudaMemcpy to upload:
cudaMemcpy(gpu_metadata, cpu_metadata, size, cudaMemcpyHostToDevice);
cudaMemcpy(gpu_aabbs, cpu_aabbs, size, cudaMemcpyHostToDevice);
// ... etc for all arrays
```

### Phase 3: Runtime Access
```cpp
// Each world gets ObjectData singleton pointing to shared data
ctx.singleton<ObjectData>().mgr = &shared_object_manager;
```

## Runtime Access Pattern

### Entity Creation
```cpp
// During level generation
Entity wall = ctx.makeRenderableEntity<PhysicsEntity>();
ctx.get<ObjectID>(wall) = ObjectID{(int32_t)SimObject::Wall};
```

### Physics System Access
```cpp
// Direct array indexing using ObjectID
const ObjectManager &obj_mgr = *ctx.singleton<ObjectData>().mgr;
const RigidBodyMetadata &metadata = obj_mgr.metadata[obj_id.idx];
AABB obj_aabb = obj_mgr.rigidBodyAABBs[obj_id.idx];
```

## Memory Efficiency

### Shared Asset Data
- **Single copy** of all asset data in GPU memory
- **All worlds** share the same ObjectManager pointer
- **No duplication** across parallel simulations
- **Read-only** access ensures thread safety

### Direct Indexing Benefits
- **O(1) lookup** - no hash maps or searches
- **Cache-friendly** - sequential array access
- **GPU-optimized** - no branching or indirection
- **Type-safe** - SimObject enum prevents invalid indices

## NVRTC Compatibility

The asset system is designed to work within NVRTC constraints:

### What Works
- POD struct definitions (ObjectManager, ObjectID)
- Direct array indexing
- Singleton access via Context
- Enum-based indexing

### What Doesn't Work
- STL containers (std::vector, std::unordered_map)
- File I/O operations
- Dynamic memory allocation
- Virtual functions or inheritance

## Adding New Assets

To add a new asset type:

1. **Add to SimObject enum** (`sim.hpp`):
```cpp
enum class SimObject : uint32_t {
    // ... existing objects
    NewObject = 4,
};
```

2. **Add descriptor** (`asset_descriptors.cpp`):
```cpp
const PhysicsAssetDescriptor PHYSICS_ASSETS[] = {
    // ... existing assets
    { .name = "new_object", .objectId = SimObject::NewObject, ... }
};
// Update NUM_PHYSICS_ASSETS
```

3. **Load the asset** (`mgr.cpp`):
```cpp
// The loader will automatically pick up the new descriptor
loadPhysicsObjects(phys_loader);
```

4. **Use in level generation**:
```cpp
Entity new_obj = ctx.makeRenderableEntity<PhysicsEntity>();
ctx.get<ObjectID>(new_obj) = ObjectID{(int32_t)SimObject::NewObject};
```

## Performance Considerations

### GPU Memory Layout
- Arrays are tightly packed for coalesced memory access
- Aligned to GPU cache lines
- Primitives grouped by object for spatial locality

### Access Patterns
- Sequential array traversal in physics systems
- Direct indexing avoids pointer chasing
- Shared read-only data eliminates cache coherency issues

## Related Documentation

- [ECS Architecture](ECS_ARCHITECTURE.md) - Overall entity component system design
- [Physics System](COLLISION_SYSTEM.md) - How assets are used in collision detection
- [Asset Loading](../development/ASSET_LOADING.md) - Detailed loading implementation