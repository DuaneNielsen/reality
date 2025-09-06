# Lidar System Code Guide for Madrona Escape Room

## Overview

This guide explains the implementation details of the lidar system in the Madrona Escape Room. The lidar system provides 360° depth sensing for agents using ray tracing, with GPU optimization for high-performance simulation.

## Architecture

The lidar system follows Madrona's Entity Component System (ECS) architecture with these key components:

- **Component**: `Lidar` - stores 30 depth samples in a circle around each agent
- **System**: `lidarSystem()` - traces rays and updates lidar data each frame  
- **Export**: Tensor interface for Python ML training integration
- **GPU Optimization**: Warp-level parallelism with 32 threads per agent

## Core Data Structures

### LidarSample (types.hpp:118-121)
```cpp
struct LidarSample {
    float depth;        // Normalized distance [0,1]
    float encodedType;  // Entity type encoding [0,1]
};
```

### Lidar Component (types.hpp:124-127)
```cpp
struct Lidar {
    LidarSample samples[consts::numLidarSamples];  // 30 samples in circle
};
```

### Constants (consts.hpp:40-41)
```cpp
inline constexpr madrona::CountT numLidarSamples = 30;
```

## Lidar System Implementation

### Core Ray Tracing (sim.cpp:329-384)

The `lidarSystem()` function is the heart of the lidar implementation:

```cpp
inline void lidarSystem(Engine &ctx, Entity e, Lidar &lidar)
{
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);
```

**Key Features:**
- Traces 30 rays in a perfect circle around the agent
- Uses agent's rotation to orient the rays correctly
- Leverages BVH (Bounding Volume Hierarchy) for fast ray-object intersection

### Ray Direction Calculation

```cpp
auto traceRay = [&](int32_t idx) {
    float theta = 2.f * math::pi * (
        float(idx) / float(consts::numLidarSamples)) + math::pi / 2.f;
    float x = cosf(theta);
    float y = sinf(theta);
    Vector3 ray_dir = (x * right + y * agent_fwd).normalize();
```

- **Theta calculation**: Distributes rays evenly around 360°
- **Coordinate transform**: Converts polar to Cartesian coordinates
- **Agent-relative**: Rays are oriented relative to agent's facing direction

### BVH Ray Tracing

```cpp
Entity hit_entity = bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t, 
                                &hit_normal, 200.f);
```

- **Ray origin**: Slightly above agent position (`+ 0.5f * math::up`)
- **Max distance**: 200 units maximum ray length
- **Returns**: Hit entity, distance, and surface normal

### GPU/CPU Optimization

```cpp
#ifdef MADRONA_GPU_MODE
    // GPU: Use warp-level parallelism
    int32_t idx = threadIdx.x % 32;
    if (idx < consts::numLidarSamples) {
        traceRay(idx);
    }
#else
    // CPU: Sequential loop
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i);
    }
#endif
```

**GPU Optimization:**
- Uses `CustomParallelForNode` with 32-thread warps
- Each thread handles one ray (30 active threads, 2 idle)
- Massive parallelism: all rays traced simultaneously per agent

## Helper Functions

### Distance Normalization (sim.cpp:227-232)
```cpp
static inline float distObs(float v) {
    return fminf(v / 200.f, 1.f);  // Normalize to [0,1] range
}
```

### Entity Type Encoding (sim.cpp:235-239)
```cpp
static inline float encodeType(EntityType type) {
    return (float)type / (float)EntityType::NumTypes;
}
```

**Entity Type Mappings:**
- `NoEntity` (0) → 0.0
- `Cube` (1) → 0.25  
- `Wall` (2) → 0.5
- `Agent` (3) → 0.75

## ECS Integration

### Component Registration (sim.cpp:51)
```cpp
registry.registerComponent<Lidar>();
```

### Export Registration (sim.cpp:88-89)
```cpp
registry.exportColumn<Agent, Lidar>((uint32_t)ExportID::Lidar);
```

### Task Graph Integration (sim.cpp:663-677)

The lidar system runs after BVH construction and before entity sorting:

```cpp
#ifdef MADRONA_GPU_MODE
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,  // 32 threads, 1 entity per warp
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine, lidarSystem,
#endif
            Entity, Lidar
        >>({post_reset_broadphase});
```

**Dependencies:**
- **Requires**: `post_reset_broadphase` (BVH must be built first)
- **Feeds into**: `sort_agents` (entity sorting for GPU efficiency)

## Python Integration

### Manager Interface (mgr.cpp:795-804)
```cpp
Tensor Manager::lidarTensor() const {
    return impl_->exportTensor(ExportID::Lidar, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents, 
                                   consts::numLidarSamples,
                                   2,  // depth + encodedType
                               });
}
```

**Tensor Shape**: `[worlds, agents, samples, values]`
- 4 worlds × 1 agent × 30 samples × 2 values = `(4, 1, 30, 2)`

### Python API (manager.py:137-138)
```python
def lidar_tensor(self):
    return self._get_tensor(lib.mer_get_lidar_tensor)
```

### C API (madrona_escape_room_c_api.cpp:288-298)
```cpp
MER_Result mer_get_lidar_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->lidarTensor();
    convertTensor(tensor, out_tensor);
    return MER_SUCCESS;
}
```

## Viewer Integration

### Observation Printing (viewer.cpp:455, 463-464)
```cpp
auto lidar_printer = mgr.lidarTensor().makePrinter();

// In printObs():
printf("Lidar\n");
lidar_printer.print();
```

### Key Handler (viewer.cpp:507-510)
```cpp
if (input.keyHit(Key::O)) {
    printObs();  // Print all observations including lidar
}
```

**Usage**: Press 'O' in the viewer to see lidar output in terminal.

## Testing

### Tensor Shape Test (test_bindings.py:67-69)
```python
lidar = mgr.lidar_tensor().to_torch()
assert lidar.shape == (4, 1, 30, 2)  # 30 lidar samples, 2 values each
```

### Value Range Test (test_bindings.py:214-217)
```python
lidar = mgr.lidar_tensor().to_torch()
assert lidar.min() >= 0, "Lidar values should be non-negative"
assert lidar.max() <= 1, "Lidar values should be normalized"
```

## Performance Characteristics

### GPU Performance
- **Parallelism**: 30 rays per agent traced simultaneously
- **Warp Efficiency**: 30/32 threads active (93.75% utilization)
- **Memory Access**: Coalesced reads from BVH structure
- **Throughput**: Scales linearly with agent count

### CPU Performance  
- **Sequential**: Rays traced one by one per agent
- **Vectorization**: Compiler can optimize ray direction calculations
- **Cache Friendly**: BVH traversal benefits from spatial locality

## Configuration Options

### Adjustable Parameters

**Number of Samples** (consts.hpp):
```cpp
inline constexpr madrona::CountT numLidarSamples = 30;  // Can be changed
```

**Max Ray Distance** (sim.cpp:351):
```cpp
bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t, &hit_normal, 200.f);
//                                                                  ^^^^^ Adjustable
```

**Ray Height Offset** (sim.cpp:351):
```cpp
bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t, &hit_normal, 200.f);
//                 ^^^^^^^^^^^^^^^ Adjustable height above agent
```

### GPU Warp Size Considerations

For different sample counts, adjust the CustomParallelForNode parameters:

```cpp
// For 64 samples:
auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
    lidarSystem, 64, 1,  // Use 64 threads (2 warps)

// For 128 samples:  
auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
    lidarSystem, 128, 1,  // Use 128 threads (4 warps)
```

## Troubleshooting

### Common Issues

**1. Lidar values all zero:**
- Check BVH is built before lidar system runs
- Verify task graph dependencies are correct
- Ensure entities have collision geometry

**2. Python binding errors:**
- Rebuild after C++ changes: `./build.sh`
- Check C API function is exported in header
- Verify ctypes binding matches C function signature

**3. GPU performance issues:**
- Monitor warp utilization (should be >90% for 30 samples)
- Check for divergent branches in ray tracing code
- Profile memory access patterns in BVH traversal

**4. Viewer not showing lidar:**
- Press 'O' key (not 'o') to trigger observation printing
- Check terminal output, not GUI
- Ensure lidar system is actually running (step simulation first)

## Future Improvements

### Potential Optimizations

**1. Adaptive Ray Count:**
```cpp
// Adjust sample count based on agent speed/importance
int32_t adaptive_samples = base_samples * importance_factor;
```

**2. Multi-Height Lidar:**
```cpp
// Trace rays at different heights for 3D sensing
for (float height : {0.0f, 0.5f, 1.0f}) {
    Vector3 ray_origin = pos + height * math::up;
    // ... trace rays at this height
}
```

**3. Semantic Encoding:**
```cpp
// More detailed entity type information
struct LidarSample {
    float depth;
    float material_id;   // Surface material
    float reflectance;   // Surface properties
    float velocity;      // Doppler-like effect
};
```

This lidar system provides a solid foundation for agent navigation and can be extended for more sophisticated sensing capabilities as needed.