# Lidar System Restoration Plan for Madrona Escape Room

## Overview
This document provides a detailed plan to restore the lidar system that was removed in commit b6d9649. The restoration must be done manually due to merge conflicts with newer features (compass system, StepsTaken, etc.).

## Pre-Reading Phase

### Files to Read Before Starting
1. **src/types.hpp** - Understand current component structure
2. **src/sim.cpp** - Review current systems and task graph
3. **src/sim.hpp** - Check current ExportID enum
4. **src/consts.hpp** - Verify constant locations
5. **src/mgr.cpp** - Understand tensor export pattern
6. **src/mgr.hpp** - Review current manager interface
7. **src/viewer.cpp** - Check current observation printing
8. **tests/python/test_bindings.py** - Review test structure

## Detailed Restoration Steps with Code Blocks

### Step 1: Add Lidar Constant to consts.hpp

**Location**: After line 35 (after the action bucket definitions)

```cpp
// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;
```

### Step 2: Add Lidar Structures to types.hpp

**Location**: After line 109 (after the StepsTaken struct)

```cpp
// [GAME_SPECIFIC]
struct LidarSample {
    float depth;
    float encodedType;
};

// [GAME_SPECIFIC]
// Linear depth values and entity type in a circle around the agent
struct Lidar {
    LidarSample samples[consts::numLidarSamples];
};
```

**Also modify Agent archetype** (around line 317):
Change from:
```cpp
SelfObservation, CompassObservation, StepsTaken,
```
To:
```cpp
SelfObservation, CompassObservation, Lidar, StepsTaken,
```

### Step 3: Update ExportID Enum in sim.hpp

**Location**: In the ExportID enum (around line 29)

Change from:
```cpp
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    SelfObservation,
    CompassObservation,
    StepsTaken,
    Progress,
    NumExports,
};
```

To:
```cpp
enum class ExportID : uint32_t {
    Reset,
    Action,
    Reward,
    Done,
    SelfObservation,
    CompassObservation,
    Lidar,
    StepsTaken,
    Progress,
    NumExports,
};
```

### Step 4: Register and Export Lidar Component in sim.cpp

**Location 1**: In registerTypes() function (after line 52)
```cpp
registry.registerComponent<Lidar>();
```

**Location 2**: In registerTypes() exports section (after line 93, after CompassObservation export)
```cpp
registry.exportColumn<Agent, Lidar>(
    (uint32_t)ExportID::Lidar);
```

### Step 5: Add Lidar System Function to sim.cpp

**Location**: After the compassSystem function (around line 325)

```cpp
// [GAME_SPECIFIC] Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
inline void lidarSystem(Engine &ctx,
                        Entity e,
                        Lidar &lidar)
{
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);

    auto traceRay = [&](int32_t idx) {
        float theta = 2.f * math::pi * (
            float(idx) / float(consts::numLidarSamples)) + math::pi / 2.f;
        float x = cosf(theta);
        float y = sinf(theta);

        Vector3 ray_dir = (x * right + y * agent_fwd).normalize();

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(pos + 0.5f * math::up, ray_dir, &hit_t,
                         &hit_normal, 200.f);

        if (hit_entity == Entity::none()) {
            lidar.samples[idx] = {
                .depth = 0.f,
                .encodedType = encodeType(EntityType::None),
            };
        } else {
            EntityType entity_type = ctx.get<EntityType>(hit_entity);

            lidar.samples[idx] = {
                .depth = distObs(hit_t),
                .encodedType = encodeType(entity_type),
            };
        }
    };


    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for 
    // warp level programming
    int32_t idx = threadIdx.x % 32;

    if (idx < consts::numLidarSamples) {
        traceRay(idx);
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i);
    }
#endif
}
```

### Step 6: Add Lidar to Task Graph in sim.cpp

**Location**: In setupTasks() function, after compass_sys (around line 664)

```cpp
// [GAME_SPECIFIC] The lidar system
#ifdef MADRONA_GPU_MODE
    // [BOILERPLATE] Note the use of CustomParallelForNode to create a taskgraph node
    // that launches a warp of threads (32) for each invocation (1).
    // The 32, 1 parameters could be changed to 32, 32 to create a system
    // that cooperatively processes 32 entities within a warp.
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 32, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            Lidar
        >>({post_reset_broadphase});
```

**Also update the sort dependencies** (around line 676):
Change from:
```cpp
auto sort_agents = queueSortByWorld<Agent>(
    builder, {compass_sys});
```
To:
```cpp
auto sort_agents = queueSortByWorld<Agent>(
    builder, {compass_sys, lidar});
```

**And update the void cast** (around line 681):
Change from:
```cpp
(void)compass_sys;
```
To:
```cpp
(void)compass_sys;
(void)lidar;
```

### Step 7: Add Lidar Tensor Method to mgr.hpp

**Location**: After progressTensor() declaration (around line 159)

```cpp
madrona::py::Tensor lidarTensor() const;
```

### Step 8: Implement Lidar Tensor in mgr.cpp

**Location**: After progressTensor() implementation (around line 806)

```cpp
//[GAME_SPECIFIC]
Tensor Manager::lidarTensor() const
{
    return impl_->exportTensor(ExportID::Lidar, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   consts::numLidarSamples,
                                   2,
                               });
}
```

### Step 9: Create Python Bindings File

**Create new file**: src/bindings.cpp

```cpp
#include <nanobind/nanobind.h>
#include <nanobind/tensor.h>

#include "mgr.hpp"

namespace nb = nanobind;

namespace madEscape {

NB_MODULE(madrona_escape_room_python_example, m) {
    nb::class_<Manager>(m, "Manager")
        .def(nb::init<Manager::Config>())
        .def("step", &Manager::step)
        .def("reset_tensor", &Manager::resetTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("self_observation_tensor", &Manager::selfObservationTensor)
        .def("compass_tensor", &Manager::compassTensor)
        .def("lidar_tensor", &Manager::lidarTensor)
        .def("steps_taken_tensor", &Manager::stepsTakenTensor)
        .def("progress_tensor", &Manager::progressTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("trigger_reset", &Manager::triggerReset)
        .def("set_action", &Manager::setAction);
}

}
```

### Step 10: Update Viewer to Show Lidar

**Location 1**: In viewer.cpp, add printer creation (after line 455)
```cpp
auto lidar_printer = mgr.lidarTensor().makePrinter();
```

**Location 2**: In printObs function (after steps_taken print, around line 463)
```cpp
printf("Lidar\n");
lidar_printer.print();
```

### Step 11: Add Tests to test_bindings.py

**Location 1**: In test_tensor_shapes() function
```python
# Test lidar tensor shape
lidar = mgr.lidar_tensor().to_torch()
assert lidar.shape == (4, 2, 30, 2)  # 30 lidar samples, 2 values each
```

**Location 2**: In test_observation_values() function
```python
# Lidar should have normalized values
lidar = mgr.lidar_tensor().to_torch()
assert lidar.min() >= 0, "Lidar values should be non-negative"
assert lidar.max() <= 1, "Lidar values should be normalized"
```

## Build and Verification Steps

1. **Compile the project**:
   ```bash
   ./build.sh
   ```

2. **Run C++ tests**:
   ```bash
   ./build/mad_escape_tests
   ```

3. **Run Python tests**:
   ```bash
   uv run --group dev pytest tests/python/test_bindings.py::test_tensor_shapes -v
   uv run --group dev pytest tests/python/test_bindings.py::test_observation_values -v
   ```

4. **Test with viewer**:
   ```bash
   ./build/viewer CPU 1
   ```
   Press 'o' to print observations and verify lidar output

## Important Notes

### Dependencies
- The `encodeType()` function already exists (used by room observations)
- The `distObs()` function already exists for normalization
- BVH is already set up in the task graph

### GPU Optimization
- Lidar uses CustomParallelForNode with 32 threads per agent
- Each thread traces one ray (warp-level parallelism)
- Maximum 30 samples used (threads 30-31 idle)

### Compatibility
- Preserves compass system (added after lidar removal)
- Preserves StepsTaken (added after lidar removal)
- Does not restore PartnerObservations (removed separately)
- Does not restore RoomEntityObservations (removed separately)

## Potential Issues and Solutions

1. **Build errors**: Check that all components are registered before use
2. **Missing encodeType**: This function should exist for EntityType encoding
3. **Task graph dependencies**: Ensure lidar runs after BVH build
4. **Python bindings**: May need to adjust based on current binding approach
5. **Test failures**: Update expected tensor counts in tests

## Verification Checklist

- [ ] Code compiles without errors
- [ ] All C++ tests pass
- [ ] Python tensor shape tests pass
- [ ] Lidar values are in [0, 1] range
- [ ] Viewer shows lidar observations
- [ ] No performance regression
- [ ] GPU version works correctly