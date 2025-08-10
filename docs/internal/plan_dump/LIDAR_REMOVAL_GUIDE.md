# Step-by-Step Guide to Remove Lidar System from Madrona Escape Room

This guide documents the complete process of removing the lidar system from the Madrona Escape Room simulation. The lidar system provides 360-degree depth sensing with 30 samples in a circle around each agent.

## Overview

The lidar system consists of:
- 360-degree depth sensing using raycasting through the BVH (Bounding Volume Hierarchy)
- 30 samples in a circle around each agent
- Depth and entity type information for each sample
- GPU-optimized warp-level implementation
- Exported tensor for Python access

## Step-by-Step Removal Process

### Step 1: Remove Lidar Components from types.hpp

**File**: `src/types.hpp`

1. **Remove LidarSample struct** (lines 120-123):
   ```cpp
   // DELETE THIS:
   struct LidarSample {
       float depth;
       float encodedType;
   };
   ```

2. **Remove Lidar struct** (lines 127-129):
   ```cpp
   // DELETE THIS:
   struct Lidar {
       LidarSample samples[consts::numLidarSamples];
   };
   ```

3. **Remove Lidar from Agent archetype** (line 213):
   ```cpp
   // Change from:
   SelfObservation, PartnerObservations, RoomEntityObservations, Lidar, StepsRemaining,
   // To:
   SelfObservation, PartnerObservations, RoomEntityObservations, StepsRemaining,
   ```

### Step 2: Remove Lidar from Simulation Enums (sim.hpp)

**File**: `src/sim.hpp`

1. **Remove Lidar from ExportID enum** (line 31):
   ```cpp
   enum class ExportID : uint32_t {
       Reset,
       Action,
       Reward,
       Done,
       SelfObservation,
       PartnerObservations,
       RoomEntityObservations,
       Lidar,  // DELETE THIS LINE
       StepsRemaining,
       NumExports,
   };
   ```

### Step 3: Update Component Registration (sim.cpp)

**File**: `src/sim.cpp`

1. **Remove Lidar component registration** in `registerTypes()` (around line 52):
   ```cpp
   // DELETE THIS LINE:
   registry.registerComponent<Lidar>();
   ```

2. **Remove Lidar export** (lines 85-86):
   ```cpp
   // DELETE THESE LINES:
   registry.exportColumn<Agent, Lidar>(
       (uint32_t)ExportID::Lidar);
   ```

### Step 4: Remove Lidar System (sim.cpp)

**File**: `src/sim.cpp`

1. **Remove the entire lidarSystem function** (lines 374-433):
   ```cpp
   // DELETE THIS ENTIRE FUNCTION:
   // [GAME_SPECIFIC] Launches consts::numLidarSamples per agent.
   inline void lidarSystem(Engine &ctx,
                           Entity e,
                           Lidar &lidar)
   {
       // ... entire function body ...
   }
   ```

2. **Note**: Keep the `encodeType()` function (lines 293-296) as it's still used by room entity observations

### Step 5: Update Task Graph (sim.cpp)

**File**: `src/sim.cpp` in `setupTasks()` function

1. **Remove lidar system node creation** (lines 593-606):
   ```cpp
   // DELETE THIS:
   #ifdef MADRONA_GPU_MODE
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

2. **Update GPU sorting dependencies** (line 618):
   ```cpp
   // Change from:
   auto sort_agents = queueSortByWorld<Agent>(
       builder, {lidar, collect_obs});
   // To:
   auto sort_agents = queueSortByWorld<Agent>(
       builder, {collect_obs});
   ```

3. **Remove unused variable suppression** (line 623):
   ```cpp
   // DELETE THIS LINE:
   (void)lidar;
   ```

### Step 6: Remove Lidar Constant (consts.hpp)

**File**: `src/consts.hpp`

1. **Remove numLidarSamples constant** (line 42):
   ```cpp
   // DELETE THIS LINE:
   inline constexpr madrona::CountT numLidarSamples = 30;
   ```

### Step 7: Update Manager Interface (mgr.hpp)

**File**: `src/mgr.hpp`

1. **Remove lidarTensor declaration** (line 45):
   ```cpp
   // DELETE THIS LINE:
   madrona::py::Tensor lidarTensor() const;
   ```

### Step 8: Update Manager Implementation (mgr.cpp)

**File**: `src/mgr.cpp`

1. **Remove lidarTensor function** (lines 695-703):
   ```cpp
   // DELETE THIS ENTIRE FUNCTION:
   Tensor Manager::lidarTensor() const
   {
       return impl_->exportTensor(ExportID::Lidar,
                                  TensorElementType::Float32,
                                  {
                                      impl_->cfg.numWorlds,
                                      consts::numAgents,
                                      consts::numLidarSamples,
                                      2,
                                  });
   }
   ```

### Step 9: Update Python Bindings (bindings.cpp)

**File**: `src/bindings.cpp`

1. **Remove lidar_tensor binding** (line 48):
   ```cpp
   // DELETE THIS LINE:
   .def("lidar_tensor", &Manager::lidarTensor)
   ```

### Step 10: Update Viewer (viewer.cpp)

**File**: `src/viewer.cpp`

1. **Remove lidar printer creation** (line 139):
   ```cpp
   // DELETE THIS LINE:
   auto lidar_printer = mgr.lidarTensor().makePrinter();
   ```

2. **Remove lidar observation printing** (around lines 155-156):
   ```cpp
   // DELETE THESE LINES:
   printf("Lidar\n");
   lidar_printer.print();
   ```

### Step 11: Update Unit Tests (test_bindings.py)

**File**: `test_bindings.py`

1. **Remove lidar tensor shape test** in `test_tensor_shapes()` (lines 111-112):
   ```python
   # DELETE THESE LINES:
   lidar = mgr.lidar_tensor().to_torch()
   assert lidar.shape == (4, 2, 30, 2)  # 30 lidar samples, 2 values each
   ```

2. **Remove lidar from tensor memory layout test** in `test_tensor_memory_layout()` (line 182):
   ```python
   # Remove from tensors_to_check list:
   ("lidar", mgr.lidar_tensor().to_torch()),
   ```

3. **Remove lidar observation value tests** in `test_observation_values()` (lines 245-247):
   ```python
   # DELETE THESE LINES:
   # Lidar should have normalized values
   lidar = mgr.lidar_tensor().to_torch()
   assert lidar.min() >= 0, "Lidar values should be non-negative"
   assert lidar.max() <= 1, "Lidar values should be normalized"
   ```

## Important Notes

### What to Keep

1. **EntityType enum**: Still used by room entity observations
2. **encodeType() function**: Still used by room entity observations
3. **BVH/broadphase setup**: Used by other systems

### Dependencies

The lidar system has no dependent systems, making it safe to remove without affecting other functionality. It only depends on:
- EntityType enum (shared with room observations)
- BVH for raycasting (shared infrastructure)

## Verification Steps

After completing all changes:

1. **Build the project** to ensure no compilation errors:
   ```bash
   cd build && make -j8
   ```

2. **Run the simulation** to verify lidar is removed:
   ```bash
   ./build/headless CPU 4 100
   ```

3. **Run unit tests** to ensure all tests pass:
   ```bash
   uv run --extra test pytest test_bindings.py -v --tb=short
   ```

4. **Test Python access** (if PyTorch is available):
   ```python
   # Should no longer have lidar_tensor() method
   from madrona_escape_room import EscapeRoomSimulator
   sim = EscapeRoomSimulator(num_worlds=1)
   # sim.lidar_tensor() should raise AttributeError
   ```

5. **Verify room observations** still work correctly

## Summary

This removal eliminates:
- 2 component structs (LidarSample, Lidar)
- 1 system function (lidarSystem)
- 1 constant (numLidarSamples)
- 1 export tensor (lidarTensor)
- 1 enum value (ExportID::Lidar)
- Associated Python bindings and viewer code

The result is a simpler simulation without the computational overhead of 30 raycasts per agent per frame, while maintaining all other observation systems.