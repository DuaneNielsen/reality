# Stress Test Setup Summary

## Date: 2025-08-26
## Initial Commit: 62b92dc ("feat: add compile_ascii_level() convenience wrapper")
## Current Commit: 335b4ce ("refactor: clean up Python bindings architecture and fix constant dependencies")

## Objective
Run the stress test script `scratch/stress_test_manager_lifecycle.py` to reproduce a segmentation fault in SimManager lifecycle.

## Changes Required

### 1. Build Configuration
- **Issue**: Initial build failed due to missing Madrona toolchain
- **Fix**: Reconfigured CMake to use Madrona toolchain:
  ```bash
  cmake -B build -DCMAKE_TOOLCHAIN_FILE=external/madrona/cmake/madrona-toolchain.cmake
  ```

### 2. Python Bindings Generation
- **Issue**: `uv run` during bindings generation triggered unwanted package builds
- **Fix**: Created isolated environment for code generation:
  - Created `codegen/pyproject.toml` with only libclang dependency
  - Modified `src/CMakeLists.txt` to use `--project` flag:
    ```cmake
    ${UV_EXECUTABLE} run --project ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/pyproject.toml
    ```

### 3. DLPack Extension Compilation
- **Issue**: `dlpack_extension.cpp` failed to compile with system g++ (missing `_LIBCPP_VERSION`)
- **Root Cause**: Madrona headers require clang's libc++, incompatible with g++'s libstdc++
- **Fix**: Disabled dlpack extension in `setup.py`:
  ```python
  # ext_modules=[dlpack_extension],  # Disabled for now - build issues with mixed toolchains
  ```

### 4. Stress Test Script Updates
- **Issue 1**: Import error - `madrona` module doesn't exist
- **Fix**: Changed import from `madrona` to use correct module:
  ```python
  from madrona_escape_room import SimManager, ExecMode, create_default_level
  ```

- **Issue 2**: Missing level parameter causing "No compiled level provided" error
- **Fix**: Added default level to SimManager constructor:
  ```python
  mgr = SimManager(
      exec_mode=ExecMode.CPU,
      gpu_id=0,
      num_worlds=1,
      rand_seed=42,
      auto_reset=True,
      compiled_levels=create_default_level(),  # Added this line
  )
  ```

## Result
Successfully reproduced the segmentation fault:
- Test ran successfully past iteration 200
- Crashed between iterations 200-250 (consistent with expected range of 165-216)
- Segfault occurred in `lib.mer_create_manager()` at manager.py:79
- Confirms the issue is in repeated creation/destruction of SimManager instances

## Technical Notes
- The dlpack extension provides PyTorch tensor interoperability via DLPack protocol
- Disabling it means `tensor.to_torch()` won't work, but doesn't affect basic simulation
- The toolchain mismatch issue stems from Madrona using custom clang++ with libc++ while system uses g++ with libstdc++
- Future fix would require modifying `setup.py` to consistently use Madrona's clang++ for all extensions

## Required Constants for Stress Test

When `generated_constants.py` is missing or incomplete, these constants need to be hardcoded:

### Core Enums
```python
# Execution Mode (from madrona)
class ExecMode:
    CPU = 0
    CUDA = 1

# Tensor Element Types (from madrona.py)
class TensorElementType:
    UInt8 = 0
    Int8 = 1
    Int16 = 2
    Int32 = 3
    Int64 = 4
    Float16 = 5
    Float32 = 6

# Entity Types (from madEscape)
class EntityType:
    NoEntity = 0
    Cube = 1
    Wall = 2
    Agent = 3
    NumTypes = 4

# Physics Response Types (from madrona.phys)
class ResponseType:
    Dynamic = 0
    Kinematic = 1
    Static = 2
```

### Action Constants (from madEscape::consts::action)
```python
class action:
    class move_amount:
        STOP = 0
        SLOW = 1
        MEDIUM = 2
        FAST = 3
    
    class move_angle:
        FORWARD = 0
        FORWARD_RIGHT = 1
        RIGHT = 2
        BACKWARD_RIGHT = 3
        BACKWARD = 4
        BACKWARD_LEFT = 5
        LEFT = 6
        FORWARD_LEFT = 7
    
    class rotate:
        FAST_LEFT = 0
        SLOW_LEFT = 1
        NONE = 2
        SLOW_RIGHT = 3
        FAST_RIGHT = 4
```

### Game Constants (from madEscape::consts)
```python
class consts:
    # World dimensions
    worldLength = 40.0
    worldWidth = 20.0
    wallWidth = 1.0
    agentRadius = 1.0
    
    # Rewards
    rewardPerDist = 0.05
    slackReward = -0.005
    
    # Episode settings
    episodeLen = 200
    deltaT = 0.04
    
    # Nested namespaces
    class limits:
        maxTiles = 1024
        maxSpawns = 8
        maxNameLength = 64
        maxLevelNameLength = 64
        maxWorlds = 10000
        maxAgentsPerWorld = 100
        maxGridSize = 64
        maxScale = 100.0
        maxCoordinate = 1000.0
    
    class physics:
        gravityAcceleration = 9.8
        cubeInverseMass = 0.075
        wallInverseMass = 0.0
        agentInverseMass = 1.0
        planeInverseMass = 0.0
        
        class objectIndex:
            cube = 0
            wall = 1
            agent = 2
            plane = 3
    
    class math:
        pi = 3.14159265359
        degreesInHalfCircle = 180.0
    
    class rendering:
        # Material indices
        class materialIndex:
            cube = 0
            wall = 1
            agentBody = 2
            agentParts = 3
            floor = 4
            axisX = 5
            button = 6
            axisY = 7
            axisZ = 8
```

## Testing Results Across Commits

- **Commit 62b92dc**: ❌ Segfault between iterations 200-250 (original issue)
- **Commit 335b4ce**: ❌ Illegal instruction crash at iteration ~50 (Current - 2025-08-26)
- **Commit 4775fa5**: ✅ No segfault - 300 iterations completed  
- **Commit 59cafd9**: ✅ No segfault - 300 iterations completed
- **Commit 1fcdc62**: ✅ No segfault - 300 iterations completed

The crash appears to be introduced between commits 1fcdc62 and 335b4ce. The issue manifests as either a segfault or illegal instruction error, both occurring in `lib.mer_create_manager()` at manager.py:79 during repeated SimManager creation/destruction cycles.