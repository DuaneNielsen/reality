# Plan: Horizontal Lidar (128-beam) Pytest Implementation

## Key Files to Review Before Implementation

### C++ Files (Phase 1)
- **`src/consts.hpp`** - Camera FOV constants, add lidar-specific FOV
- **`src/mgr.hpp`** - Manager configuration struct, add custom FOV field
- **`src/level_gen.cpp`** - Agent camera attachment, modify FOV parameter
- **`external/madrona/src/render/ecs_system.cpp`** - FOV calculation implementation

### Python Files (Phase 2 & 4)  
- **`madrona_escape_room/manager.py`** - SimManager class and configuration
- **`tests/python/conftest.py`** - Test fixtures, understand cpu_manager pattern
- **`tests/python/test_depth_sensor.py`** - Existing depth sensor tests for reference
- **`madrona_escape_room/level_compiler.py`** - ASCII to CompiledLevel conversion

### Reference Files
- **`docs/sensors/DEPTH_SENSOR_QUICKSTART.md`** - Current depth sensor documentation
- **`tests/python/test_level_compiler.py`** - Examples of creating test levels
- **`src/types.hpp`** - Manager configuration struct definitions

## Overview
Create a pytest that validates 128 horizontal lidar beams with 120° FOV, using a simple wall-in-front test scenario.

## Phase 1: C++ Configuration Changes
**Goal:** Enable configurable vertical FOV for lidar setup

### 1.1 Update Constants (src/consts.hpp)
- Add new constant: `cameraFovYDegreesLidar = 1.55f` (calculated for 120° horizontal with 128:1 aspect)
- Keep existing `cameraFovYDegrees = 100.0f` as default

### 1.2 Update Manager Configuration (src/mgr.hpp)
- Add optional field: `float customVerticalFov = 0.0f` (0 = use default)
- Allow tests to override the vertical FOV

### 1.3 Update Level Generation (src/level_gen.cpp)  
- Modify `attachEntityToView()` call to use custom FOV if specified
- `attachEntityToView(ctx, agent, mgr_cfg.customVerticalFov > 0 ? mgr_cfg.customVerticalFov : consts::rendering::cameraFovYDegrees, ...)`

## Phase 2: Python Interface Updates (HARDCODED FOR POC)
**Goal:** Hardcode configuration instead of full interface

### 2.1 Hardcode Configuration in Test
- Instead of exposing through Python interface, hardcode the lidar settings directly in the test
- Use existing manager configuration but override batch render dimensions
- Set `config.batchRenderViewWidth = 128` and `config.batchRenderViewHeight = 1` in test setup

### 2.2 Skip C API Changes for Now
- Don't modify C API bindings - just hardcode values in C++ for this POC
- Focus on proving the concept works before building full interface

## Phase 3: Test Level Creation
**Goal:** Simple test scenario with predictable geometry

### 3.1 Create Lidar Test Level
- **Layout**: Agent at (0, 5, 0) facing north (0, 10, 0)
- **Wall**: Vertical wall at y=10 from x=-10 to x=+10 
- **Expected**: Wall fills horizontal FOV from -60° to +60°
- **Distance**: 5 units from agent to wall (constant depth reading)

### 3.2 Level Compiler Integration
- Create ASCII level string for the test scenario
- Use existing level compiler to generate CompiledLevel

## Phase 4: Pytest Implementation
**Goal:** Validate lidar beam calculations

### 4.1 Test Structure (`tests/python/test_horizontal_lidar.py`)
```python
@pytest.mark.depth_sensor  
def test_128_beam_horizontal_lidar():
    # Hardcode lidar configuration
    config = Manager.Config()
    config.batchRenderViewWidth = 128
    config.batchRenderViewHeight = 1
    config.enableBatchRenderer = True
    
    # Setup manager with lidar test level
    # Verify tensor shape: (worlds, agents, 1, 128, 1)
    # Check that center beams hit wall at expected distance
    # Verify edge beams see expected geometry
```

### 4.2 Hardcoded Setup
- Create manager directly with hardcoded lidar settings
- Skip custom fixture for now - just hardcode everything in test function
- Load lidar test level directly

### 4.3 Validation Logic
- **Depth readings**: Center beams (angles ~0°) should read ~5.0 units
- **Edge beams**: Beams at ±60° should also hit wall 
- **Angular mapping**: Verify beam[64] corresponds to forward direction
- **Consistency**: All beams should read approximately same distance (wall is straight)

## Phase 5: Expected Calculations
**Goal:** Predict exact lidar readings

### 5.1 Geometry Math
- Agent at (0, 5, 0), wall at y=10
- Distance from agent to any point on wall: sqrt((x-0)² + (10-5)²) = sqrt(x² + 25)
- For wall from x=-10 to +10: minimum distance = 5.0 units (at x=0)

### 5.2 Beam Angle Mapping
- 128 beams across 120° FOV
- Beam angles: -60° + (i * 120/128) for i in [0,127]
- Center beam (index 64): 0° angle, should hit wall at (0, 10, 0)

### 5.3 Test Assertions
- `assert depth_tensor.shape == (1, 1, 1, 128, 1)`
- `assert abs(depth_readings[64] - 5.0) < 0.1`  # Center beam
- `assert all(4.8 < depth < 6.2 for depth in depth_readings)`  # All beams hit wall

## Phase 6: Integration & Testing
**Goal:** Ensure everything works together

### 6.1 Build & Test
- Build project with new changes
- Run pytest to validate functionality  
- Debug any issues with FOV calculations or tensor shapes

### 6.2 Documentation
- Add comments explaining the lidar configuration
- Document the relationship between vertical FOV and horizontal beam spread

## Success Criteria
1. ✅ Pytest runs without errors
2. ✅ Tensor shape is exactly (1, 1, 1, 128, 1)  
3. ✅ Depth readings match geometric predictions (±10% tolerance)
4. ✅ Center beam reads ~5.0 units (straight-ahead distance)
5. ✅ All 128 beams provide meaningful range data

## POC Notes
- This is a proof-of-concept focusing on validating the horizontal lidar concept
- Phase 2 is intentionally hardcoded to avoid extensive interface changes
- Can be properly architected later once the concept is proven
- Focus is on mathematical validation of the 128-beam horizontal scanning