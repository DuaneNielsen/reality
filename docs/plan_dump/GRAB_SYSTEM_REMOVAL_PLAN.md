# Plan to Remove Grab System from Madrona Escape Room

## Overview
The grab system allows agents to pick up and carry objects (like cubes) by casting a ray and creating physics constraints. It consists of:
- `GrabState` component tracking constraint entities
- `grabSystem` that handles ray casting and constraint creation
- Action field `grab` (0 = do nothing, 1 = grab/release)
- Observation fields `isGrabbing` in both self and partner observations

## Step-by-Step Removal Plan

### 1. Update types.hpp
- Remove `GrabState` struct (lines 141-144)
- Remove `grab` field from `Action` struct (line 44)
- Remove `isGrabbing` field from `SelfObservation` struct (line 74)
- Remove `isGrabbing` field from `PartnerObservation` struct (line 88)
- Remove `GrabState` from Agent archetype (line 195)
- Update static assertion for `PartnerObservations` size from 3 floats to 2 floats (line 97)

### 2. Update sim.cpp
- Remove `GrabState` component registration (line 47)
- Remove entire `grabSystem` function (lines 180-243)
- Update `collectObservationsSystem`:
  - Remove `const GrabState &grab` parameter (line 309)
  - Remove grab state check for self observation (lines 327-328)
  - Remove `GrabState other_grab` and partner grab check (lines 337, 342-343)

### 3. Update Task Graph in setupTasks()
- Remove `grab_sys` node creation (lines 455-463)
- Update physics system dependency from `{grab_sys}` to `{broadphase_setup_sys}` (line 466)

### 4. Update level_gen.cpp
- Remove GrabState initialization (line 168)
- Remove grab constraint cleanup in `resetPersistentEntities()` (lines 225-229)
- Remove `grab` field from Action initialization in `resetPersistentEntities()` (line 243)

### 5. Update Action Space in Python
- Modify policy.py to change action dimensions from `[4, 8, 5, 2]` to `[4, 8, 5]` (line 126)
- This removes the grab action while keeping move amount, move angle, and rotate

### 6. Update mgr.cpp
- Remove `.grab = grab,` from Action struct initialization in `setAction()` (line 770)
- Note: Keep the `grab` parameter in the function signature for API compatibility

### 7. Update Viewer (if needed)
- Check if viewer prints grab state and remove if present

## Impact Analysis

### Components to Remove:
- `GrabState` struct (constraint tracking)
- `grabSystem` function (ray casting and constraint creation)
- `isGrabbing` observation fields (2 occurrences)
- `grab` action field

### Dependencies:
- Physics system depends on grab system output (constraint entities)
- Observation system reads grab state
- No other systems depend on grab functionality

### Python/Training Impact:
- Action space reduces from 4 dimensions to 3
- Observation space reduces by 2 float values (self and partner isGrabbing)
- Existing trained models will be incompatible

### Gameplay Impact:
- Agents can no longer pick up and move blocks
- Simplifies puzzle-solving to only button pressing
- May make some room types impossible to solve if they require moving blocks

## Verification Steps
1. Build project to ensure no compilation errors:
   ```bash
   cd build && make -j$(nproc)
   ```
2. Run simulation to verify agents can still move:
   ```bash
   ./build/headless CPU 4 100
   ```
3. Run Python binding unit tests:
   ```bash
   uv run --extra test pytest test_bindings.py -v --tb=short
   ```
   - Tests will verify tensor shapes match expected dimensions
   - Tests will check that removed components (grab, isGrabbing) are no longer present
   - May need to update test_bindings.py if it checks for specific tensor shapes
4. Check that action tensor has correct shape (3 dimensions instead of 4)
5. Verify training still works with new action/observation space (if needed)