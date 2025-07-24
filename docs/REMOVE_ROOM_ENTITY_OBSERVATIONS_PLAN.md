# Plan to Remove RoomEntityObservations

## Overview
Remove the RoomEntityObservations system which currently tracks up to 15 entities per room (mostly cubes) that serve no gameplay purpose since interaction mechanics have been removed.

## Changes Required

### 1. Remove from Types (src/types.hpp)
- Delete `RoomEntityObservations` struct definition (lines ~93-99)
- Remove `RoomEntityObservations` from Agent archetype (line ~172)

### 2. Remove from Simulation (src/sim.cpp)
- Remove component registration: `registerComponent<RoomEntityObservations>()` (line ~48)
- Remove export registration: `exportColumn<Agent, RoomEntityObservations>` (lines ~77-78)
- Update `collectObservationsSystem`:
  - Remove `RoomEntityObservations &room_ent_obs` parameter (line ~241)
  - Remove the entire loop that populates room entity observations (lines ~261-278)
- Update task graph in `setupTasks`:
  - Remove `RoomEntityObservations` from collectObservationsSystem node (line ~424)

### 3. Remove Export ID (src/sim.hpp)
- Remove `RoomEntityObservations` from ExportID enum (line ~29)
- Update `NumExports` count accordingly

### 4. Remove from Manager (src/mgr.cpp)
- Remove `roomEntityObservationsTensor()` method implementation (lines ~669-677)

### 5. Remove from Bindings (src/bindings.cpp)
- Remove `.def("room_entity_observations_tensor", ...)` binding (line ~45)

### 6. Update Python Code (scripts/policy.py)
- Remove `room_ent_obs_tensor = sim.room_entity_observations_tensor().to_torch()` (line ~18)
- Remove `room_ent_obs_tensor.view(...)` from obs_tensors list (line ~38)

### 7. Update Tests (test_bindings.py)
- Remove room observation tensor test (line ~101)
- Remove any assertions about room observation shape

### 8. Clean Up Entity Type System
Since we're removing entity observations, we can also simplify:
- Remove `EntityType` enum values for Button, Cube, Door (keep only None, Agent, Wall)
- Remove `encodeType()` function as it's only used for room entity observations

## Benefits
- Removes ~45 floats per agent per frame from observation space
- Eliminates unnecessary polar coordinate calculations
- Simplifies the observation pipeline
- Removes confusion about non-functional game objects
- Reduces neural network input size