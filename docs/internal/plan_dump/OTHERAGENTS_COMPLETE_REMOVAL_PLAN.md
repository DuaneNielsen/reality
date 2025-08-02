# Plan to Remove OtherAgents Component and Partner Observations

## Overview
This plan will completely remove the OtherAgents component and all partner observation functionality from the escape room simulation. This means agents will no longer observe each other.

## Components to Remove:
1. **OtherAgents** - Component storing references to other agents
2. **PartnerObservation** - Single partner observation structure  
3. **PartnerObservations** - Array of partner observations
4. **Partner observation collection** - Logic in collectObservationsSystem
5. **Python bindings** - Partner observations tensor export

## Step-by-Step Removal Plan:

### 1. Update src/types.hpp
- Remove `OtherAgents` struct definition (lines 133-135)
- Remove `PartnerObservation` struct definition (lines 84-86)
- Remove `PartnerObservations` struct definition (lines 90-92)
- Remove the static_assert for PartnerObservations size (line 97)
- Remove `OtherAgents` from Agent archetype (line 187)
- Remove `PartnerObservations` from Agent archetype (line 193)

### 2. Update src/sim.cpp
- Remove `OtherAgents` component registration (line 48)
- Remove `PartnerObservations` component registration (line 49)
- Remove `PartnerObservations` export registration (lines 79-80)
- Update `collectObservationsSystem`:
  - Remove `OtherAgents` parameter (line 244)
  - Remove `PartnerObservations` parameter (line 246)
  - Remove partner observation collection loop (lines 265-274)
- Update task graph in `setupTasks()`:
  - Remove `OtherAgents` from collectObservationsSystem inputs (line 441)
  - Remove `PartnerObservations` from outputs (line 443)

### 3. Update src/sim.hpp
- Remove `PartnerObservations` from ExportID enum (line 29)
- Adjust `NumExports` count accordingly

### 4. Update src/mgr.hpp
- Remove `partnerObservationsTensor()` method declaration (line 43)

### 5. Update src/mgr.cpp
- Remove `partnerObservationsTensor()` method implementation (lines 668-671)
- Remove any references to PartnerObservations in export handling

### 6. Update src/bindings.cpp
- Remove partner_observations_tensor binding (line 45)

### 7. Update src/viewer.cpp
- Remove partner observation printer creation (line 137)
- Remove any partner observation printing logic

### 8. Update src/level_gen.cpp
- Remove OtherAgents population code (lines 171-186)

### 9. Update scripts/policy.py
- Remove `partner_obs_tensor` from `setup_obs()` (line 18)
- Remove `partner_obs_tensor` from `obs_tensors` list (line 37)
- Update any code that depends on the observation tensor count/shape

### 10. Update Python Training Code
- Check and update any other Python scripts that might reference partner observations
- Update neural network architecture if it expects partner observations as input

## Build and Test
After making all changes:
1. Rebuild the C++ code: `cd build && make -j$(nproc)`
2. Run tests to ensure nothing is broken
3. Test the viewer to ensure visualization still works
4. Run training to verify the policy can still learn without partner observations

## Benefits
- Simplifies the codebase by removing unused observation data
- Reduces memory usage and improves performance
- Removes complexity from the observation space
- Makes the environment suitable for single-agent scenarios