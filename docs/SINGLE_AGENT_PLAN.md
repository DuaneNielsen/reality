# Plan to Reduce Number of Agents from 2 to 1

## Overview
This plan reduces the number of agents in the escape room simulation from 2 to 1. This change affects partner observations, agent initialization, and training scripts.

## Step-by-Step Removal Process

### Step 1: Update Core Constant
**File**: `src/consts.hpp`
- Change line 13: `inline constexpr madrona::CountT numAgents = 1;` (from 2)

### Step 2: Update Component Structures
**File**: `src/types.hpp`

1. **Update PartnerObservations struct** (lines 90-92):
   - The array `obs[consts::numAgents - 1]` will become size 0
   - Keep the struct but it will be empty

2. **Update static_assert** (line 97):
   ```cpp
   // Change from:
   static_assert(sizeof(PartnerObservations) == sizeof(float) * (consts::numAgents - 1) * 2);
   // To:
   static_assert(sizeof(PartnerObservations) == 0);  // No partner observations with 1 agent
   ```

3. **Update OtherAgents struct** (line 134):
   - The array `e[consts::numAgents - 1]` will become size 0
   - Keep the struct but it will be empty

### Step 3: Update Observation Collection
**File**: `src/sim.cpp`

1. **Update collectObservationsSystem** (lines 265-274):
   ```cpp
   // The loop will effectively do nothing since consts::numAgents - 1 = 0
   // Consider adding a comment or #if to make this explicit:
   #if consts::numAgents > 1
   for (CountT i = 0; i < consts::numAgents - 1; i++) {
       Entity other = other_agents.e[i];
       Vector3 other_pos = ctx.get<Position>(other);
       Vector3 to_other = other_pos - pos;
       partner_obs.obs[i] = {
           .polar = xyToPolar(to_view.rotateVec(to_other)),
       };
   }
   #endif
   ```

### Step 4: Update Reset System
**File**: `src/sim.cpp`

1. **Update resetSystem** (line 129):
   - Loop will iterate only once (i < 1)
   - No code change needed, just aware of behavior change

### Step 5: Update Level Generation
**File**: `src/level_gen.cpp`

1. **Update createPersistentEntities** (line 153):
   - Loop will create only 1 agent
   - No code change needed

2. **Update OtherAgents population** (lines 173-184):
   ```cpp
   // The outer loop runs once, inner loop doesn't execute
   // Consider adding explicit handling:
   #if consts::numAgents > 1
   for (CountT i = 0; i < consts::numAgents; i++) {
       Entity cur_agent = ctx.data().agents[i];
       OtherAgents &other_agents = ctx.get<OtherAgents>(cur_agent);
       
       CountT out_idx = 0;
       for (CountT j = 0; j < consts::numAgents; j++) {
           if (i == j) continue;
           Entity other_agent = ctx.data().agents[j];
           other_agents.e[out_idx++] = other_agent;
       }
   }
   #endif
   ```

3. **Update resetPersistentEntities** (line 201):
   - Loop will reset only 1 agent
   - Remove spawn position spreading logic (lines 215-219) since only 1 agent

### Step 6: Update Entity Count
**File**: `src/sim.cpp`

1. **Update max_total_entities** calculation (line 477):
   - Already uses `consts::numAgents`, will automatically adjust

### Step 7: Update Manager Tensor Shapes
**File**: `src/mgr.cpp`

- All tensor shape calculations already use `consts::numAgents`
- Partner observations tensor will have shape [N, 1, 0, 2]
- No code changes needed, shapes adjust automatically

### Step 8: Update Python Scripts
**File**: `scripts/policy.py`

1. **Update agent ID tensor logic** (lines 28-33):
   ```python
   # Change:
   id_tensor = torch.arange(A).float()
   if A > 1:
       id_tensor = id_tensor / (A - 1)
   
   # To:
   id_tensor = torch.arange(A).float()
   if A > 1:
       id_tensor = id_tensor / (A - 1)
   else:
       id_tensor = torch.zeros_like(id_tensor)  # Single agent gets ID 0
   ```

2. **Update id_tensor view** (line 33):
   ```python
   # Change:
   id_tensor = id_tensor.view(1, 2).expand(N, 2).reshape(batch_size, 1)
   # To:
   id_tensor = id_tensor.view(1, A).expand(N, A).reshape(batch_size, 1)
   ```

### Step 9: Room Generation Considerations
**File**: `src/level_gen.cpp`

- Current room types should still work with 1 agent:
  - SingleButton: Designed for 1 button
  - DoubleButton: May be unsolvable with 1 agent
  - CubeBlocking: Should work with 1 agent
  - CubeButtons: Should work with 1 agent

- Consider modifying room generation or disabling DoubleButton rooms

## Verification Steps

1. **Build the project**:
   ```bash
   cd build && make -j$(nproc)
   ```

2. **Run the simulation**:
   ```bash
   ./build/headless CPU 4 100
   ```

3. **Run unit tests**:
   ```bash
   uv run --extra test pytest test_bindings.py -v --tb=short
   ```

## Impact Summary

- **Minimal code changes**: Most logic already handles variable agent counts
- **Observation changes**: Partner observations become empty (size 0)
- **Training compatibility**: Existing models will be incompatible due to observation shape changes
- **Gameplay**: Some cooperative puzzles may become unsolvable

The implementation leverages existing parameterization, requiring mostly constant changes with minimal logic updates.