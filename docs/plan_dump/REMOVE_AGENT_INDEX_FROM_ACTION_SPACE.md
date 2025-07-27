# Plan to Remove Agent Index from Action Space

## Overview

Currently, actions are structured as `[num_worlds, num_agents, 3]` where the middle dimension indexes agents. Since there's only 1 agent per world (`numAgents = 1`), this dimension is redundant. This plan outlines the changes needed to simplify the action space to `[num_worlds, 3]`.

## Current State

- Action tensor shape: `[num_worlds, num_agents, 3]` where `num_agents = 1`
- Actions are exported per-agent via `registry.exportColumn<Agent, Action>`
- `setAction()` method takes `world_idx` and `agent_idx` parameters
- Action buffer offset calculated as: `world_idx * num_agents + agent_idx`

## Target State

- Action tensor shape: `[num_worlds, 3]`
- Actions exported per-world
- `setAction()` method takes only `world_idx` parameter
- Action buffer offset: just `world_idx`

## Files to Modify

### 1. **src/mgr.cpp**

#### Changes needed:
- Update `actionTensor()` method to return shape `[num_worlds, 3]`:
  ```cpp
  // Current
  return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
      {
          impl_->cfg.numWorlds,
          consts::numAgents,  // Remove this dimension
          consts::numActionComponents,
      });
  
  // New
  return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
      {
          impl_->cfg.numWorlds,
          consts::numActionComponents,
      });
  ```

- Update `setAction()` method:
  ```cpp
  // Current
  void Manager::setAction(int32_t world_idx,
                          int32_t agent_idx,
                          int32_t move_amount,
                          int32_t move_angle,
                          int32_t rotate)
  
  // New
  void Manager::setAction(int32_t world_idx,
                          int32_t move_amount,
                          int32_t move_angle,
                          int32_t rotate)
  ```

- Update action buffer offset calculation:
  ```cpp
  // Current
  auto *action_ptr = impl_->agentActionsBuffer +
      world_idx * consts::numAgents + agent_idx;
  
  // New
  auto *action_ptr = impl_->agentActionsBuffer + world_idx;
  ```

### 2. **src/mgr.hpp**

- Update `setAction()` method signature:
  ```cpp
  void setAction(int32_t world_idx,
                 int32_t move_amount,
                 int32_t move_angle,
                 int32_t rotate);
  ```

### 3. **src/viewer.cpp**

- Update viewer callback to not pass `agent_idx`:
  ```cpp
  // Current (around line 228)
  mgr.setAction(world_idx, agent_idx, move_amount, move_angle, r);
  
  // New
  mgr.setAction(world_idx, move_amount, move_angle, r);
  ```

- Update replay functionality to remove agent iteration:
  ```cpp
  // Current
  for (uint32_t j = 0; j < num_views; j++) {
      uint32_t base_idx = 3 * (cur_replay_step * num_views * num_worlds +
          i * num_views + j);
      // ...
      mgr.setAction(i, j, move_amount, move_angle, turn);
  }
  
  // New
  uint32_t base_idx = 3 * (cur_replay_step * num_worlds + i);
  // ...
  mgr.setAction(i, move_amount, move_angle, turn);
  ```

### 4. **scripts/train.py**

- Remove action tensor flattening:
  ```python
  # Current (line 124)
  actions = actions.view(-1, *actions.shape[2:])
  
  # New - actions already in correct shape
  # No flattening needed
  ```

### 5. **scripts/sim_bench.py**

- Remove incorrect 4th action component (line 30)
- Update random action generation:
  ```python
  # Current
  actions[..., 0] = torch.randint_like(actions[..., 0], 0, 4)
  actions[..., 1] = torch.randint_like(actions[..., 1], 0, 8)
  actions[..., 2] = torch.randint_like(actions[..., 2], 0, 5)
  actions[..., 3] = torch.randint_like(actions[..., 3], 0, 2)  # Remove this line
  ```

### 6. **scripts/infer.py**

- Remove action tensor flattening (same as train.py)
- Remove obsolete "Grab Probs" output:
  ```python
  # Remove these lines
  print("Grab Probs")
  print(" ", np.array_str(probs[3][0].cpu().numpy(), precision=2, suppress_small=True))
  print(" ", np.array_str(probs[3][1].cpu().numpy(), precision=2, suppress_small=True))
  ```

### 7. **tests/python/test_helpers.py**

- Remove `agent_idx` parameter from all methods:
  ```python
  # Example change
  def move_forward(self, world_idx: Optional[int] = None, speed: float = 1.0):
      if world_idx is None:
          self.actions[:, 0] = speed
          self.actions[:, 1] = 0
      else:
          self.actions[world_idx, 0] = speed
          self.actions[world_idx, 1] = 0
  ```

- Update all action array indexing from `[:, agent_idx, :]` to `[:, :]`

### 8. **tests/python/test_bindings.py**

- Update shape assertions:
  ```python
  # Current
  assert actions.shape == (4, 1, 3), f"Expected shape (4, 1, 3), got {actions.shape}"
  
  # New
  assert actions.shape == (4, 3), f"Expected shape (4, 3), got {actions.shape}"
  ```

- Update action tensor indexing throughout

### 9. **tests/python/conftest.py**

- Update action recording shape:
  ```python
  # Current (line 68)
  # Stack all actions: shape is [steps, worlds, agents, 3]
  
  # New
  # Stack all actions: shape is [steps, worlds, 3]
  ```

- Update how number of agents is determined:
  ```python
  # Current
  self.num_agents = manager.action_tensor().to_torch().shape[1]
  
  # New
  self.num_agents = 1  # Single agent per world
  ```

### 10. **src/headless.cpp**

- Remove agent loop in action generation:
  ```cpp
  // Current
  for (CountT k = 0; k < 2; k++) {
      // ...
      mgr.setAction(j, k, x, y, r);
      int64_t base_idx = j * num_steps * 2 * 3 + i * 2 * 3 + k * 3;
      // ...
  }
  
  // New
  mgr.setAction(j, x, y, r);
  int64_t base_idx = j * num_steps * 3 + i * 3;
  ```

### 11. **CLAUDE.md**

- Fix documentation error:
  ```markdown
  # Current
  # Run tests with action recording (saves to test_outputs/)
  
  # New
  # Run tests with action recording (saves to test_recordings/)
  ```

## Implementation Notes

1. **ECS Export Decision**: We kept the per-agent export mechanism (`exportColumn<Agent, Action>`) unchanged. This is ideal because:
   - Actions are still stored per-entity in the ECS
   - Future AI creatures can have their own Action components
   - The same movement system can process both player and AI actions
   - Since there's 1 agent per world, the export naturally becomes `[num_worlds, 3]`

2. The training code flattening was removed for actions but kept for rewards/dones (which are still per-agent).

3. All test files were updated to work with the new action shape.

4. The viewer code was updated in both the interactive callback and replay functionality.

5. Additional files not in the original plan (headless.cpp) were discovered and updated during testing.

## Testing Plan

1. Update all unit tests to use new action shape
2. Verify training still works correctly
3. Test viewer functionality with new action interface
4. Run benchmark to ensure no performance regression
5. Verify action recording/replay functionality

## Additional Enhancements

- **test_deterministic_actions**: Enhanced to repeat each movement 10 times with 5-step pauses between different movements, making the replay visualization clearer and more comprehensive.

## Benefits

1. Simpler API - no need to specify agent index when there's only one agent
2. Reduced memory usage - one less dimension in action tensors
3. Cleaner code - removes unnecessary indexing complexity
4. Better aligned with single-agent nature of current implementation