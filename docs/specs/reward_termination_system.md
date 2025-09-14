# Reward and Termination System Specification

## Overview

The Madrona Escape Room reward and termination system provides incremental feedback to agents based on forward progress through the environment, with automatic episode management. This specification defines the exact behavior requirements for reward calculation, episode termination, and auto-reset functionality.

## Core Requirements

### 1. Initial State Reward
- **SPEC 1**: Step 0 reward is always 0
- **Rationale**: After reset or episode start, agents should receive no reward before taking any action
- **Implementation**: `reward = 0.0` immediately after `reset_world()` or episode initialization

### 2. Forward Movement Rewards
- **SPEC 2**: Forward movement gives small incremental reward based on forward progress
- **Behavior**: When `agent.position.y > agent.progress.maxY`, reward is calculated as incremental progress
- **Formula**: `reward = (new_y - previous_max_y) / (level.world_max_y - initial_y)`
- **Constraint**: Rewards are only given for **new maximum progress**, not repeated positions

### 3. Backward Movement Behavior
- **SPEC 3**: Moving backward after moving forward does not result in reward
- **Behavior**: When `agent.position.y <= agent.progress.maxY`, no reward is given
- **Rationale**: Only forward progress contributes to task completion

### 4. Stationary Agent Behavior
- **SPEC 4**: If agent does not move, no reward is given
- **Behavior**: When agent position remains unchanged, `reward = 0.0`
- **Implementation**: Reward system only triggers on position changes that exceed `progress.maxY`

### 5. Proportional Reward Calculation
- **SPEC 5**: Reward amount is proportional to forward progress divided by max_y of level
- **Formula**:
  ```
  incremental_reward = progress_this_step / total_possible_progress
  where:
    progress_this_step = new_y - previous_max_y
    total_possible_progress = level.world_max_y - initial_y
  ```
- **Property**: Sum of all incremental rewards equals normalized progress (0.0 to 1.0)

### 6. Episode Termination
- **SPEC 6**: Episode terminates after exactly 200 steps when auto_reset is enabled
- **Behavior**: Step counter increments each simulation step, episode ends at step 200
- **Constant**: `episodeLen = 200` (defined in `consts.hpp`)
- **Implementation**: Episode termination is independent of agent progress or collision

### 7. Auto-Reset Behavior
- **SPEC 7**: When auto_reset is enabled, episodes automatically reset after termination
- **Reset Trigger**: Occurs immediately when `steps_remaining = 0`
- **State Reset**: Agent position, progress tracking, and step counter reset to initial values
- **Reward Consistency**: All reward specifications (SPEC 1-5) apply identically after auto-reset

### 8. Post-Reset Reward Behavior
- **SPEC 8**: After auto-reset, reward system behavior is identical to initial episode
- **Step 0 Rule**: First step after reset must have reward = 0.0 (same as SPEC 1)
- **Progress Reset**: `progress.maxY` and `progress.initialY` reset to new spawn position
- **Continuity**: No reward artifacts or carryover from previous episode

## Special Cases

### Collision Death Penalty
- When agent dies from collision: `reward = -1.0` (overrides any progress reward)
- Death penalty is applied only on the step when `done=1` and `collision_death.died=1`

### Episode Completion Conditions
- **Progress Completion**: Episode completes when `normalized_progress >= 1.0`
- **Time Completion**: Episode completes after exactly 200 steps (when auto_reset enabled)
- **Collision Death**: Episode completes when agent dies from collision
- **Final Reward**: Follows same incremental calculation rules regardless of completion reason
- **Auto-reset Behavior**: Preserves reward calculation consistency across episode boundaries

### 9. Complete Traversal Reward Guarantee
- **SPEC 9**: Agent traversing from spawn to world_max_y without obstacles receives total reward = 1.0 ± ε
- **Behavior**: Complete level traversal (spawn → world_max_y) yields exactly 1.0 total reward
- **Rationale**: Validates that incremental reward system correctly sums to normalized completion
- **Termination**: Episode terminates with `done=1` when agent reaches world boundary

### Progress Tracking
- `progress.maxY`: Tracks highest Y position reached by agent
- `progress.initialY`: Starting Y position, set after physics settling
- Progress values are persistent across steps until episode reset

## Implementation Notes

### Reward System Function
```cpp
// Core reward calculation logic
if (pos.y > progress.maxY) {
    float prev_maxY = progress.maxY;
    progress.maxY = pos.y;
    float total_possible_progress = level.world_max_y - progress.initialY;
    float progress_this_step = pos.y - prev_maxY;
    out_reward.v = progress_this_step / total_possible_progress;
} else {
    out_reward.v = 0.0f;
}
```

### Progress Initialization
- Progress values initialized with sentinel values (-999999.0f)
- After physics settling: `progress.maxY = progress.initialY = current_position.y`
- Initialization ensures step 0 reward is always 0

## Test Requirements

All implementations must pass the following validation tests:

### Core Reward Tests
1. `test_step_zero_reward_is_zero()` - Validates SPEC 1
2. `test_forward_movement_gives_incremental_reward()` - Validates SPEC 2
3. `test_backward_movement_gives_no_reward()` - Validates SPEC 3
4. `test_no_movement_gives_no_reward()` - Validates SPEC 4
5. `test_reward_proportional_to_progress_over_max_y()` - Validates SPEC 5

### Termination and Auto-Reset Tests
6. `test_episode_terminates_after_200_steps()` - Validates SPEC 6
7. `test_auto_reset_after_episode_termination()` - Validates SPEC 7
8. `test_post_reset_reward_consistency()` - Validates SPEC 8

### Integration Tests
9. `test_multiple_episode_cycles_with_auto_reset()` - Validates cross-episode consistency
10. `test_reward_behavior_across_episode_boundaries()` - Validates no reward artifacts

### Complete Traversal Tests
11. `test_complete_traversal_yields_unit_reward()` - Validates SPEC 9

## Known Issues

### Current Bugs
- **Step 0 Reward Bug**: Currently returning non-zero reward (~0.0139) at step 0
- **Root Cause**: Progress initialization occurring before reward calculation
- **Fix Required**: Ensure progress initialization completes before first reward calculation

## Version History
- v1.0: Initial specification based on incremental reward system requirements