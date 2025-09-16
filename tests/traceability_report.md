# Requirement Traceability Report

Total tests with spec markers: 16

## docs/specs/sim.md - agentCollisionSystem

- **TestCollisionTermination.test_collision_death_termination_code** (test_reward_termination_system.py:1700)
  - Test termination reason code 2 when agent collides with terminating object.

- **TestCollisionTermination.test_collision_reward_penalty** (test_reward_termination_system.py:1366)
  - Test that collision with DoneOnCollide objects gives -0.1 reward.

- **TestCollisionTermination.test_north_collision_terminates** (test_reward_termination_system.py:1190)
  - Test collision with terminating cube (north) ends episode.

## docs/specs/sim.md - collectObservationsSystem

- **test_depth_and_rgb_together** (test_depth_sensor.py:67)
  - Test that both depth and RGB work together in RGBD mode

- **test_depth_tensor_always_enabled** (test_depth_sensor.py:42)
  - Test dedicated depth sensor fixture that always enables batch renderer

- **test_depth_tensor_basic** (test_depth_sensor.py:11)
  - Test that depth tensor is accessible with @depth_default marker

## docs/specs/sim.md - compassSystem

- **test_compass_tensor_basic** (test_compass_tensor.py:8)
  - Test that the compass tensor is working correctly

- **test_compass_tensor_updates_after_step** (test_compass_tensor.py:48)
  - Test that compass tensor updates after simulation steps

## docs/specs/sim.md - lidarSystem

- **TestHorizontalLidar.test_128_beam_horizontal_lidar_with_fixture** (test_horizontal_lidar.py:171)
  - Test 128 horizontal lidar beams with 120° FOV using wall-in-front scenario.

## docs/specs/sim.md - movementSystem

- **test_forward_movement** (test_movement_system.py:37)
  - Test agent moving forward for entire episode

## docs/specs/sim.md - resetSystem

- **test_auto_reset_after_episode_termination** (test_reward_termination_system.py:608)
  - SPEC 7: When auto_reset is enabled, episodes automatically reset after termination

## docs/specs/sim.md - rewardSystem

- **test_backward_movement_gives_no_reward** (test_reward_termination_system.py:402)
  - SPEC 3: Moving backward after moving forward does not result in reward

- **test_forward_movement_gives_incremental_reward** (test_reward_termination_system.py:365)
  - SPEC 2: Forward movement gives small incremental reward based on forward progress

- **test_no_movement_gives_no_reward** (test_reward_termination_system.py:443)
  - SPEC 4: If agent does not move, no reward is given

- **test_step_zero_reward_is_zero** (test_reward_termination_system.py:348)
  - SPEC 1: Step 0 reward is always 0

## docs/specs/sim.md - stepTrackerSystem

- **test_episode_terminates_after_200_steps** (test_reward_termination_system.py:529)
  - SPEC 6: Episode terminates after exactly 200 steps when auto_reset is enabled

## Specification Coverage Summary

### docs/specs/sim.md Coverage:

- ✅ **movementSystem**: 1 test(s)
- ✅ **agentCollisionSystem**: 3 test(s)
- ❌ **agentZeroVelSystem**: 0 test(s)
- ✅ **stepTrackerSystem**: 1 test(s)
- ✅ **rewardSystem**: 4 test(s)
- ✅ **resetSystem**: 1 test(s)
- ❌ **initProgressAfterReset**: 0 test(s)
- ✅ **collectObservationsSystem**: 3 test(s)
- ✅ **compassSystem**: 2 test(s)
- ✅ **lidarSystem**: 1 test(s)
