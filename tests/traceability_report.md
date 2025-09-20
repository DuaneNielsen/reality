# Requirement Traceability Report

Total tests with spec markers: 88

## docs/specs/level_compiler.md - BoundaryWallGeneration

- **TestEndToEndIntegration.test_boundary_wall_offset_default_zero** (test_ascii_level_compiler.py:527)
  - Test that boundary_wall_offset defaults to 0.0

- **TestEndToEndIntegration.test_boundary_wall_offset_specified** (test_ascii_level_compiler.py:547)
  - Test that boundary_wall_offset can be specified

- **TestEndToEndIntegration.test_boundary_wall_offset_validation** (test_ascii_level_compiler.py:568)
  - Test validation of boundary_wall_offset parameter

- **TestEndToEndIntegration.test_boundary_wall_offset_without_auto_walls** (test_ascii_level_compiler.py:617)
  - Test that boundary_wall_offset is ignored when auto_boundary_walls is False

## docs/specs/level_compiler.md - CompiledLevel

- **TestEdgeCases.test_binary_roundtrip_c_api_arrays** (test_level_compiler_c_api.py:173)
  - Test that binary save/load works with C API-sized arrays.

- **TestEdgeCases.test_compiled_arrays_correct_size** (test_level_compiler_c_api.py:132)
  - Test that compiled level arrays are sized to MAX_TILES_C_API.

- **TestEndToEndIntegration.test_dataclass_to_ctypes_conversion** (test_ascii_level_compiler.py:175)
  - Test that CompiledLevel dataclass works with ctypes

- **TestEdgeCases.test_unused_array_slots_zero** (test_level_compiler_c_api.py:151)
  - Test that unused array slots are properly zeroed.

## docs/specs/level_compiler.md - Configuration

- **TestEdgeCases.test_fallback_when_c_api_unavailable** (test_level_compiler_c_api.py:317)
  - Test fallback behavior when C API is not available.

- **TestEdgeCases.test_max_tiles_c_api_value** (test_level_compiler_c_api.py:30)
  - Test that MAX_TILES_C_API returns expected value.

- **TestEdgeCases.test_max_tiles_consistency** (test_level_compiler_c_api.py:37)
  - Test that MAX_TILES_C_API matches direct C API call.

## docs/specs/level_compiler.md - Invariants

- **TestEdgeCases.test_exact_square_root_size** (test_level_compiler_c_api.py:287)
  - Test levels at exact square root of MAX_TILES_C_API.

- **TestEdgeCases.test_level_within_limits** (test_level_compiler_c_api.py:50)
  - Test that levels within limits compile successfully.

- **TestEdgeCases.test_maximum_size_level** (test_level_compiler_c_api.py:67)
  - Test level at exactly the maximum size.

- **TestEdgeCases.test_minimum_size_level** (test_level_compiler_c_api.py:270)
  - Test minimum possible level size (3x3).

- **TestEdgeCases.test_rectangular_at_limit** (test_level_compiler_c_api.py:300)
  - Test rectangular level at the size limit.

## docs/specs/level_compiler.md - JSON Level Format (Single Level)

- **TestEndToEndIntegration.test_explicit_done_on_collision_flags** (test_ascii_level_compiler.py:379)
  - Test that done_on_collision flags are set correctly via JSON

- **TestEndToEndIntegration.test_json_level_with_agent_facing** (test_ascii_level_compiler.py:291)
  - Test JSON level with agent facing angles

- **TestEndToEndIntegration.test_json_level_without_facing** (test_ascii_level_compiler.py:324)
  - Test JSON level without agent_facing defaults to 0

- **TestEndToEndIntegration.test_mixed_collision_flags** (test_ascii_level_compiler.py:472)
  - Test level with mixed collision behavior

- **TestEdgeCases.test_spawn_random_flag** (test_level_compiler_c_api.py:250)
  - Test that spawn_random flag works correctly.

- **TestEndToEndIntegration.test_validation_of_done_on_collision_type** (test_ascii_level_compiler.py:507)
  - Test that done_on_collision must be a boolean

## docs/specs/level_compiler.md - JSON Multi-Level Format

- **TestEndToEndIntegration.test_boundary_wall_offset_multi_level** (test_ascii_level_compiler.py:591)
  - Test boundary_wall_offset in multi-level format

## docs/specs/level_compiler.md - System Integration

- **TestEndToEndIntegration.test_backward_compatibility** (test_ascii_level_compiler.py:278)
  - Test that SimManager still works without level_ascii

- **TestEndToEndIntegration.test_complete_pipeline** (test_ascii_level_compiler.py:645)
  - Test complete pipeline from ASCII to simulation

- **TestEndToEndIntegration.test_json_level_in_sim_manager** (test_ascii_level_compiler.py:345)
  - Test using JSON level with SimManager

- **TestEndToEndIntegration.test_manager_creation_with_ascii_level** (test_ascii_level_compiler.py:244)
  - Test creating manager with ASCII level

- **TestEndToEndIntegration.test_multiple_worlds_same_level** (test_ascii_level_compiler.py:262)
  - Test multiple worlds with same ASCII level

## docs/specs/level_compiler.md - compile_ascii_level

- **TestEndToEndIntegration.test_default_tileset_done_on_collision** (test_ascii_level_compiler.py:439)
  - Test that default tileset sets done_on_collision correctly

- **TestEndToEndIntegration.test_error_cases** (test_ascii_level_compiler.py:74)
  - Test error handling for invalid levels

- **TestEndToEndIntegration.test_level_with_obstacles** (test_ascii_level_compiler.py:47)
  - Test compiling a level with cube obstacles

- **TestEndToEndIntegration.test_multiple_level_compilation** (test_ascii_level_compiler.py:216)
  - Test compiling multiple levels

- **TestEndToEndIntegration.test_simple_room_compilation** (test_ascii_level_compiler.py:22)
  - Test compiling a simple room

## docs/specs/level_compiler.md - compile_level

- **TestEdgeCases.test_64x64_level_rejected** (test_level_compiler_c_api.py:108)
  - Test that maximum dimension levels are correctly rejected when too large.

- **TestEdgeCases.test_oversized_level_rejected** (test_level_compiler_c_api.py:88)
  - Test that levels exceeding MAX_TILES_C_API are rejected.

## docs/specs/level_compiler.md - compile_multi_level

- **TestEndToEndIntegration.test_multi_level_compilation** (test_ascii_level_compiler.py:94)
  - Test compiling multi-level JSON format

- **TestEndToEndIntegration.test_multi_level_validation_errors** (test_ascii_level_compiler.py:149)
  - Test multi-level format validation errors

## docs/specs/level_compiler.md - validate_compiled_level

- **TestEndToEndIntegration.test_compiled_level_validation** (test_ascii_level_compiler.py:203)
  - Test compiled level validation

- **TestEdgeCases.test_validation_accepts_c_api_arrays** (test_level_compiler_c_api.py:217)
  - Test that validation accepts arrays sized to MAX_TILES_C_API.

- **TestEdgeCases.test_validation_rejects_wrong_sized_arrays** (test_level_compiler_c_api.py:230)
  - Test that validation rejects arrays not sized to MAX_TILES_C_API.

## docs/specs/mgr.md - isRecording

- **test_recording_state_persistence** (test_native_recording.py:380)
  - Test that recording state persists across operations

## docs/specs/mgr.md - loadReplay

- **test_action_data_corruption_detection** (test_native_recording_replay_roundtrip.py:500)
  - Test that action data corruption is detected during replay

- **test_format_error_conditions** (test_native_recording.py:494)
  - Test error condition handling for current format

- **test_partial_file_replay_handling** (test_native_recording_replay_roundtrip.py:709)
  - Test handling of partial/incomplete recording files

- **test_v3_format_validation** (test_multi_level_recording_replay.py:268)
  - Test that only v3 format files are accepted.

## docs/specs/mgr.md - readReplayMetadata

- **test_multi_level_metadata_storage** (test_multi_level_recording_replay.py:219)
  - Test that recording properly stores metadata for multiple levels.

## docs/specs/mgr.md - replayStep

- **test_gpu_recording_and_replay_complete** (test_native_recording_gpu.py:15)
  - Test complete GPU recording and replay functionality in one test

- **test_roundtrip_basic_consistency** (test_native_recording_replay_roundtrip.py:16)
  - Test basic record → replay → verify cycle.

- **test_roundtrip_edge_cases** (test_native_recording_replay_roundtrip.py:308)
  - Test edge cases in record/replay

- **test_roundtrip_observation_consistency** (test_native_recording_replay_roundtrip.py:42)
  - Test observation recording with automatic replay verification

- **test_roundtrip_session_replay** (test_native_recording_replay_roundtrip.py:288)
  - Test a single record/replay session using automatic verification fixture

- **test_roundtrip_trajectory_file_verification** (test_native_recording_replay_roundtrip.py:75)
  - Test that trajectory traces match exactly between record and replay by comparing trace files.

- **test_roundtrip_trajectory_file_verification_with_debug** (test_native_recording_replay_roundtrip.py:170)
  - Test that debug session trajectory traces match replay traces.

- **test_roundtrip_with_reset** (test_native_recording_replay_roundtrip.py:379)
  - Test recording/replay across episode resets with automatic verification

- **test_trajectory_file_verification_detects_differences** (test_native_recording_replay_roundtrip.py:203)
  - Test that our trajectory verification detects when trajectories differ

## docs/specs/mgr.md - startRecording

- **test_action_sequence_detailed_validation** (test_native_recording_replay_roundtrip.py:406)
  - Test detailed action sequence validation with step-by-step verification

- **test_current_format_specification_compliance** (test_native_recording.py:412)
  - Test current format (version 3) specification compliance and struct layout

- **test_field_alignment_and_padding** (test_native_recording.py:576)
  - Test field alignment and padding in current format

- **test_file_structure_integrity_validation** (test_native_recording_replay_roundtrip.py:607)
  - Test file structure integrity and boundary verification

- **test_gpu_recording_lifecycle** (test_native_recording.py:45)
  - Test basic recording start/stop cycle on GPU

- **test_multi_level_recording_roundtrip** (test_multi_level_recording_replay.py:68)
  - Test recording and replaying across multiple different levels.

- **test_recording_error_handling** (test_native_recording.py:143)
  - Test error conditions in recording

- **test_recording_file_format** (test_native_recording.py:180)
  - Test comprehensive recording file format validation - Version 3 complete validation

- **test_recording_lifecycle** (test_native_recording.py:15)
  - Test basic recording start/stop cycle

- **test_recording_with_steps** (test_native_recording.py:97)
  - Test recording with actual simulation steps

## docs/specs/mgr.md - stopRecording

- **test_recording_empty_session** (test_native_recording.py:354)
  - Test recording session with no steps

## docs/specs/sim.md - Step 2: Reset Agent Physics and Spawning

- **test_replay_spawn_position_bug_simple** (test_native_replay.py:381)
  - Simplified test that demonstrates the spawn position bug clearly

- **test_replay_spawn_position_determinism** (test_native_replay.py:301)
  - Test that random spawn positions are deterministic during replay

- **test_spawn_positions_are_random_between_episodes** (test_native_replay.py:430)
  - Test that spawn_random=True produces different positions across episodes

## docs/specs/sim.md - agentCollisionSystem

- **TestCollisionTermination.test_collision_death_termination_code** (test_reward_termination_system.py:1707)
  - Test termination reason code 2 when agent collides with terminating object.

- **TestCollisionTermination.test_collision_reward_penalty** (test_reward_termination_system.py:1372)
  - Test that collision with DoneOnCollide objects gives -0.1 reward.

- **TestCollisionTermination.test_north_collision_terminates** (test_reward_termination_system.py:1195)
  - Test collision with terminating cube (north) ends episode.

## docs/specs/sim.md - collectObservationsSystem

- **test_depth_and_rgb_together** (test_depth_sensor.py:67)
  - Test that both depth and RGB work together in RGBD mode

- **test_depth_tensor_always_enabled** (test_depth_sensor.py:42)
  - Test dedicated depth sensor fixture that always enables batch renderer

- **test_depth_tensor_basic** (test_depth_sensor.py:11)
  - Test that depth tensor is accessible with @depth_default marker

## docs/specs/sim.md - compassSystem

- **test_compass_points_to_target** (test_target_compass.py:11)
  - Test that compass points toward target entity, not agent rotation

- **test_compass_tensor_basic** (test_compass_tensor.py:8)
  - Test that the compass tensor is working correctly

- **test_compass_tensor_updates_after_step** (test_compass_tensor.py:48)
  - Test that compass tensor updates after simulation steps

- **test_compass_updates_with_agent_movement** (test_target_compass.py:71)
  - Test that compass direction updates as agent moves relative to target

## docs/specs/sim.md - customMotionSystem

- **test_target_entity_exists** (test_target_compass.py:111)
  - Test that target entity is created and positioned correctly

## docs/specs/sim.md - lidarSystem

- **TestHorizontalLidar.test_128_beam_horizontal_lidar_with_fixture** (test_horizontal_lidar.py:171)
  - Test 128 horizontal lidar beams with 120° FOV using wall-in-front scenario.

## docs/specs/sim.md - movementSystem

- **test_forward_movement** (test_movement_system.py:37)
  - Test agent moving forward for entire episode

## docs/specs/sim.md - resetSystem

- **test_auto_reset_after_episode_termination** (test_reward_termination_system.py:612)
  - SPEC 7: When auto_reset is enabled, episodes automatically reset after termination

## docs/specs/sim.md - rewardSystem

- **test_backward_movement_gives_no_reward** (test_reward_termination_system.py:405)
  - SPEC 3: Moving backward after moving forward does not result in reward

- **test_forward_movement_gives_incremental_reward** (test_reward_termination_system.py:368)
  - SPEC 2: Forward movement gives small incremental reward based on forward progress

- **test_no_movement_gives_no_reward** (test_reward_termination_system.py:446)
  - SPEC 4: If agent does not move, no reward is given

- **test_step_zero_reward_is_zero** (test_reward_termination_system.py:350)
  - SPEC 1: Step 0 reward is always 0

## docs/specs/sim.md - stepTrackerSystem

- **test_episode_terminates_after_200_steps** (test_reward_termination_system.py:533)
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
- ✅ **compassSystem**: 4 test(s)
- ✅ **lidarSystem**: 1 test(s)
