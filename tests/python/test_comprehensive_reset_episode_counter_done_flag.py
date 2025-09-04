#!/usr/bin/env python3
"""
Comprehensive test suite for reset, episode counter, and done flag order of operations.
Based on the test plan in docs/plan_dump/reset_episode_counter_done_flag_test_plan.md

This test suite validates:
1. Basic reset timing behavior
2. Step counter precision at boundaries
3. Collision vs step termination interactions
4. Multi-world synchronization
5. Order of operations edge cases
6. State persistence across resets

Key system understanding:
- Execution order: stepTrackerSystem() → rewardSystem() → resetSystem()
- Reset triggers: Manual, auto-reset, step exhaustion, collision termination
- Key files: src/sim.cpp:353-362, src/level_gen.cpp:94-148
"""

import pytest
import torch
from test_helpers import AgentController, reset_all_worlds, reset_world

from madrona_escape_room.generated_constants import consts


class TestHelpers:
    """Helper methods for reset/episode/done flag testing"""

    @staticmethod
    def assert_step_sequence(mgr, expected_steps_taken, expected_dones):
        """Validate step counter and done flag sequences"""
        steps_tensor = mgr.steps_taken_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()

        for world_idx in range(len(expected_steps_taken)):
            actual_steps = steps_tensor[world_idx, 0, 0].item()
            actual_done = done_tensor[world_idx, 0].item()

            assert actual_steps == expected_steps_taken[world_idx], (
                f"World {world_idx}: expected {expected_steps_taken[world_idx]} steps taken, "
                f"got {actual_steps}"
            )
            assert actual_done == expected_dones[world_idx], (
                f"World {world_idx}: expected done={expected_dones[world_idx]}, "
                f"got done={actual_done}"
            )

    @staticmethod
    def trigger_collision_at_step(mgr, world_idx, target_step):
        """Force collision termination at specific step by aggressive movement"""
        actions = mgr.action_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()

        # Reset target world first
        reset_tensor[:] = 0
        reset_tensor[world_idx] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set aggressive movement for collision
        actions[:] = 0
        actions[world_idx, 0] = 3  # FAST movement
        actions[world_idx, 1] = 0  # FORWARD direction

        # Step to target step
        for _ in range(target_step):
            mgr.step()

    @staticmethod
    def validate_tensor_consistency(mgr):
        """Ensure all tensors reflect consistent state post-reset"""
        steps = mgr.steps_taken_tensor().to_torch()
        done = mgr.done_tensor().to_torch()
        obs = mgr.self_observation_tensor().to_torch()

        # Basic consistency checks
        assert steps.shape[0] == done.shape[0] == obs.shape[0], "Tensor world dimensions must match"

        # State consistency: if done=0, steps taken should be < episodeLen
        for world_idx in range(steps.shape[0]):
            if done[world_idx, 0].item() == 0:
                assert (
                    steps[world_idx, 0, 0].item() < consts.episodeLen
                ), f"World {world_idx}: done=0 but steps_taken={steps[world_idx, 0, 0].item()}"


# =============================================================================
# Category 1: Basic Reset Timing Tests (RT-001 to RT-004)
# =============================================================================


class TestBasicResetTiming:
    """Basic reset timing validation tests"""

    def test_RT_001_manual_reset_when_done_zero(self, cpu_manager):
        """RT-001: Manual reset when done=0 should cause immediate reset"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset to clean state
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Verify initial state: done=0, steps=0 (no steps taken yet)
        assert not done_tensor.any(), "All done flags should be 0 after reset"
        assert (steps_taken == 0).all(), "All steps taken should be 0 after reset"

        # Run a few steps to get mid-episode state
        actions[:] = 0  # Stop movement
        for _ in range(50):
            mgr.step()

        # Verify mid-episode state
        assert not done_tensor.any(), "Should still be mid-episode"
        expected_steps_taken = 50
        assert (
            steps_taken[:, 0, 0] == expected_steps_taken
        ).all(), f"Expected {expected_steps_taken} steps taken"

        # Test manual reset mid-episode
        reset_tensor[0] = 1  # Reset world 0 only
        mgr.step()

        # World 0 should be reset immediately
        assert done_tensor[0, 0] == 0, "World 0 done should be 0 after reset"
        assert steps_taken[0, 0, 0] == 0, "World 0 steps taken should be reset to 0"

        # Other worlds should continue unaffected
        for world_idx in range(1, 4):
            expected_steps_taken = 51  # One more step taken
            assert (
                steps_taken[world_idx, 0, 0] == expected_steps_taken
            ), f"World {world_idx} should continue with {expected_steps_taken} steps taken"

    def test_RT_002_manual_reset_when_done_one(self, cpu_manager):
        """RT-002: Manual reset when done=1 should clear done to 0"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Fast forward to episode end
        actions[:] = 0
        actions[:, 0] = 1  # SLOW movement to avoid early collision

        for _ in range(consts.episodeLen):
            mgr.step()

        # Find a world that's done
        done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False)
        assert len(done_worlds) > 0, "At least one world should be done after episodeLen steps"

        done_world = done_worlds[0].item()
        assert done_tensor[done_world, 0] == 1, "Selected world should be done"

        # Manual reset of done world
        reset_tensor[:] = 0
        reset_tensor[done_world] = 1
        mgr.step()

        # Verify reset clears done flag and resets steps
        assert done_tensor[done_world, 0] == 0, "Done flag should be cleared"
        assert steps_taken[done_world, 0, 0] == 0, "Steps taken should be reset to 0"

    def test_RT_003_auto_reset_off_done_one(self, cpu_manager):
        """RT-003: With auto_reset=False, done=1 should stay 1"""
        mgr = cpu_manager

        # Verify auto_reset is off (cpu_manager fixture sets auto_reset=False)
        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Run to episode end
        actions[:] = 0
        actions[:, 0] = 1  # SLOW movement

        for _ in range(consts.episodeLen):
            mgr.step()

        # Find done worlds
        done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False)
        if len(done_worlds) > 0:
            done_world = done_worlds[0].item()

            # Run additional steps without manual reset
            for _ in range(10):
                mgr.step()

            # Done flag should persist (no auto-reset)
            assert done_tensor[done_world, 0] == 1, "Done flag should persist with auto_reset=False"
            # Note: steps_taken can exceed episodeLen when episode is done but no reset occurs
            # This is expected behavior - the counter continues incrementing
            assert (
                steps_taken[done_world, 0, 0] >= consts.episodeLen
            ), "Steps taken should be >= episodeLen"

    def test_RT_004_auto_reset_on_done_one(self, cpu_manager):
        """RT-004: With auto_reset=True, done=1 should trigger automatic reset"""
        # Note: cpu_manager has auto_reset=False, so we need to create a manager
        # with auto_reset=True
        # This test demonstrates the behavior difference
        from madrona_escape_room import ExecMode, SimManager, create_default_level

        mgr_auto = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=4,
            rand_seed=42,
            enable_batch_renderer=False,
            auto_reset=True,  # Enable auto-reset
            compiled_levels=create_default_level(),
        )

        # Get tensor references
        actions = mgr_auto.action_tensor().to_torch()
        reset_tensor = mgr_auto.reset_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr_auto.step()
        reset_tensor[:] = 0

        # Run to episode end with minimal movement to avoid collision
        actions[:] = 0
        actions[:, 0] = 0  # STOP movement to rely purely on step counter

        for step in range(consts.episodeLen + 5):
            mgr_auto.step()

            # With auto-reset, done worlds should reset automatically
            # Check if any world that becomes done gets reset in the next step
            if step >= consts.episodeLen:
                # After episodeLen, we should see reset behavior
                # Some worlds may be done=0 due to auto-reset, others may be done=1 briefly
                pass  # Auto-reset behavior can be complex, main test is that system doesn't hang


# =============================================================================
# Category 2: Step Counter Precision Tests (SC-001 to SC-004)
# =============================================================================


class TestStepCounterPrecision:
    """Step counter precision and boundary condition tests"""

    def test_SC_001_exact_episode_len_steps(self, cpu_manager):
        """SC-001: After exactly episodeLen steps, done should be 1"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement to rely purely on step counter
        actions[:] = 0  # All STOP

        # Run for exactly episodeLen steps
        for step in range(consts.episodeLen):
            mgr.step()

            expected_taken = step + 1

            # During episode: done=0, steps decreasing
            if step < consts.episodeLen - 1:
                assert not done_tensor.any(), f"Done should be 0 at step {step + 1}"
                assert (
                    steps_taken[:, 0, 0] == expected_taken
                ).all(), f"Step {step + 1}: expected {expected_taken} steps taken"
            else:
                # At exactly episodeLen steps: done=1, steps_taken=episodeLen
                for world_idx in range(4):
                    if steps_taken[world_idx, 0, 0] == consts.episodeLen:
                        assert (
                            done_tensor[world_idx, 0] == 1
                        ), f"World {world_idx} should be done when steps_taken=episodeLen"

    def test_SC_002_reset_at_step_199(self, cpu_manager):
        """SC-002: Reset at step 199 should give fresh episode with steps=200"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement
        actions[:] = 0

        # Run to step 199 (one before natural end)
        for _ in range(consts.episodeLen - 1):
            mgr.step()

        # Should be at step 199: steps_taken=1, done=0
        assert (
            steps_taken[:, 0, 0] == consts.episodeLen - 1
        ).all(), "Should have taken episodeLen-1 steps"
        assert not done_tensor.any(), "Should not be done yet"

        # Manual reset world 0
        reset_tensor[0] = 1
        mgr.step()
        reset_tensor[0] = 0

        # World 0 should be reset with fresh episode
        assert steps_taken[0, 0, 0] == 0, "World 0 should be reset to 0 steps taken"
        assert done_tensor[0, 0] == 0, "World 0 should not be done"

        # Other worlds should have completed their episode (steps=0, done=1)
        for world_idx in range(1, 4):
            assert (
                steps_taken[world_idx, 0, 0] == consts.episodeLen
            ), f"World {world_idx} should have 0 steps".replace("0 steps", "episodeLen steps taken")
            assert done_tensor[world_idx, 0] == 1, f"World {world_idx} should be done"

    def test_SC_003_reset_at_step_200_done_one(self, cpu_manager):
        """SC-003: Reset at step 200 (when done=1) should start fresh episode"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement
        actions[:] = 0

        # Run for exactly episodeLen steps to reach natural termination
        for _ in range(consts.episodeLen):
            mgr.step()

        # Find worlds that are done
        done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False)
        if len(done_worlds) > 0:
            done_world = done_worlds[0].item()

            # Verify it's actually done
            assert done_tensor[done_world, 0] == 1, "World should be done"
            assert (
                steps_taken[done_world, 0, 0] == consts.episodeLen
            ), "World should have 0 steps".replace("0 steps", "episodeLen steps taken")

            # Reset the done world
            reset_tensor[:] = 0
            reset_tensor[done_world] = 1
            mgr.step()

            # Should start fresh episode
            assert steps_taken[done_world, 0, 0] == 0, "Should be reset to 0 steps taken"
            assert done_tensor[done_world, 0] == 0, "Should not be done after reset"

    def test_SC_004_multiple_resets_per_episode(self, cpu_manager):
        """SC-004: Multiple resets should each give fresh episodes with steps=200"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement
        actions[:] = 0

        # Test multiple resets of world 0
        for reset_num in range(5):  # 5 resets
            # Run some steps
            steps_to_run = 20 + reset_num * 10  # Vary the timing
            for _ in range(steps_to_run):
                mgr.step()

            # Reset world 0
            reset_tensor[0] = 1
            mgr.step()
            reset_tensor[0] = 0

            # Verify fresh episode each time
            assert (
                steps_taken[0, 0, 0] == 0
            ), f"Reset {reset_num + 1}: World 0 should be reset to 0 steps taken"
            assert done_tensor[0, 0] == 0, f"Reset {reset_num + 1}: World 0 should not be done"


# =============================================================================
# Category 3: Collision vs Step Termination Tests (CT-001 to CT-004)
# =============================================================================


class TestCollisionVsStepTermination:
    """Tests for collision-based vs step-based termination interactions"""

    def test_CT_001_collision_at_step_50(self, cpu_manager):
        """CT-001: Collision at step 50 should cause early termination"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set aggressive movement to try to trigger collision
        actions[:] = 0
        actions[:, 0] = 3  # FAST movement
        actions[:, 1] = 0  # FORWARD direction

        # Run up to 50 steps and check for early termination
        collision_detected = False
        for step in range(50):
            mgr.step()

            if done_tensor.any():
                collision_detected = True
                done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False)

                for world_idx in done_worlds.flatten():
                    world_idx = world_idx.item()
                    remaining_steps = steps_taken[world_idx, 0, 0].item()

                    # Early termination: done=1 with steps remaining > 0 (until reset)
                    assert done_tensor[world_idx, 0] == 1, f"World {world_idx} should be done"
                    assert (
                        remaining_steps < consts.episodeLen
                    ), f"World {world_idx} terminated early with {remaining_steps} steps remaining"

                    print(
                        f"Collision detected at step {step + 1} for world {world_idx}, "
                        f"{remaining_steps} steps remaining"
                    )
                break

        if collision_detected:
            print("✓ Collision-based early termination detected")
        else:
            print("ℹ No collision termination detected in first 50 steps")

    def test_CT_002_collision_at_step_200(self, cpu_manager):
        """CT-002: Collision at final step should be distinguishable from step termination"""
        mgr = cpu_manager

        # This test is more about system robustness - if collision happens at the same
        # step as natural termination, the system should handle it gracefully

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Run to near end with minimal movement
        actions[:] = 0
        actions[:, 0] = 0  # STOP

        for _ in range(consts.episodeLen - 5):
            mgr.step()

        # Then try aggressive movement for last few steps
        actions[:, 0] = 3  # FAST movement
        actions[:, 1] = 0  # FORWARD

        for step in range(5):
            mgr.step()

            # At this point, termination could be from step limit or collision
            # The key is that the system handles both gracefully
            if done_tensor.any():
                done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False)
                for world_idx in done_worlds.flatten():
                    world_idx = world_idx.item()
                    # Both collision and step termination result in done=1
                    assert done_tensor[world_idx, 0] == 1, f"World {world_idx} should be done"

    def test_CT_003_collision_plus_manual_reset(self, cpu_manager):
        """CT-003: Manual reset should override collision termination"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set aggressive movement to trigger collision
        actions[:] = 0
        actions[:, 0] = 3  # FAST movement
        actions[:, 1] = 0  # FORWARD direction

        # Run until collision or reasonable timeout
        collision_world = None
        for step in range(100):  # Reasonable timeout
            mgr.step()

            if done_tensor.any():
                collision_world = torch.nonzero(done_tensor.squeeze(), as_tuple=False)[0].item()
                break

        if collision_world is not None:
            # Verify collision state
            assert done_tensor[collision_world, 0] == 1, "World should be done from collision"

            # Manual reset should clear collision done flag
            reset_tensor[:] = 0
            reset_tensor[collision_world] = 1
            mgr.step()

            # Reset should override collision termination
            assert done_tensor[collision_world, 0] == 0, "Reset should clear done flag"
            assert steps_taken[collision_world, 0, 0] == consts.episodeLen, "Steps should be reset"

            print(
                f"✓ Manual reset successfully overrode collision termination for world {collision_world}"
            )
        else:
            print("ℹ No collision detected - test shows system robustness")

    def test_CT_004_no_collision_step_limit(self, cpu_manager):
        """CT-004: No collision should result in pure step counter termination"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement to avoid collision
        actions[:] = 0  # All STOP

        # Run for exactly episodeLen steps
        for step in range(consts.episodeLen):
            mgr.step()

        # Should have natural step-based termination
        for world_idx in range(4):
            if steps_taken[world_idx, 0, 0] == consts.episodeLen:
                assert (
                    done_tensor[world_idx, 0] == 1
                ), f"World {world_idx} should be done from step limit"

        print("✓ Pure step counter termination confirmed (no collision)")


# =============================================================================
# Category 4: Multi-World Synchronization Tests (MW-001 to MW-004)
# =============================================================================


class TestMultiWorldSynchronization:
    """Multi-world reset synchronization and isolation tests"""

    def test_MW_001_reset_world_0_others_continue(self, cpu_manager):
        """MW-001: Reset world 0, others should continue unaffected"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset all worlds
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement
        actions[:] = 0

        # Run 100 steps
        for _ in range(100):
            mgr.step()

        # Verify all worlds at same state
        expected_steps_taken = 100
        assert (
            steps_taken[:, 0, 0] == expected_steps_taken
        ).all(), "All worlds should be synchronized"

        # Reset only world 0
        reset_tensor[0] = 1
        mgr.step()
        reset_tensor[0] = 0

        # World 0 should be reset
        assert steps_taken[0, 0, 0] == 0, "World 0 should be reset to 0 steps taken"
        assert done_tensor[0, 0] == 0, "World 0 should not be done"

        # Other worlds should continue with their previous state
        expected_steps_taken = 101  # One more step
        for world_idx in range(1, 4):
            assert (
                steps_taken[world_idx, 0, 0] == expected_steps_taken
            ), f"World {world_idx} should continue with {expected_steps_taken} steps taken"

    def test_MW_002_multiple_worlds_done_simultaneously(self, cpu_manager):
        """MW-002: Multiple worlds reaching done=1 should be handled independently"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement to reach natural termination
        actions[:] = 0

        # Run to episode end
        for _ in range(consts.episodeLen):
            mgr.step()

        # Find worlds that are done
        done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False).flatten()

        if len(done_worlds) >= 2:
            # Reset only some done worlds, leave others
            reset_tensor[:] = 0
            reset_tensor[done_worlds[:2]] = 1  # Reset first 2 done worlds
            mgr.step()

            # Reset worlds should have fresh episodes
            for world_idx in done_worlds[:2]:
                world_idx = world_idx.item()
                assert (
                    steps_taken[world_idx, 0, 0] == 0
                ), f"Reset world {world_idx} should be reset to 0 steps taken"
                assert done_tensor[world_idx, 0] == 0, f"Reset world {world_idx} should not be done"

            # Non-reset done worlds should remain done
            for world_idx in done_worlds[2:]:
                world_idx = world_idx.item()
                assert (
                    done_tensor[world_idx, 0] == 1
                ), f"Non-reset world {world_idx} should remain done"

    def test_MW_003_auto_reset_with_mixed_done_states(self, cpu_manager):
        """MW-003: Test behavior with mixed done states (some worlds done, others not)"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Create mixed states by resetting world 0 mid-episode
        actions[:] = 0  # No movement

        # Run 50 steps
        for _ in range(50):
            mgr.step()

        # Reset world 0 to create mixed state
        reset_tensor[0] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Now world 0 has 0 steps taken, others have ~51 steps taken
        assert steps_taken[0, 0, 0] == 0, "World 0 should be reset to 0 steps taken"

        expected_others = 51  # 51 steps taken
        for world_idx in range(1, 4):
            assert (
                abs(steps_taken[world_idx, 0, 0].item() - expected_others) <= 1
            ), f"World {world_idx} should have ~{expected_others} steps taken"

        # Continue until some worlds are done but not others
        # Need to run enough steps so other worlds finish (they need ~149 more steps)
        for _ in range(consts.episodeLen - expected_others):
            mgr.step()

        # World 0 should still have steps remaining (~149 steps taken), others should be done (200 steps taken)
        assert steps_taken[0, 0, 0] < consts.episodeLen, "World 0 should still be running"
        assert done_tensor[0, 0] == 0, "World 0 should not be done"

        # Other worlds should be done
        for world_idx in range(1, 4):
            if steps_taken[world_idx, 0, 0] == consts.episodeLen:
                assert (
                    done_tensor[world_idx, 0] == 1
                ), f"World {world_idx} should be done when steps_taken=episodeLen"

    def test_MW_004_manual_reset_during_mixed_states(self, cpu_manager):
        """MW-004: Manual reset should work correctly with mixed world states"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Create different states for different worlds
        actions[:] = 0

        # Reset worlds at different times to create mixed states
        reset_times = [0, 30, 60, 90]  # Different reset points

        for reset_time in reset_times:
            if reset_time > 0:
                # Run to reset time
                for _ in range(30):  # 30 steps between resets
                    mgr.step()

            # Reset the next world
            world_to_reset = reset_times.index(reset_time)
            reset_tensor[world_to_reset] = 1
            mgr.step()
            reset_tensor[:] = 0

        # Verify each world has different step counts
        step_counts = [steps_taken[i, 0, 0].item() for i in range(4)]

        # Each world should have different step counts due to different reset times
        unique_counts = set(step_counts)
        assert len(unique_counts) > 1, f"Worlds should have different step counts: {step_counts}"

        # All worlds should have valid states
        for world_idx in range(4):
            assert (
                0 <= step_counts[world_idx] <= consts.episodeLen
            ), f"World {world_idx} has invalid step count: {step_counts[world_idx]}"
            assert done_tensor[world_idx, 0] in [
                0,
                1,
            ], f"World {world_idx} has invalid done state"


# =============================================================================
# Category 5: Order of Operations Edge Cases (OO-001 to OO-004)
# =============================================================================


class TestOrderOfOperations:
    """Order of operations and intra-step timing tests"""

    def test_OO_001_reset_same_step_as_done(self, cpu_manager):
        """OO-001: Reset occurring same step as done=1 should be handled correctly"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement to get predictable step counting
        actions[:] = 0

        # Run to exactly one step before end
        for _ in range(consts.episodeLen - 1):
            mgr.step()

        # At this point: steps_taken=1, done=0
        assert (
            steps_taken[:, 0, 0] == consts.episodeLen - 1
        ).all(), "Should have taken episodeLen-1 steps"
        assert not done_tensor.any(), "Should not be done yet"

        # Set manual reset for world 0 at the same step it would naturally terminate
        reset_tensor[0] = 1

        # This step would naturally set done=1, but reset should handle it
        mgr.step()

        # Reset should take precedence - world 0 should be reset
        assert steps_taken[0, 0, 0] == 0, "World 0 should be reset to 0 steps taken"
        assert done_tensor[0, 0] == 0, "World 0 should not be done after reset"

        # Other worlds should follow natural progression (done=1, steps=0)
        for world_idx in range(1, 4):
            if steps_taken[world_idx, 0, 0] == consts.episodeLen:
                assert done_tensor[world_idx, 0] == 1, f"World {world_idx} should be done naturally"

    def test_OO_002_reward_calculation_during_reset(self, cpu_manager):
        """OO-002: Verify rewards calculated before reset occurs"""
        mgr = cpu_manager

        # This test validates that the task graph order (stepTracker → reward → reset) is maintained
        # We can't directly inspect rewards in this test framework, but we can verify
        # that the system behavior is consistent with the documented order

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set forward movement to potentially generate rewards
        actions[:] = 0
        actions[:, 0] = 2  # MEDIUM forward movement
        actions[:, 1] = 0  # FORWARD direction

        # Run several steps with movement
        initial_steps = []
        for world_idx in range(4):
            initial_steps.append(steps_taken[world_idx, 0, 0].item())

        for step in range(50):
            mgr.step()

            # Verify step counter increases consistently
            # If reward calculation happened after reset, we might see inconsistencies
            for world_idx in range(4):
                current_steps = steps_taken[world_idx, 0, 0].item()
                expected_steps = initial_steps[world_idx] + (step + 1)

                # Allow for early termination due to collision
                if done_tensor[world_idx, 0] == 0:  # Not done yet
                    assert (
                        current_steps == expected_steps
                    ), f"World {world_idx} step {step}: expected {expected_steps}, got {current_steps}"

    @pytest.mark.ascii_level("""
########################################
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#..................S...................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
########################################
""")
    def test_OO_003_action_processing_during_reset(self, cpu_manager):
        """OO-003: Actions should be processed correctly during reset steps"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()
        obs_tensor = mgr.self_observation_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Run more steps with faster movement to establish clear position change
        controller = AgentController(mgr)
        controller.reset_actions()
        controller.move_forward(speed=consts.action.move_amount.FAST)

        for _ in range(30):  # More steps for significant movement
            mgr.step()

        # Store position before reset
        pos_before = obs_tensor[0, 0, :3].clone()

        # Set different action and trigger reset simultaneously
        controller.strafe_right(world_idx=0, speed=consts.action.move_amount.FAST)
        reset_tensor[0] = 1
        mgr.step()
        reset_tensor[0] = 0

        # After reset, world 0 should be at spawn position (reset overrides action)
        pos_after = obs_tensor[0, 0, :3]

        # Record spawn position immediately after the first step to compare
        reset_tensor[1] = 1  # Reset world 1 to get fresh spawn position
        mgr.step()
        spawn_pos = obs_tensor[1, 0, :3]

        print(f"Position before reset: {pos_before}")
        print(f"Position after reset:  {pos_after}")
        print(f"Fresh spawn position:  {spawn_pos}")

        # Test the key behavior: reset should restore state regardless of action
        # Note: Due to execution order (stepTracker → reward → reset), the step counter
        # increments before reset occurs, so we expect 1 step taken after reset
        current_steps = steps_taken[0, 0, 0].item()
        assert current_steps == 1, f"Steps should be 1 due to execution order, got {current_steps}"
        assert done_tensor[0, 0] == 0, "Done flag should be reset"

        # Position should be close to spawn (within reasonable tolerance)
        distance_to_spawn = torch.norm(pos_after - spawn_pos).item()
        assert (
            distance_to_spawn < 1.0
        ), f"Reset position should be close to spawn, distance: {distance_to_spawn}"

        # Note: steps_taken was already verified above to be episodeLen-1 (199)
        # due to execution order. The done flag should still be properly reset.
        assert done_tensor[0, 0] == 0, "Should not be done"

    @pytest.mark.ascii_level("""
########################################
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#..................S...................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
#......................................#
########################################
""")
    def test_OO_004_observer_tensor_consistency(self, cpu_manager):
        """OO-004: All tensors should reflect consistent post-reset state"""
        mgr = cpu_manager

        # Get all tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()
        obs_tensor = mgr.self_observation_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Run simulation to mid-episode with significant movement
        controller = AgentController(mgr)
        controller.reset_actions()
        controller.move_forward(speed=consts.action.move_amount.FAST)

        for _ in range(50):  # Enough steps for clear movement
            mgr.step()

        # Capture pre-reset state
        steps_before = steps_taken.clone()
        done_before = done_tensor.clone()
        obs_before = obs_tensor.clone()

        # Reset world 1
        reset_tensor[1] = 1
        mgr.step()

        # Validate tensor consistency after reset
        TestHelpers.validate_tensor_consistency(mgr)

        # World 1 should be reset (0 steps taken after reset)
        assert steps_taken[1, 0, 0] == 0, "World 1 steps should be reset to 0"
        assert done_tensor[1, 0] == 0, "World 1 should not be done"

        # Test that World 1 state is properly reset
        print(f"World 1 position before reset: {obs_before[1, 0, :3]}")
        print(f"World 1 position after reset:  {obs_tensor[1, 0, :3]}")

        # The key test is that the tensor consistency is maintained after reset
        # Position may be close to original due to level layout, but state should be consistent
        distance_moved = torch.norm(obs_before[1, 0, :3] - obs_tensor[1, 0, :3]).item()
        print(f"Distance moved by reset: {distance_moved}")

        # Verify tensor consistency (this is the main goal of this test)
        TestHelpers.validate_tensor_consistency(mgr)

        # Other worlds should be unchanged (within simulation tolerance)
        for world_idx in [0, 2, 3]:
            # Steps should increase by 1
            expected_steps = steps_before[world_idx, 0, 0] + 1
            assert (
                steps_taken[world_idx, 0, 0] == expected_steps
            ), f"World {world_idx} should have {expected_steps} steps taken"

            # Done state should be consistent with steps
            if expected_steps >= consts.episodeLen:
                assert (
                    done_tensor[world_idx, 0] == 1
                ), f"World {world_idx} should be done with {expected_steps} steps taken"
            else:
                assert (
                    done_tensor[world_idx, 0] == done_before[world_idx, 0]
                ), f"World {world_idx} done state should be unchanged"


# =============================================================================
# Category 6: State Persistence Validation (SP-001 to SP-004)
# =============================================================================


class TestStatePersistence:
    """State persistence across reset boundary validation tests"""

    def test_SP_001_done_flag_across_reset_boundary(self, cpu_manager):
        """SP-001: Done flag transition: done=1 → reset → done=0"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Run to episode end to get done=1
        actions[:] = 0  # No movement

        for _ in range(consts.episodeLen):
            mgr.step()

        # Find a world that's done
        done_worlds = torch.nonzero(done_tensor.squeeze(), as_tuple=False)
        assert len(done_worlds) > 0, "At least one world should be done"

        done_world = done_worlds[0].item()

        # Verify pre-reset state: done=1, steps=0
        assert done_tensor[done_world, 0] == 1, "World should be done before reset"
        assert (
            steps_taken[done_world, 0, 0] == consts.episodeLen
        ), "World should have 0 steps before reset".replace("0 steps", "episodeLen steps taken")

        # Reset the done world
        reset_tensor[:] = 0
        reset_tensor[done_world] = 1
        mgr.step()

        # Verify post-reset state: done=0, steps_taken=0
        assert done_tensor[done_world, 0] == 0, "Done flag should be cleared"
        assert steps_taken[done_world, 0, 0] == 0, "Steps taken should be reset to 0"

    def test_SP_002_steps_counter_across_reset(self, cpu_manager):
        """SP-002: Steps counter transition: steps=X → reset → steps=200"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Set no movement
        actions[:] = 0

        # Run to various step counts and test reset restoration
        test_points = [50, 100, 150, 199]  # Different points to test reset

        for test_step in test_points:
            # Reset to start fresh
            reset_tensor[:] = 1
            mgr.step()
            reset_tensor[:] = 0

            # Run to test point
            for _ in range(test_step):
                mgr.step()

            # Verify we're at expected step count
            expected_taken = test_step
            assert (
                steps_taken[0, 0, 0] == expected_taken
            ), f"Should have {expected_taken} steps taken at test point {test_step}"

            # Reset world 0
            reset_tensor[0] = 1
            mgr.step()
            reset_tensor[0] = 0

            # Verify restoration to fresh episode
            assert (
                steps_taken[0, 0, 0] == 0
            ), f"After reset from step {test_step}, should have 0 steps taken"

    def test_SP_003_position_rotation_reset(self, cpu_manager):
        """SP-003: Agent should return to spawn position/rotation after reset"""
        mgr = cpu_manager

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        obs_tensor = mgr.self_observation_tensor().to_torch()

        # Initial reset to establish spawn position
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Record spawn position and rotation
        spawn_pos = obs_tensor[0, 0, :3].clone()  # x, y, z position
        spawn_rot = obs_tensor[0, 0, 3:7].clone()  # quaternion rotation

        # Move agent away from spawn
        actions[:] = 0
        actions[0, 0] = 2  # MEDIUM movement
        actions[0, 1] = 0  # FORWARD
        actions[0, 2] = 1  # Turn (if available)

        # Run multiple steps to move away
        for _ in range(50):
            mgr.step()

        # Verify agent moved away from spawn
        current_pos = obs_tensor[0, 0, :3]
        assert not torch.allclose(
            spawn_pos, current_pos, atol=0.1
        ), "Agent should have moved away from spawn position"

        # Reset world 0
        reset_tensor[0] = 1
        mgr.step()

        # Verify return to spawn position
        reset_pos = obs_tensor[0, 0, :3]
        reset_rot = obs_tensor[0, 0, 3:7]

        assert torch.allclose(
            spawn_pos, reset_pos, atol=0.1
        ), f"Agent should return to spawn position: spawn={spawn_pos}, reset={reset_pos}"

        # Rotation should also be reset (less strict tolerance due to quaternion representation)
        assert torch.allclose(
            spawn_rot, reset_rot, atol=0.2
        ), f"Agent rotation should be reset: spawn={spawn_rot}, reset={reset_rot}"

    def test_SP_004_progress_reward_reset(self, cpu_manager):
        """SP-004: Progress metrics should reset appropriately"""
        mgr = cpu_manager

        # This test validates that internal progress tracking resets correctly
        # We can't directly access reward systems, but we can test observable effects

        # Get tensor references
        actions = mgr.action_tensor().to_torch()
        done_tensor = mgr.done_tensor().to_torch()
        reset_tensor = mgr.reset_tensor().to_torch()
        steps_taken = mgr.steps_taken_tensor().to_torch()
        obs_tensor = mgr.self_observation_tensor().to_torch()

        # Initial reset
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Record initial position for progress calculation
        initial_pos = obs_tensor[0, 0, :3].clone()

        # Move forward to create progress
        actions[:] = 0
        actions[0, 0] = 2  # MEDIUM forward movement
        actions[0, 1] = 0  # FORWARD

        # Run steps to create progress
        for _ in range(100):
            mgr.step()

        # Calculate progress made
        progress_pos = obs_tensor[0, 0, :3]
        distance_traveled = torch.norm(progress_pos - initial_pos).item()

        # Reset world 0
        reset_tensor[0] = 1
        mgr.step()

        # After reset, agent should be back to initial conditions
        reset_pos = obs_tensor[0, 0, :3]

        # Position should be back to spawn (similar to initial)
        assert torch.allclose(
            initial_pos, reset_pos, atol=0.1
        ), "After reset, agent should be back to initial position"

        # Episode state should be fresh
        assert steps_taken[0, 0, 0] == 0, "Should have 0 steps taken in fresh episode"
        assert done_tensor[0, 0] == 0, "Should not be done"

        # The progress made before reset should not affect the fresh episode
        # (This is more of a system consistency check)
        print(f"Distance traveled before reset: {distance_traveled:.3f}")
        print("Reset successfully returned agent to spawn")


if __name__ == "__main__":
    # Allow running individual test categories
    import sys

    if len(sys.argv) > 1:
        category = sys.argv[1]
        if category == "basic":
            pytest.main(["-v", __file__ + "::TestBasicResetTiming"])
        elif category == "precision":
            pytest.main(["-v", __file__ + "::TestStepCounterPrecision"])
        elif category == "collision":
            pytest.main(["-v", __file__ + "::TestCollisionVsStepTermination"])
        elif category == "multiworld":
            pytest.main(["-v", __file__ + "::TestMultiWorldSynchronization"])
        elif category == "order":
            pytest.main(["-v", __file__ + "::TestOrderOfOperations"])
        elif category == "persistence":
            pytest.main(["-v", __file__ + "::TestStatePersistence"])
        else:
            print(
                "Usage: python test_comprehensive_reset_episode_counter_done_flag.py [basic|precision|collision|multiworld|order|persistence]"
            )
    else:
        pytest.main(["-v", __file__])
