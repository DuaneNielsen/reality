"""
Test collision-based episode termination with per-tile collision flags.

Tests a 3x3 enclosed level where the agent can move in 4 directions to collide
with different objects that have specific collision termination behavior.
"""

import pytest
from test_helpers import AgentController, ObservationReader

from madrona_escape_room.generated_constants import consts

# Custom 3x3 enclosed level with specific collision behaviors
# Layout:
#   #C#
#   OS#
#   ###
# Where:
# S = Spawn (agent facing north)
# C = North: Cube with done_on_collide=true (should terminate)
# # = East: Wall with done_on_collide=false (should continue)
# O = West: Cylinder with done_on_collide=true (should terminate)
# # = South: Wall with done_on_collide=false (should continue)


@pytest.mark.json_level(
    {
        "ascii": "#C#\nOS#\n###",
        "tileset": {
            "#": {"asset": "wall", "done_on_collision": False},  # Non-terminating walls
            "C": {"asset": "cube", "done_on_collision": True},  # Terminating cube (north)
            "O": {"asset": "cylinder", "done_on_collision": True},  # Terminating cylinder (west)
            "S": {"asset": "spawn"},  # Agent spawn point
        },
        "scale": 2.5,
        "agent_facing": [0.0],  # Face north (0 radians)
        "name": "collision_test_3x3",
    }
)
class TestCollisionTermination:
    """Test collision-based episode termination with custom per-tile collision flags."""

    def test_north_collision_terminates(self, cpu_manager):
        """Test collision with terminating cube (north) ends episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state - episode should be running
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move north toward the terminating cube
        for _ in range(5):  # Multiple steps to ensure collision
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Check if collision terminated the episode (single agent in world 0)
            if observer.get_done_flag(0):
                return  # SUCCESS - episode terminated as expected

        # If we reach here, episode didn't terminate
        assert False, "Episode should have terminated from north collision with terminating cube"

    def test_east_collision_continues(self, cpu_manager):
        """Test collision with non-terminating wall (east) continues episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move east toward the non-terminating wall
        for step in range(10):  # More steps to test continued simulation
            controller.reset_actions()
            controller.strafe_right(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Episode should continue running even after collision
            assert not observer.get_done_flag(0), (
                f"Episode should continue after east wall collision at step {step}"
            )

        # SUCCESS - episode continued running after collision with non-terminating wall

    def test_west_collision_terminates(self, cpu_manager):
        """Test collision with terminating cylinder (west) ends episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move west toward the terminating cylinder
        for _ in range(5):  # Multiple steps to ensure collision
            controller.reset_actions()
            controller.strafe_left(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Check if collision terminated the episode
            if observer.get_done_flag(0):
                return  # SUCCESS - episode terminated as expected

        # If we reach here, episode didn't terminate
        assert False, "Episode should have terminated from west collision with terminating cylinder"

    def test_south_collision_continues(self, cpu_manager):
        """Test collision with non-terminating wall (south) continues episode."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"

        # Move south toward the non-terminating wall
        for step in range(10):  # More steps to test continued simulation
            controller.reset_actions()
            controller.move_backward(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Episode should continue running even after collision
            assert not observer.get_done_flag(0), (
                f"Episode should continue after south wall collision at step {step}"
            )

        # SUCCESS - episode continued running after collision with non-terminating wall

    def test_level_configuration_validation(self, cpu_manager):
        """Validate the custom level has correct collision configuration."""
        mgr = cpu_manager

        # Verify the custom level was loaded correctly
        # This is more of a sanity check for the level compilation
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset and get initial state
        controller.reset_actions()
        mgr.step()

        # Check that we can access game state (indicates level loaded successfully)
        initial_position = observer.get_normalized_position(0)
        assert initial_position is not None, "Should be able to read agent position"

        # Check that the episode starts in running state
        assert not observer.get_done_flag(0), "Episode should start in running state"

        # Verify we have proper tensor access
        actions = mgr.action_tensor().to_torch()
        assert actions.shape[0] == 4, "Should have 4 worlds (from cpu_manager fixture)"
        assert actions.shape[1] == 3, "Should have 3 action components (single agent per world)"

    def test_collision_behavior_differences(self, cpu_manager):
        """Test that different collision objects actually behave differently."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Test terminating collision (north - cube)
        controller.reset_actions()
        mgr.step()

        north_terminated = False
        for _ in range(8):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            if observer.get_done_flag(0):
                north_terminated = True
                break

        # Reset for next test - collision terminated, manual reset needed since auto_reset=False
        if north_terminated:
            # Manually trigger reset since auto_reset is disabled in cpu_manager fixture
            reset_tensor = mgr.reset_tensor().to_torch()
            reset_tensor[0] = 1  # Reset world 0
            controller.reset_actions()
            mgr.step()
            reset_tensor[0] = 0  # Clear reset flag

        # Test non-terminating collision (east - wall)
        assert not observer.get_done_flag(0), "Should be reset and running for east test"

        east_terminated = False
        for _ in range(8):
            controller.reset_actions()
            controller.strafe_right(speed=consts.action.move_amount.FAST)
            mgr.step()

            if observer.get_done_flag(0):
                east_terminated = True
                break

        # Validate the different behaviors
        assert north_terminated, "North collision with cube should terminate episode"
        assert not east_terminated, "East collision with wall should not terminate episode"

    def test_collision_reward_penalty(self, cpu_manager):
        """Test that collision with DoneOnCollide objects gives -1 reward."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Verify initial state
        assert not observer.get_done_flag(0), "Episode should not be done initially"
        assert observer.get_reward(0) == 0.0, "Reward should be 0 during episode"

        # Move north toward the terminating cube (DoneOnCollide=True)
        collision_occurred = False
        for step in range(8):  # Multiple steps to ensure collision
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Check if collision terminated the episode
            if observer.get_done_flag(0):
                # Verify -1 reward on collision death
                reward = observer.get_reward(0)
                assert reward == -1.0, f"Expected -1.0 reward on collision, got {reward}"
                print(f"✓ Collision reward test passed: reward = {reward}")
                return

        # If we reach here, collision didn't occur
        assert False, "Collision should have occurred with terminating cube"

    def test_normal_episode_end_vs_collision_rewards(self, cpu_manager):
        """Test that normal episode timeout gives progress reward, collision gives -1."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Test 1: Normal episode timeout (should give progress reward)
        controller.reset_actions()
        mgr.step()

        # Stay still to avoid collision and let episode timeout
        for _ in range(200):  # Run full episode length
            controller.reset_actions()  # No movement
            mgr.step()

        # Check timeout reward (should be progress-based, >= 0)
        timeout_reward = observer.get_reward(0)
        assert observer.get_done_flag(0), "Episode should be done after timeout"
        assert timeout_reward >= 0.0, (
            f"Timeout should give non-negative reward, got {timeout_reward}"
        )
        print(f"✓ Timeout reward: {timeout_reward}")

        # Reset for collision test
        reset_tensor = mgr.reset_tensor().to_torch()
        reset_tensor[0] = 1  # Reset world 0
        mgr.step()
        reset_tensor[0] = 0  # Clear reset flag

        # Test 2: Collision death (should give -1 reward)
        assert not observer.get_done_flag(0), "Episode should be reset and running"

        # Move toward terminating object
        collision_occurred = False
        for step in range(8):
            controller.reset_actions()
            controller.move_forward(speed=consts.action.move_amount.FAST)
            mgr.step()

            if observer.get_done_flag(0):
                collision_occurred = True
                collision_reward = observer.get_reward(0)
                assert collision_reward == -1.0, (
                    f"Expected -1.0 collision reward, got {collision_reward}"
                )
                print(f"✓ Collision reward: {collision_reward}")
                break

        assert collision_occurred, "Collision should have occurred"

        # Verify different rewards for different episode end types
        print(f"✓ Reward comparison: timeout={timeout_reward}, collision={collision_reward}")
        assert timeout_reward > collision_reward, (
            "Timeout reward should be higher than collision penalty"
        )

    def test_non_terminating_collision_no_penalty(self, cpu_manager):
        """Test that collision with non-DoneOnCollide objects doesn't give -1 reward."""
        mgr = cpu_manager
        controller = AgentController(mgr)
        observer = ObservationReader(mgr)

        # Reset to ensure clean state
        controller.reset_actions()
        mgr.step()

        # Move east toward non-terminating wall (DoneOnCollide=False)
        for step in range(10):  # Collide with wall multiple times
            controller.reset_actions()
            controller.strafe_right(speed=consts.action.move_amount.FAST)
            mgr.step()

            # Episode should continue, reward should remain 0
            assert not observer.get_done_flag(0), (
                f"Episode should continue after non-terminating collision at step {step}"
            )
            reward = observer.get_reward(0)
            assert reward == 0.0, (
                f"Reward should be 0 during non-terminating collision, got {reward} at step {step}"
            )

        # Continue until episode timeout to verify normal progress reward
        remaining_steps = 200 - 10  # Already took 10 steps
        for _ in range(remaining_steps):
            controller.reset_actions()  # Stay still
            mgr.step()

        # Should get normal progress reward, not collision penalty
        final_reward = observer.get_reward(0)
        assert observer.get_done_flag(0), "Episode should be done after timeout"
        assert final_reward >= 0.0, (
            f"Non-terminating collision should not affect final reward, got {final_reward}"
        )
        print(f"✓ Non-terminating collision final reward: {final_reward}")
