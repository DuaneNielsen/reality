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
            assert not observer.get_done_flag(
                0
            ), f"Episode should continue after east wall collision at step {step}"

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
            assert not observer.get_done_flag(
                0
            ), f"Episode should continue after south wall collision at step {step}"

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
        assert actions.shape[1] == 1, "Should have 1 agent per world"
        assert actions.shape[2] == 3, "Should have 3 action components"

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

        # Reset for next test - collision should have terminated, triggering auto-reset
        if north_terminated:
            # Wait a step for auto-reset to take effect
            controller.reset_actions()
            mgr.step()

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
