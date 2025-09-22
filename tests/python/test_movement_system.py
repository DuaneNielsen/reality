#!/usr/bin/env python3
"""
Test agent movement system including forward movement, strafing, and rotation.
"""

import math

import pytest
from test_helpers import AgentController, ObservationReader

import madrona_escape_room

# Large open level for movement testing - 32x16 room with agent in center facing south
LARGE_OPEN_LEVEL = {
    "ascii": """################################
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............S...............#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
#..............................#
################################""",
    "scale": 2.5,
    "agent_facing": [3.14159],  # Face south (180 degrees) to move away from north wall
}


@pytest.mark.xfail(
    reason="Test expects incremental rewards, now using target-based completion rewards"
)
@pytest.mark.ascii_level(LARGE_OPEN_LEVEL["ascii"])
@pytest.mark.spec("docs/specs/sim.md", "movementSystem")
def test_forward_movement(cpu_manager):
    """Test agent moving forward for entire episode"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset to start fresh episode
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1  # Reset all worlds
    mgr.step()
    reset_tensor[:] = 0

    # Set forward movement for all steps
    controller.reset_actions()
    controller.move_forward(speed=madrona_escape_room.action.move_amount.MEDIUM)

    # Debug: print action values and tensor info
    actions = mgr.action_tensor().to_torch()
    print(f"Action tensor shape: {actions.shape}")
    print(
        f"Action values set: moveAmount={actions[0, 0]}, moveAngle={actions[0, 1]}, "
        f"rotate={actions[0, 2]}"
    )
    print(f"Full action for world 0: {actions[0].tolist()}")

    # Run for entire episode (200 steps)
    for step in range(200):
        mgr.step()
        if step < 5:  # Check first few steps
            print(f"Step {step}: actions = {actions[0].tolist()}")

    # Check final state
    final_pos = observer.get_position(0)
    final_y = final_pos[1]
    max_y = observer.get_max_y_progress(0)

    # Agent should have moved forward significantly
    print(f"Final Y position: {final_y}")
    print(f"Max Y reached: {max_y}")

    # Basic sanity check - agent should have moved forward
    assert max_y > 0.1, f"Agent should have moved forward, but max_y = {max_y}"


@pytest.mark.ascii_level(LARGE_OPEN_LEVEL["ascii"])
def test_strafe_left(cpu_manager):
    """Test agent strafing left (moving left while maintaining forward orientation)"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset to start fresh episode
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1  # Reset all worlds
    mgr.step()
    reset_tensor[:] = 0

    # Get initial position
    initial_pos = observer.get_position(0)
    initial_x, initial_y = initial_pos[0], initial_pos[1]
    initial_theta = observer.get_rotation(0)

    print(f"Initial position: x={initial_x:.3f}, y={initial_y:.3f}, theta={initial_theta:.3f}")

    # IMPORTANT: When movement forces are applied to an agent with non-zero rotation,
    # the physics system generates small torques (origin unknown) that cause the
    # rotation to decay to zero over ~20 steps. Although agentZeroVelSystem zeroes
    # angular velocity each frame, the torques persist during this settling period.
    # We must apply forces and wait for the rotation to settle before testing.
    controller.reset_actions()
    controller.stop()  # No movement during settling

    # Run settling phase (20 steps)
    print("Letting agent settle without forces...")
    prev_theta = initial_theta
    for step in range(20):
        mgr.step()

        # Get observations
        current_theta = observer.get_rotation(0)

        # Calculate angular velocity
        angular_vel = current_theta - prev_theta
        prev_theta = current_theta

        # Print every 10 steps
        if step % 10 == 9:
            print(
                f"Step {step + 1} (settling): theta={current_theta:.6f}, "
                f"angular_vel={angular_vel:.6f}"
            )

    # Get settled baseline position
    settled_pos = observer.get_position(0)
    settled_x, settled_y = settled_pos[0], settled_pos[1]
    settled_theta = observer.get_rotation(0)
    print(f"Settled baseline: x={settled_x:.3f}, y={settled_y:.3f}, theta={settled_theta:.6f}")

    # Now test strafing from this baseline
    print("\nTesting strafe left from settled position...")
    controller.reset_actions()
    controller.strafe_left(speed=madrona_escape_room.action.move_amount.MEDIUM)

    # Run strafe test for 50 steps
    for step in range(50):
        mgr.step()

        # Get observations
        current_pos = observer.get_position(0)
        current_x, current_y = current_pos[0], current_pos[1]
        current_theta = observer.get_rotation(0)

        # Print every 10 steps
        if step % 10 == 9:
            print(
                f"Step {step + 1} (strafe): x={current_x:.3f}, y={current_y:.3f}, "
                f"theta={current_theta:.6f}"
            )

    # Final values
    final_pos = observer.get_position(0)
    final_x, final_y = final_pos[0], final_pos[1]
    final_theta = observer.get_rotation(0)

    print(f"Final position: x={final_x:.3f}, y={final_y:.3f}, theta={final_theta:.3f}")
    print(f"Delta: dx={final_x - settled_x:.3f}, dy={final_y - settled_y:.3f}")

    # Calculate expected movement direction based on agent's orientation
    # Convert theta from normalized (-1 to 1) to radians
    agent_angle = settled_theta * math.pi

    # Apply 2D rotation to transform local force to world frame
    local_fx = -1.0  # Strafe left in local frame
    local_fy = 0.0
    expected_dx = local_fx * math.cos(agent_angle) - local_fy * math.sin(agent_angle)
    expected_dy = local_fx * math.sin(agent_angle) + local_fy * math.cos(agent_angle)

    actual_dx = final_x - settled_x
    actual_dy = final_y - settled_y

    # Normalize vectors for comparison
    actual_length = math.sqrt(actual_dx**2 + actual_dy**2)
    expected_length = math.sqrt(expected_dx**2 + expected_dy**2)

    if actual_length > 0.01:  # Avoid division by zero
        # Normalize both vectors
        actual_dx_norm = actual_dx / actual_length
        actual_dy_norm = actual_dy / actual_length
        expected_dx_norm = expected_dx / expected_length
        expected_dy_norm = expected_dy / expected_length

        print(f"Expected direction: ({expected_dx_norm:.3f}, {expected_dy_norm:.3f})")
        print(f"Actual direction: ({actual_dx_norm:.3f}, {actual_dy_norm:.3f})")

        # Check movement direction (should be roughly 90 degrees left of facing)
        dot_product = actual_dx_norm * expected_dx_norm + actual_dy_norm * expected_dy_norm
        print(f"Movement direction alignment: {dot_product:.3f} (1.0 = perfect)")
        assert dot_product > 0.9, "Agent should strafe left, but moved in wrong direction"

    # Just verify that movement happened
    assert actual_length > 0.1, f"Agent should have moved, but only moved {actual_length:.3f} units"


@pytest.mark.ascii_level(LARGE_OPEN_LEVEL["ascii"])
def test_strafe_right(cpu_manager):
    """Test agent strafing right"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0

    # Let agent settle first
    controller.reset_actions()
    controller.stop()

    # Run settling phase
    for _ in range(20):
        mgr.step()

    # Get settled position
    settled_pos = observer.get_position(0)
    settled_x = settled_pos[0]

    # Strafe right
    controller.reset_actions()
    controller.strafe_right(speed=madrona_escape_room.action.move_amount.MEDIUM)

    # Run for 50 steps
    for _ in range(50):
        mgr.step()

    # Check final position
    final_pos = observer.get_position(0)
    final_x = final_pos[0]

    # Agent should have moved right (positive X)
    delta_x = final_x - settled_x
    print(f"Strafe right delta X: {delta_x:.3f}")
    assert delta_x > 0.1, f"Agent should have moved right, but delta_x = {delta_x}"


@pytest.mark.ascii_level(LARGE_OPEN_LEVEL["ascii"])
def test_backward_movement(cpu_manager):
    """Test agent moving backward"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0

    # Get initial position
    initial_y = observer.get_position(0)[1]

    # Move backward
    controller.reset_actions()
    controller.move_backward(speed=madrona_escape_room.action.move_amount.MEDIUM)

    # Run for 50 steps
    for _ in range(50):
        mgr.step()

    # Check final position
    final_y = observer.get_position(0)[1]

    # Agent should have moved backward (negative Y)
    delta_y = final_y - initial_y
    print(f"Backward movement delta Y: {delta_y:.3f}")
    assert delta_y < -0.1, f"Agent should have moved backward, but delta_y = {delta_y}"


@pytest.mark.ascii_level(LARGE_OPEN_LEVEL["ascii"])
def test_rotation_in_place(cpu_manager):
    """Test agent rotating in place without movement"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    # Reset world
    reset_tensor = mgr.reset_tensor().to_torch()
    reset_tensor[:] = 1
    mgr.step()
    reset_tensor[:] = 0

    # Get initial rotation
    initial_theta = observer.get_rotation(0)
    initial_pos = observer.get_position(0)

    # Rotate left in place
    controller.reset_actions()
    controller.rotate_only(rotation=madrona_escape_room.action.rotate.SLOW_LEFT)

    # Run for 30 steps
    for _ in range(30):
        mgr.step()

    # Check rotation changed
    final_theta = observer.get_rotation(0)
    final_pos = observer.get_position(0)

    # Position should not change much
    pos_delta = math.sqrt(
        (final_pos[0] - initial_pos[0]) ** 2 + (final_pos[1] - initial_pos[1]) ** 2
    )
    print(f"Position delta during rotation: {pos_delta:.3f}")
    assert pos_delta < 0.1, f"Agent should rotate in place, but moved {pos_delta:.3f} units"

    # Rotation should have changed
    theta_delta = final_theta - initial_theta
    print(f"Rotation delta: {theta_delta:.3f}")
    assert (
        abs(theta_delta) > 0.05
    ), f"Agent should have rotated, but theta only changed by {theta_delta:.3f}"


@pytest.mark.ascii_level(LARGE_OPEN_LEVEL["ascii"])
def test_movement_speed_differences(cpu_manager):
    """Test that different movement speeds produce different distances"""
    mgr = cpu_manager
    controller = AgentController(mgr)
    observer = ObservationReader(mgr)

    distances = {}
    speeds = [
        ("STOP", madrona_escape_room.action.move_amount.STOP),
        ("SLOW", madrona_escape_room.action.move_amount.SLOW),
        ("MEDIUM", madrona_escape_room.action.move_amount.MEDIUM),
        ("FAST", madrona_escape_room.action.move_amount.FAST),
    ]

    for speed_name, speed_value in speeds:
        # Reset world
        reset_tensor = mgr.reset_tensor().to_torch()
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        # Get initial position
        initial_y = observer.get_position(0)[1]

        # Move forward at specified speed
        controller.reset_actions()
        controller.move_forward(speed=speed_value)

        # Debug: print action values
        actions = mgr.action_tensor().to_torch()
        print(f"{speed_name} (value={speed_value}): action tensor = {actions[0].tolist()}")

        # Run for fewer steps to avoid hitting walls at higher speeds
        for _ in range(5):
            mgr.step()

        # Measure distance
        final_y = observer.get_position(0)[1]
        distance = final_y - initial_y
        distances[speed_name] = distance
        print(f"{speed_name} speed: moved {distance:.3f} units")

    # Verify speed ordering (movement is in negative Y direction)
    assert abs(distances["STOP"]) < 0.01, "STOP should produce no movement"
    assert abs(distances["SLOW"]) > abs(distances["STOP"]), "SLOW should move more than STOP"
    assert abs(distances["MEDIUM"]) > abs(distances["SLOW"]), "MEDIUM should move more than SLOW"
    assert abs(distances["FAST"]) > abs(distances["MEDIUM"]), "FAST should move more than MEDIUM"
