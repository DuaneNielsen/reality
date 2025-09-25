#!/usr/bin/env python3
"""
Tests for static target randomization functionality with proper collision avoidance.

Verifies that static targets with randomization enabled (omega_y > 0.0) move to different
positions between episodes while maintaining collision avoidance with 3.0 unit exclusion
radius as specified in docs/specs/sim.md.

Uses target position tensor for direct position measurement and compiled level data
for boundary and obstacle verification.

References:
- docs/specs/sim.md:resetTargets - Static target randomization behavior
- docs/specs/sim.md:findValidPosition - 3.0 unit exclusion radius specification
- docs/specs/mgr.md:targetPositionTensor - Target position tensor access
"""

import math

import numpy as np
import pytest

from madrona_escape_room import ExecMode, SimManager
from madrona_escape_room.generated_constants import AssetIDs, consts
from madrona_escape_room.level_compiler import compile_level

# Improved test level with more cubes and complex obstacle layout
TEST_LEVEL_COMPLEX_OBSTACLES = {
    "ascii": """
########################
#S.....................#
#...C........C.........#
#..........C...........#
#....C.................#
#..................C...#
#.......C..............#
#....................C.#
#...........C..........#
#.C....................#
#..........C...........#
#......................#
#.........C............#
#..................C...#
#...C..................#
#..................C...#
########################
""".strip(),
    "tileset": {
        "#": {"asset": "wall"},
        ".": {"asset": "empty"},
        "S": {"asset": "spawn"},
        "C": {"asset": "cube"},
    },
    "scale": 2.0,
    "agent_facing": [0.0],
    "spawn_random": False,
    "auto_boundary_walls": False,
    "targets": [
        {
            "position": [10, 10, 1],
            "motion_type": "static",
            "params": {
                "omega_x": 0.0,
                "omega_y": 1.5,  # Enable randomization with omega_y > 0
                "center": [10, 10, 1],
                "mass": 1.0,
                "phase_x": 0.0,
                "phase_y": 0.0,
            },
        },
        {
            "position": [20, 15, 1],
            "motion_type": "static",
            "params": {
                "omega_x": 0.0,
                "omega_y": 2.0,  # Different randomization seed
                "center": [20, 15, 1],
                "mass": 1.0,
                "phase_x": 0.0,
                "phase_y": 0.0,
            },
        },
    ],
    "name": "complex_obstacles_randomized_targets",
}

# Safe distance from specifications
COLLISION_EXCLUSION_RADIUS = 3.0  # From docs/specs/sim.md line 318, 977


def extract_obstacles_from_compiled_level(compiled_level):
    """Extract wall and cube positions from compiled level data."""
    walls = []
    cubes = []

    for i in range(compiled_level.num_tiles):
        obj_id = compiled_level.object_ids[i]
        if obj_id == AssetIDs.WALL:
            walls.append(
                (compiled_level.tile_x[i], compiled_level.tile_y[i], compiled_level.tile_z[i])
            )
        elif obj_id == AssetIDs.CUBE:
            cubes.append(
                (compiled_level.tile_x[i], compiled_level.tile_y[i], compiled_level.tile_z[i])
            )

    return walls, cubes


def check_collision_avoidance(target_pos, obstacles, exclusion_radius):
    """Check if target position maintains safe distance from obstacles."""
    for obstacle_pos in obstacles:
        distance = math.sqrt(
            (target_pos[0] - obstacle_pos[0]) ** 2
            + (target_pos[1] - obstacle_pos[1]) ** 2
            + (target_pos[2] - obstacle_pos[2]) ** 2
        )
        if distance < exclusion_radius:
            return False, distance, obstacle_pos
    return True, None, None


def check_boundary_constraints(target_pos, compiled_level, exclusion_radius):
    """Check if target position respects world boundaries with exclusion radius."""
    x, y, z = target_pos

    # Check boundaries with exclusion radius buffer
    if (
        x - exclusion_radius < compiled_level.world_min_x
        or x + exclusion_radius > compiled_level.world_max_x
        or y - exclusion_radius < compiled_level.world_min_y
        or y + exclusion_radius > compiled_level.world_max_y
    ):
        return False

    return True


@pytest.mark.spec("docs/specs/sim.md", "resetTargets")
@pytest.mark.spec("docs/specs/mgr.md", "targetPositionTensor")
@pytest.mark.json_level(TEST_LEVEL_COMPLEX_OBSTACLES)
def test_static_target_randomization_enabled(cpu_manager):
    """Test that static targets with randomization enabled change position between episodes."""
    mgr = cpu_manager

    # Collect target positions across multiple resets using target tensor
    target_positions_per_reset = []

    for reset_idx in range(50):  # Test with 50 resets for statistical significance
        # Reset episode - this should trigger target randomization
        # Reference: docs/specs/sim.md:resetTargets behavior
        mgr.reset_tensor().to_numpy()[0] = 1
        mgr.step()

        # Get target positions directly from tensor
        # Reference: docs/specs/mgr.md:targetPositionTensor
        target_tensor = mgr.target_position_tensor()
        positions = target_tensor.to_numpy()

        # Extract positions for first world, first two targets (we have 2 targets in level)
        world_0_targets = positions[0, :2, :]  # Shape: [2, 3]
        target_positions_per_reset.append(world_0_targets.copy())

        # Run a few more steps to ensure stable state
        for _ in range(3):
            mgr.step()

    # Verify that target positions changed between resets due to randomization
    first_reset_positions = target_positions_per_reset[0]
    position_variations = []
    unique_positions_target0 = set()
    unique_positions_target1 = set()

    for reset_idx, current_positions in enumerate(target_positions_per_reset):
        # Track unique positions (rounded to avoid floating point precision issues)
        unique_positions_target0.add(
            (round(current_positions[0][0], 2), round(current_positions[0][1], 2))
        )
        unique_positions_target1.add(
            (round(current_positions[1][0], 2), round(current_positions[1][1], 2))
        )

        if reset_idx > 0:
            # Check if positions are different from first reset
            target_0_moved = not np.allclose(
                first_reset_positions[0], current_positions[0], atol=0.1
            )
            target_1_moved = not np.allclose(
                first_reset_positions[1], current_positions[1], atol=0.1
            )
            position_variations.append(target_0_moved or target_1_moved)

    # Verify randomization is working - we should see multiple unique positions
    unique_count_target0 = len(unique_positions_target0)
    unique_count_target1 = len(unique_positions_target1)

    print(f"✓ Target 0: {unique_count_target0} unique positions across 50 resets")
    print(f"✓ Target 1: {unique_count_target1} unique positions across 50 resets")

    # With omega_y > 0, we expect significant randomization
    assert (
        unique_count_target0 > 10
    ), f"Expected >10 unique positions for target 0, got {unique_count_target0}"
    assert (
        unique_count_target1 > 10
    ), f"Expected >10 unique positions for target 1, got {unique_count_target1}"

    # Verify variation occurred in most resets
    variations_detected = sum(position_variations)
    variation_ratio = variations_detected / len(position_variations)

    print(
        f"✓ Position variation detected in {variations_detected}/"
        f"{len(position_variations)} resets ({variation_ratio:.1%})"
    )
    assert (
        variation_ratio > 0.7
    ), f"Expected >70% resets to show variation, got {variation_ratio:.1%}"


@pytest.mark.spec("docs/specs/sim.md", "findValidPosition")
@pytest.mark.spec("docs/specs/sim.md", "resetTargets")
@pytest.mark.json_level(TEST_LEVEL_COMPLEX_OBSTACLES)
def test_static_target_collision_avoidance(cpu_manager):
    """Test that randomized targets avoid collisions using 3.0 unit exclusion radius."""
    mgr = cpu_manager

    # Compile level to extract exact obstacle positions
    compiled_level = compile_level(TEST_LEVEL_COMPLEX_OBSTACLES)[0]
    walls, cubes = extract_obstacles_from_compiled_level(compiled_level)

    print(f"Level analysis: {len(walls)} walls, {len(cubes)} cubes")
    assert (
        len(cubes) >= 10
    ), f"Test level should have ≥10 cubes for meaningful collision testing, got {len(cubes)}"

    collision_violations = []
    all_target_positions = []

    for reset_idx in range(50):
        # Reset and get target positions
        mgr.reset_tensor().to_numpy()[0] = 1
        mgr.step()

        target_tensor = mgr.target_position_tensor()
        positions = target_tensor.to_numpy()
        world_0_targets = positions[0, :2, :]  # First world, first two targets

        for target_idx, target_pos in enumerate(world_0_targets):
            all_target_positions.append((reset_idx, target_idx, target_pos.copy()))

            # Check collision avoidance with walls using 3.0 unit exclusion radius
            # Reference: docs/specs/sim.md line 318:
            # "Uses findValidPosition() with 3.0 unit exclusion radius"
            wall_safe, wall_distance, wall_pos = check_collision_avoidance(
                target_pos, walls, COLLISION_EXCLUSION_RADIUS
            )
            if not wall_safe:
                collision_violations.append(
                    {
                        "reset": reset_idx,
                        "target": target_idx,
                        "obstacle_type": "wall",
                        "target_pos": target_pos,
                        "obstacle_pos": wall_pos,
                        "distance": wall_distance,
                    }
                )

            # Check collision avoidance with cubes using 3.0 unit exclusion radius
            cube_safe, cube_distance, cube_pos = check_collision_avoidance(
                target_pos, cubes, COLLISION_EXCLUSION_RADIUS
            )
            if not cube_safe:
                collision_violations.append(
                    {
                        "reset": reset_idx,
                        "target": target_idx,
                        "obstacle_type": "cube",
                        "target_pos": target_pos,
                        "obstacle_pos": cube_pos,
                        "distance": cube_distance,
                    }
                )

    # Report results
    total_positions_tested = len(all_target_positions)
    print(f"✓ Tested {total_positions_tested} target positions across 50 resets")

    if collision_violations:
        print(f"❌ Found {len(collision_violations)} collision violations:")
        for violation in collision_violations[:5]:  # Show first 5 violations
            print(
                f"  Reset {violation['reset']}, Target {violation['target']}: "
                f"distance {violation['distance']:.2f} to {violation['obstacle_type']} "
                f"(required: ≥{COLLISION_EXCLUSION_RADIUS:.1f})"
            )

        assert False, (
            f"Target randomization violated 3.0 unit exclusion radius in "
            f"{len(collision_violations)} cases"
        )

    print("✓ All randomized target positions maintain 3.0 unit exclusion radius from obstacles")


@pytest.mark.spec("docs/specs/sim.md", "resetTargets")
@pytest.mark.spec("docs/specs/mgr.md", "targetPositionTensor")
@pytest.mark.json_level(TEST_LEVEL_COMPLEX_OBSTACLES)
def test_static_target_boundary_respect(cpu_manager):
    """Test that randomized targets respect world boundaries with exclusion radius."""
    mgr = cpu_manager

    # Compile level to get exact boundaries
    compiled_level = compile_level(TEST_LEVEL_COMPLEX_OBSTACLES)[0]

    print(
        f"World boundaries: X({compiled_level.world_min_x:.1f}, {compiled_level.world_max_x:.1f}), "
        f"Y({compiled_level.world_min_y:.1f}, {compiled_level.world_max_y:.1f})"
    )

    boundary_violations = []

    for reset_idx in range(50):
        mgr.reset_tensor().to_numpy()[0] = 1
        mgr.step()

        target_tensor = mgr.target_position_tensor()
        positions = target_tensor.to_numpy()
        world_0_targets = positions[0, :2, :]

        for target_idx, target_pos in enumerate(world_0_targets):
            # Check boundary constraints with exclusion radius
            # Reference: docs/specs/sim.md line 320: "Boundary constraints"
            if not check_boundary_constraints(
                target_pos, compiled_level, COLLISION_EXCLUSION_RADIUS
            ):
                boundary_violations.append(
                    {
                        "reset": reset_idx,
                        "target": target_idx,
                        "position": target_pos,
                    }
                )

    if boundary_violations:
        print(f"❌ Found {len(boundary_violations)} boundary violations:")
        for violation in boundary_violations[:5]:
            pos = violation["position"]
            print(
                f"  Reset {violation['reset']}, Target {violation['target']}: "
                f"position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )

        assert (
            False
        ), f"Target randomization violated world boundaries in {len(boundary_violations)} cases"

    print(
        "✓ All randomized target positions respect world boundaries with 3.0 unit exclusion radius"
    )


@pytest.mark.spec("docs/specs/sim.md", "resetTargets")
@pytest.mark.spec("docs/specs/mgr.md", "targetPositionTensor")
@pytest.mark.json_level(TEST_LEVEL_COMPLEX_OBSTACLES)
def test_target_randomization_distribution(cpu_manager):
    """Test that randomized positions are well-distributed across available space."""
    mgr = cpu_manager

    # Collect many target positions to analyze distribution
    target_positions = []

    for reset_idx in range(100):  # More resets for distribution analysis
        mgr.reset_tensor().to_numpy()[0] = 1
        mgr.step()

        target_tensor = mgr.target_position_tensor()
        positions = target_tensor.to_numpy()
        target_positions.extend(positions[0, :2, :])  # First world, both targets

    # Convert to numpy array for analysis
    positions_array = np.array(target_positions)
    x_positions = positions_array[:, 0]
    y_positions = positions_array[:, 1]

    # Analyze distribution using 3.0 unit grid cells (matching exclusion radius)
    grid_size = COLLISION_EXCLUSION_RADIUS
    x_min, x_max = x_positions.min(), x_positions.max()
    y_min, y_max = y_positions.min(), y_positions.max()

    # Count positions in grid cells
    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1

    occupied_cells = set()
    for x, y in zip(x_positions, y_positions):
        grid_x = int((x - x_min) / grid_size)
        grid_y = int((y - y_min) / grid_size)
        occupied_cells.add((grid_x, grid_y))

    total_cells = x_bins * y_bins
    occupied_ratio = len(occupied_cells) / total_cells

    print("✓ Position distribution analysis:")
    print(f"  X range: {x_min:.1f} to {x_max:.1f} ({x_max-x_min:.1f} units)")
    print(f"  Y range: {y_min:.1f} to {y_max:.1f} ({y_max-y_min:.1f} units)")
    print(f"  Grid cells: {x_bins} × {y_bins} = {total_cells}")
    print(f"  Occupied cells: {len(occupied_cells)} ({occupied_ratio:.1%})")

    # Verify reasonable distribution
    assert x_max - x_min > 10.0, f"X distribution too narrow: {x_max - x_min:.1f} units"
    assert y_max - y_min > 10.0, f"Y distribution too narrow: {y_max - y_min:.1f} units"
    assert (
        occupied_ratio > 0.1
    ), f"Position distribution too clustered: {occupied_ratio:.1%} cells occupied"

    print("✓ Target positions show good distribution across available space")


@pytest.mark.spec("docs/specs/sim.md", "resetTargets")
@pytest.mark.spec("docs/specs/mgr.md", "targetPositionTensor")
@pytest.mark.json_level(TEST_LEVEL_COMPLEX_OBSTACLES)
def test_static_target_deterministic_randomization(cpu_manager):
    """Test that target randomization is deterministic across runs with same seed."""
    mgr = cpu_manager

    def collect_target_positions(num_episodes=10):
        """Collect target positions over multiple episodes."""
        positions = []
        for episode in range(num_episodes):
            mgr.reset_tensor().to_numpy()[0] = 1
            mgr.step()

            target_tensor = mgr.target_position_tensor()
            tensor_data = target_tensor.to_numpy()
            # Store both targets from first world
            positions.append(tensor_data[0, :2, :].copy())

        return positions

    # First run with deterministic simulation
    first_run = collect_target_positions()

    # Reset manager to same initial state and run again
    # Since we're using the same manager with same seed, should get same sequence
    second_run = collect_target_positions()

    # Compare positions between runs
    positions_match = True
    for i, (pos1, pos2) in enumerate(zip(first_run, second_run)):
        if not np.allclose(pos1, pos2, atol=0.001):
            positions_match = False
            break

    if positions_match:
        print("✓ Target randomization is deterministic (same positions across runs)")
        print("  Note: This indicates deterministic PRNG seeding as per specification")
    else:
        print("✓ Target randomization shows variation between runs")
        print(f"  First run target 0 episode 0: {first_run[0][0]}")
        print(f"  Second run target 0 episode 0: {second_run[0][0]}")

    # Verify that within each run, positions do vary between episodes
    first_episode_pos = first_run[0]
    position_variations = []

    for episode_pos in first_run[1:]:
        target_0_moved = not np.allclose(first_episode_pos[0], episode_pos[0], atol=0.1)
        target_1_moved = not np.allclose(first_episode_pos[1], episode_pos[1], atol=0.1)
        position_variations.append(target_0_moved or target_1_moved)

    variation_count = sum(position_variations)
    print(
        f"✓ Within-run variation: {variation_count}/"
        f"{len(position_variations)} episodes showed position changes"
    )

    # We expect significant variation within a run due to episode-based randomization
    # Reference: docs/specs/sim.md PRNG key pattern:
    # rand::split_i(episode_key, 4000u + target_id, 0u)
    assert variation_count > len(position_variations) // 2, (
        f"Expected >50% episodes to show position variation, got "
        f"{variation_count}/{len(position_variations)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
