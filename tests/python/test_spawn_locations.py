#!/usr/bin/env python3
"""
Test spawn location functionality for Madrona Escape Room.
Tests that agents are placed at spawn points marked with 'S' in ASCII levels.

NOTE: These tests currently FAIL because the C++ implementation has hardcoded
spawn positions and doesn't use the spawn points from compiled levels yet.
"""

import numpy as np
import pytest
from test_helpers import AgentController, ObservationReader, reset_world

# Test levels with specific spawn configurations using ASCII strings
SINGLE_SPAWN_CENTER = """##########
#........#
#........#
#...S....#
#........#
#........#
##########"""

SINGLE_SPAWN_CORNER = """##########
#S.......#
#........#
#........#
#........#
#........#
##########"""

MULTIPLE_SPAWNS = """##########
#S.......#
#........#
#........#
#........#
#.......S#
##########"""

SPAWN_NEAR_WALL = """##########
#.S......#
#........#
#........#
#........#
#........#
##########"""

# Test level for coordinate transformation with known spawn position
COORDINATE_TEST_LEVEL = """#####
#...#
#.S.#
#...#
#####"""

# Test level for multiple spawn parsing
MULTIPLE_SPAWN_TEST_LEVEL = """#######
#S....#
#.....#
#....S#
#######"""


@pytest.mark.custom_level(SINGLE_SPAWN_CENTER)
def test_single_spawn_center(cpu_manager):
    """Test that agent spawns at center position marked with S"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    # Compile the level to get the actual world boundaries
    from madrona_escape_room.level_compiler import compile_ascii_level

    compiled = compile_ascii_level(SINGLE_SPAWN_CENTER, scale=2.5)

    # Get world boundaries for proper denormalization
    world_min_x = compiled.world_min_x
    world_min_y = compiled.world_min_y
    world_width = compiled.world_max_x - compiled.world_min_x
    world_length = compiled.world_max_y - compiled.world_min_y

    # Reset to apply spawn position
    reset_world(mgr, 0)

    # Get normalized position and denormalize correctly
    norm_pos = observer.get_normalized_position(0, agent_idx=0)
    pos_x = norm_pos[0] * world_width + world_min_x
    pos_y = norm_pos[1] * world_length + world_min_y

    # Expected position: S is at grid (4, 3) in a 10x7 grid
    # With fixed coordinate transformation:
    # Grid (4, 3) -> World ((4 - 5 + 0.5) * 2.5, -(3 - 3.5 + 0.5) * 2.5) = (-1.25, 0)
    expected_x = (4 - 10 / 2.0 + 0.5) * 2.5  # -1.25
    expected_y = -(3 - 7 / 2.0 + 0.5) * 2.5  # 0

    print(f"Agent spawned at: X={pos_x:.2f}, Y={pos_y:.2f}")
    print(f"Expected spawn: X={expected_x:.2f}, Y={expected_y:.2f}")

    # Now this should pass with the correct denormalization
    assert abs(pos_x - expected_x) < 0.1, f"X position {pos_x} should be near {expected_x}"
    assert abs(pos_y - expected_y) < 0.1, f"Y position {pos_y} should be near {expected_y}"


@pytest.mark.custom_level(SINGLE_SPAWN_CORNER)
def test_single_spawn_corner(cpu_manager):
    """Test that agent spawns at corner position marked with S"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    # Compile the level to get the actual world boundaries
    from madrona_escape_room.level_compiler import compile_ascii_level

    compiled = compile_ascii_level(SINGLE_SPAWN_CORNER, scale=2.5)

    # Get world boundaries for proper denormalization
    world_min_x = compiled.world_min_x
    world_min_y = compiled.world_min_y
    world_width = compiled.world_max_x - compiled.world_min_x
    world_length = compiled.world_max_y - compiled.world_min_y

    reset_world(mgr, 0)

    # Get normalized position and denormalize correctly
    norm_pos = observer.get_normalized_position(0, agent_idx=0)
    pos_x = norm_pos[0] * world_width + world_min_x
    pos_y = norm_pos[1] * world_length + world_min_y

    # S is at grid (1, 1) in a 10x7 grid
    # With fixed coordinate transformation:
    # Grid (1, 1) -> World ((1 - 5 + 0.5) * 2.5, -(1 - 3.5 + 0.5) * 2.5) = (-8.75, 5)
    expected_x = (1 - 10 / 2.0 + 0.5) * 2.5  # -8.75
    expected_y = -(1 - 7 / 2.0 + 0.5) * 2.5  # 5

    print(f"Agent spawned at: X={pos_x:.2f}, Y={pos_y:.2f}")
    print(f"Expected spawn: X={expected_x:.2f}, Y={expected_y:.2f}")

    assert abs(pos_x - expected_x) < 0.1, f"X position {pos_x} should be near {expected_x}"
    assert abs(pos_y - expected_y) < 0.1, f"Y position {pos_y} should be near {expected_y}"


@pytest.mark.custom_level(MULTIPLE_SPAWNS)
def test_multiple_spawn_locations(cpu_manager):
    """Test that the first agent uses the first spawn point when multiple S markers exist"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    # Compile the level to get the actual world boundaries
    from madrona_escape_room.level_compiler import compile_ascii_level

    compiled = compile_ascii_level(MULTIPLE_SPAWNS, scale=2.5)

    # Get world boundaries for proper denormalization
    world_min_x = compiled.world_min_x
    world_min_y = compiled.world_min_y
    world_width = compiled.world_max_x - compiled.world_min_x
    world_length = compiled.world_max_y - compiled.world_min_y

    reset_world(mgr, 0)

    # Get normalized position and denormalize correctly
    norm_pos0 = observer.get_normalized_position(0, agent_idx=0)
    pos0_x = norm_pos0[0] * world_width + world_min_x
    pos0_y = norm_pos0[1] * world_length + world_min_y

    # First S at (1, 1), second S at (8, 5)
    # With the fixed coordinate transformation:
    # Grid (1, 1) -> World ((1 - 5 + 0.5) * 2.5, -(1 - 3.5 + 0.5) * 2.5) = (-8.75, 5)
    expected_x0 = (1 - 10 / 2.0 + 0.5) * 2.5  # -8.75
    expected_y0 = -(1 - 7 / 2.0 + 0.5) * 2.5  # 5

    print(f"Agent 0 spawned at: X={pos0_x:.2f}, Y={pos0_y:.2f}")
    print(f"Expected spawn 0: X={expected_x0:.2f}, Y={expected_y0:.2f}")

    # First agent should use first spawn point
    assert (
        abs(pos0_x - expected_x0) < 0.1
    ), f"Agent 0 X position {pos0_x} should be near {expected_x0}"
    assert (
        abs(pos0_y - expected_y0) < 0.1
    ), f"Agent 0 Y position {pos0_y} should be near {expected_y0}"

    # NOTE: When 2 agents per world are supported, the second agent would use
    # the second spawn at (8, 5)


@pytest.mark.custom_level(SPAWN_NEAR_WALL)
def test_spawn_near_wall(cpu_manager):
    """Test that agent can spawn close to walls"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    # Compile the level to get the actual world boundaries
    from madrona_escape_room.level_compiler import compile_ascii_level

    compiled = compile_ascii_level(SPAWN_NEAR_WALL, scale=2.5)

    # Get world boundaries for proper denormalization
    world_min_x = compiled.world_min_x
    world_min_y = compiled.world_min_y
    world_width = compiled.world_max_x - compiled.world_min_x
    world_length = compiled.world_max_y - compiled.world_min_y

    reset_world(mgr, 0)

    # Get normalized position and denormalize correctly
    norm_pos = observer.get_normalized_position(0, agent_idx=0)
    pos_x = norm_pos[0] * world_width + world_min_x
    pos_y = norm_pos[1] * world_length + world_min_y

    # S is at grid (2, 1) - next to wall at (1, 1)
    # With coordinate transformation:
    # Grid (2, 1) -> World ((2 - 5 + 0.5) * 2.5, -(1 - 3.5 + 0.5) * 2.5) = (-6.25, 5)
    expected_x = (2 - 10 / 2.0 + 0.5) * 2.5  # -6.25
    expected_y = -(1 - 7 / 2.0 + 0.5) * 2.5  # 5

    print(f"Agent spawned at: X={pos_x:.2f}, Y={pos_y:.2f}")
    print(f"Expected spawn near wall: X={expected_x:.2f}, Y={expected_y:.2f}")

    # Verify spawn position
    assert abs(pos_x - expected_x) < 0.1, f"X position {pos_x} should be near {expected_x}"
    assert abs(pos_y - expected_y) < 0.1, f"Y position {pos_y} should be near {expected_y}"

    # Verify it's close to wall - wall at grid (1,1) is at world (-8.75, 5)
    wall_x = (1 - 10 / 2.0 + 0.5) * 2.5  # Wall position at x=-8.75
    distance_to_wall = abs(pos_x - wall_x)
    print(f"Distance to wall: {distance_to_wall:.2f} units")
    assert distance_to_wall < 3.0, "Agent should be close to wall"


@pytest.mark.custom_level(COORDINATE_TEST_LEVEL)
def test_spawn_coordinate_transformation(cpu_manager):
    """Test the coordinate transformation from grid to world space"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    # Compile the level to get the actual world boundaries
    from madrona_escape_room.level_compiler import compile_ascii_level

    compiled = compile_ascii_level(COORDINATE_TEST_LEVEL, scale=2.5)

    # Get world boundaries for proper denormalization
    world_min_x = compiled.world_min_x
    world_min_y = compiled.world_min_y
    world_width = compiled.world_max_x - compiled.world_min_x
    world_length = compiled.world_max_y - compiled.world_min_y

    reset_world(mgr, 0)

    # Get normalized position and denormalize correctly
    norm_pos = observer.get_normalized_position(0, agent_idx=0)
    pos_x = norm_pos[0] * world_width + world_min_x
    pos_y = norm_pos[1] * world_length + world_min_y

    # S is at grid position (2, 2) in a 5x5 grid
    # With coordinate transformation:
    # Grid (2, 2) -> World ((2 - 2.5 + 0.5) * 2.5, -(2 - 2.5 + 0.5) * 2.5) = (0, 0)
    expected_x = (2 - 5 / 2.0 + 0.5) * 2.5  # 0
    expected_y = -(2 - 5 / 2.0 + 0.5) * 2.5  # 0

    print(f"Agent spawned at: X={pos_x:.2f}, Y={pos_y:.2f}")
    print(f"Expected spawn: X={expected_x:.2f}, Y={expected_y:.2f}")

    assert abs(pos_x - expected_x) < 0.01, f"Spawn X {pos_x} should be {expected_x}"
    assert abs(pos_y - expected_y) < 0.01, f"Spawn Y {pos_y} should be {expected_y}"


def test_no_spawn_marker():
    """Test that level without spawn marker raises an error"""
    from madrona_escape_room.level_compiler import compile_ascii_level

    # Level with no S marker
    level_no_spawn = """
    #####
    #...#
    #...#
    #...#
    #####
    """

    with pytest.raises(ValueError, match="No spawn points"):
        compile_ascii_level(level_no_spawn)


@pytest.mark.custom_level(MULTIPLE_SPAWN_TEST_LEVEL)
def test_multiple_spawn_parsing(cpu_manager):
    """Test that multiple spawn points are correctly parsed"""
    from madrona_escape_room.level_compiler import compile_level

    # Use the level_compiler directly to check spawn parsing
    # since the cpu_manager only shows where the first agent spawns
    level_json = {
        "ascii": """#######
#S....#
#.....#
#....S#
#######""",
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "scale": 1.0,  # Use scale=1 for simpler math
        "name": "test_multiple_spawns",
    }

    compiled = compile_level(level_json)

    # Check that we have multiple spawn points
    assert compiled.num_spawns == 2, f"Expected 2 spawn points, got {compiled.num_spawns}"

    # First S at (1, 1), second S at (5, 3) in a 7x5 grid
    # With coordinate transformation:
    # Grid (1, 1) -> World ((1 - 3.5 + 0.5) * 1, -(1 - 2.5 + 0.5) * 1) = (-2, 1)
    # Grid (5, 3) -> World ((5 - 3.5 + 0.5) * 1, -(3 - 2.5 + 0.5) * 1) = (2, -1)
    expected_spawns = [
        ((1 - 7 / 2.0 + 0.5) * 1.0, -(1 - 5 / 2.0 + 0.5) * 1.0),  # (-2, 1)
        ((5 - 7 / 2.0 + 0.5) * 1.0, -(3 - 5 / 2.0 + 0.5) * 1.0),  # (2, -1)
    ]

    # Check both spawn points exist (order may vary)
    for i, expected in enumerate(expected_spawns):
        found = False
        for j in range(compiled.num_spawns):
            actual_x = compiled.spawn_x[j]
            actual_y = compiled.spawn_y[j]
            if abs(actual_x - expected[0]) < 0.01 and abs(actual_y - expected[1]) < 0.01:
                found = True
                break
        assert found, f"Expected spawn point {expected} not found in compiled level spawn points"


@pytest.mark.skip(reason="Agent rotation at spawn not yet implemented")
def test_spawn_rotation():
    """Test that agents spawn with correct rotation (future feature)"""
    # This would test spawn rotation if we add directional spawn markers
    # like 'N', 'S', 'E', 'W' for spawn with direction
    pass
