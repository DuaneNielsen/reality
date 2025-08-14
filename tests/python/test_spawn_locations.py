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

# Test levels with specific spawn configurations
SINGLE_SPAWN_CENTER = """
##########
#........#
#........#
#...S....#
#........#
#........#
##########
"""

SINGLE_SPAWN_CORNER = """
##########
#S.......#
#........#
#........#
#........#
#........#
##########
"""

MULTIPLE_SPAWNS = """
##########
#S.......#
#........#
#........#
#........#
#.......S#
##########
"""

SPAWN_NEAR_WALL = """
##########
#.S......#
#........#
#........#
#........#
#........#
##########
"""


@pytest.mark.custom_level(SINGLE_SPAWN_CENTER)
def test_single_spawn_center(cpu_manager):
    """Test that agent spawns at center position marked with S"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    # Reset to apply spawn position
    reset_world(mgr, 0)

    # Get agent position
    pos = observer.get_position(0, agent_idx=0)

    # Expected position: S is at grid (4, 3) in a 10x7 grid
    # With fixed coordinate transformation:
    # Grid (4, 3) -> World ((4 - 5 + 0.5) * 2, -(3 - 3.5 + 0.5) * 2) = (-1, 0)
    expected_x = (4 - 10 / 2.0 + 0.5) * 2.0  # -1
    expected_y = -(3 - 7 / 2.0 + 0.5) * 2.0  # 0

    print(f"Agent spawned at: X={pos[0]:.2f}, Y={pos[1]:.2f}")
    print(f"Expected spawn: X={expected_x:.2f}, Y={expected_y:.2f}")

    # NOTE: This currently fails because C++ uses hardcoded positions
    assert abs(pos[0] - expected_x) < 0.1, f"X position {pos[0]} should be near {expected_x}"
    assert abs(pos[1] - expected_y) < 0.1, f"Y position {pos[1]} should be near {expected_y}"


@pytest.mark.custom_level(SINGLE_SPAWN_CORNER)
def test_single_spawn_corner(cpu_manager):
    """Test that agent spawns at corner position marked with S"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    reset_world(mgr, 0)

    pos = observer.get_position(0, agent_idx=0)

    # S is at grid (1, 1) in a 10x7 grid
    # With fixed coordinate transformation:
    # Grid (1, 1) -> World ((1 - 5 + 0.5) * 2, -(1 - 3.5 + 0.5) * 2) = (-7, 4)
    expected_x = (1 - 10 / 2.0 + 0.5) * 2.0  # -7
    expected_y = -(1 - 7 / 2.0 + 0.5) * 2.0  # 4

    print(f"Agent spawned at: X={pos[0]:.2f}, Y={pos[1]:.2f}")
    print(f"Expected spawn: X={expected_x:.2f}, Y={expected_y:.2f}")

    assert abs(pos[0] - expected_x) < 0.1, f"X position {pos[0]} should be near {expected_x}"
    assert abs(pos[1] - expected_y) < 0.1, f"Y position {pos[1]} should be near {expected_y}"


@pytest.mark.custom_level(MULTIPLE_SPAWNS)
def test_multiple_spawn_locations(cpu_manager):
    """Test that the first agent uses the first spawn point when multiple S markers exist"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    reset_world(mgr, 0)

    # Get position for the single agent (only 1 agent per world currently)
    pos0 = observer.get_position(0, agent_idx=0)

    # First S at (1, 1), second S at (8, 5)
    # With the fixed coordinate transformation:
    # Grid (1, 1) -> World ((1 - 5 + 0.5) * 2, -(1 - 3.5 + 0.5) * 2) = (-7, 4)
    expected_x0 = (1 - 10 / 2.0 + 0.5) * 2.0  # -7
    expected_y0 = -(1 - 7 / 2.0 + 0.5) * 2.0  # 4

    print(f"Agent 0 spawned at: X={pos0[0]:.2f}, Y={pos0[1]:.2f}")
    print(f"Expected spawn 0: X={expected_x0:.2f}, Y={expected_y0:.2f}")

    # First agent should use first spawn point
    assert (
        abs(pos0[0] - expected_x0) < 0.1
    ), f"Agent 0 X position {pos0[0]} should be near {expected_x0}"
    assert (
        abs(pos0[1] - expected_y0) < 0.1
    ), f"Agent 0 Y position {pos0[1]} should be near {expected_y0}"

    # NOTE: When 2 agents per world are supported, the second agent would use
    # the second spawn at (8, 5)


@pytest.mark.custom_level(SPAWN_NEAR_WALL)
def test_spawn_near_wall(cpu_manager):
    """Test that agent can spawn close to walls"""
    mgr = cpu_manager
    observer = ObservationReader(mgr)

    reset_world(mgr, 0)

    pos = observer.get_position(0, agent_idx=0)

    # S is at grid (2, 1) - one tile from left wall
    # With fixed coordinate transformation:
    # Grid (2, 1) -> World ((2 - 5 + 0.5) * 2, -(1 - 3.5 + 0.5) * 2) = (-5, 4)
    expected_x = (2 - 10 / 2.0 + 0.5) * 2.0  # -5
    expected_y = -(1 - 7 / 2.0 + 0.5) * 2.0  # 4

    print(f"Agent spawned at: X={pos[0]:.2f}, Y={pos[1]:.2f}")
    print(f"Expected spawn near wall: X={expected_x:.2f}, Y={expected_y:.2f}")

    # Verify spawn position
    assert abs(pos[0] - expected_x) < 0.1, f"X position {pos[0]} should be near {expected_x}"
    assert abs(pos[1] - expected_y) < 0.1, f"Y position {pos[1]} should be near {expected_y}"

    # Verify it's close to wall (wall at x=-9, agent at x=-5, so 4 units away)
    wall_x = (0 - 10 / 2.0 + 0.5) * 2.0  # Left wall position at x=-9
    distance_to_wall = abs(pos[0] - wall_x)
    assert distance_to_wall < 3.0, "Agent should be close to wall"


def test_spawn_coordinate_transformation():
    """Test the coordinate transformation from grid to world space"""
    from madrona_escape_room.level_compiler import compile_level

    # Test level with known spawn position
    level = """
    #####
    #...#
    #.S.#
    #...#
    #####
    """

    compiled = compile_level(level, scale=2.0)

    # S is at grid position (2, 2) in a 5x5 grid
    # With fixed coordinate transformation:
    # Grid (2, 2) -> World ((2 - 2.5 + 0.5) * 2, -(2 - 2.5 + 0.5) * 2) = (0, 0)
    expected_x = (2 - 5 / 2.0 + 0.5) * 2.0  # 0
    expected_y = -(2 - 5 / 2.0 + 0.5) * 2.0  # 0

    # Check spawn points were parsed
    assert "_spawn_points" in compiled
    assert len(compiled["_spawn_points"]) == 1

    spawn_x, spawn_y = compiled["_spawn_points"][0]

    print(f"Spawn point in world coords: ({spawn_x:.2f}, {spawn_y:.2f})")
    print(f"Expected: ({expected_x:.2f}, {expected_y:.2f})")

    assert abs(spawn_x - expected_x) < 0.01, f"Spawn X {spawn_x} should be {expected_x}"
    assert abs(spawn_y - expected_y) < 0.01, f"Spawn Y {spawn_y} should be {expected_y}"


def test_no_spawn_marker():
    """Test that level without spawn marker raises an error"""
    from madrona_escape_room.level_compiler import compile_level

    # Level with no S marker
    level_no_spawn = """
    #####
    #...#
    #...#
    #...#
    #####
    """

    with pytest.raises(ValueError, match="No spawn points"):
        compile_level(level_no_spawn)


def test_multiple_spawn_parsing():
    """Test that multiple spawn points are correctly parsed"""
    from madrona_escape_room.level_compiler import compile_level

    level = """
    #######
    #S....#
    #.....#
    #....S#
    #######
    """

    compiled = compile_level(level, scale=1.0)  # Use scale=1 for simpler math

    assert "_spawn_points" in compiled
    assert len(compiled["_spawn_points"]) == 2

    spawns = compiled["_spawn_points"]

    # First S at (1, 1), second S at (5, 3) in a 7x5 grid
    # With fixed coordinate transformation:
    # Grid (1, 1) -> World ((1 - 3.5 + 0.5) * 1, -(1 - 2.5 + 0.5) * 1) = (-2, 1)
    # Grid (5, 3) -> World ((5 - 3.5 + 0.5) * 1, -(3 - 2.5 + 0.5) * 1) = (2, -1)
    expected_spawns = [
        ((1 - 7 / 2.0 + 0.5) * 1.0, -(1 - 5 / 2.0 + 0.5) * 1.0),  # (-2, 1)
        ((5 - 7 / 2.0 + 0.5) * 1.0, -(3 - 5 / 2.0 + 0.5) * 1.0),  # (2, -1)
    ]

    # Check both spawn points exist (order may vary)
    for expected in expected_spawns:
        found = False
        for spawn in spawns:
            if abs(spawn[0] - expected[0]) < 0.01 and abs(spawn[1] - expected[1]) < 0.01:
                found = True
                break
        assert found, f"Expected spawn point {expected} not found in {spawns}"


@pytest.mark.skip(reason="Agent rotation at spawn not yet implemented")
def test_spawn_rotation():
    """Test that agents spawn with correct rotation (future feature)"""
    # This would test spawn rotation if we add directional spawn markers
    # like 'N', 'S', 'E', 'W' for spawn with direction
    pass
