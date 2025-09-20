#!/usr/bin/env python3
"""
Test script for target configuration in level compiler.
Verifies that targets can be specified in JSON and compiled correctly.
"""

import numpy as np
import pytest

from madrona_escape_room.level_compiler import compile_level


@pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
def test_static_target_compilation():
    """Test compilation of a single static target."""
    level_json = {
        "ascii": ["####", "#S.#", "####"],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "targets": [{"position": [5.0, 10.0, 1.0], "motion_type": "static"}],
    }

    compiled_levels = compile_level(level_json)
    assert len(compiled_levels) == 1

    level = compiled_levels[0]
    assert level.num_targets == 1
    assert level.target_x[0] == 5.0
    assert level.target_y[0] == 10.0
    assert level.target_z[0] == 1.0
    assert level.target_motion_type[0] == 0  # Static

    # Static targets should have no special parameters (first 8 slots)
    for i in range(8):
        assert level.target_params[i] == 0.0


@pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
def test_figure8_target_compilation():
    """Test compilation of a figure-8 oscillator target."""
    level_json = {
        "ascii": ["####", "#S.#", "####"],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "targets": [
            {
                "position": [10.0, 5.0, 1.0],
                "motion_type": "figure8",
                "params": {
                    "omega_x": 1.5,
                    "omega_y": 2.0,
                    "center": [12.0, 7.0, 1.5],
                    "mass": 0.8,
                    "phase_x": 10.0,
                    "phase_y": 5.0,
                },
            }
        ],
    }

    compiled_levels = compile_level(level_json)
    assert len(compiled_levels) == 1

    level = compiled_levels[0]
    assert level.num_targets == 1
    assert level.target_x[0] == 10.0
    assert level.target_y[0] == 5.0
    assert level.target_z[0] == 1.0
    assert level.target_motion_type[0] == 1  # Figure-8

    # Check figure-8 parameters (flattened array)
    assert level.target_params[0] == 1.5  # omega_x
    assert level.target_params[1] == 2.0  # omega_y
    assert level.target_params[2] == 12.0  # center_x
    assert level.target_params[3] == 7.0  # center_y
    assert level.target_params[4] == 1.5  # center_z
    assert level.target_params[5] == 0.8  # mass
    assert level.target_params[6] == 10.0  # phase_x
    assert level.target_params[7] == 5.0  # phase_y


@pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
def test_multiple_targets_compilation():
    """Test compilation of multiple targets with different motion types."""
    level_json = {
        "ascii": ["####", "#S.#", "####"],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "targets": [
            {"position": [0.0, 0.0, 1.0], "motion_type": "static"},
            {
                "position": [5.0, 5.0, 2.0],
                "motion_type": "figure8",
                "params": {
                    "omega_x": 3.0,
                    "omega_y": 1.0,
                    "center": [5.0, 5.0, 2.0],
                    "mass": 2.0,
                    "phase_x": 8.0,
                    "phase_y": 12.0,
                },
            },
            {"position": [-3.0, 7.0, 0.5], "motion_type": "static"},
        ],
    }

    compiled_levels = compile_level(level_json)
    assert len(compiled_levels) == 1

    level = compiled_levels[0]
    assert level.num_targets == 3

    # Check first target (static)
    assert level.target_x[0] == 0.0
    assert level.target_y[0] == 0.0
    assert level.target_z[0] == 1.0
    assert level.target_motion_type[0] == 0

    # Check second target (figure-8)
    assert level.target_x[1] == 5.0
    assert level.target_y[1] == 5.0
    assert level.target_z[1] == 2.0
    assert level.target_motion_type[1] == 1
    assert level.target_params[8 + 0] == 3.0  # omega_x
    assert level.target_params[8 + 1] == 1.0  # omega_y
    assert level.target_params[8 + 5] == 2.0  # mass

    # Check third target (static)
    assert level.target_x[2] == -3.0
    assert level.target_y[2] == 7.0
    assert level.target_z[2] == 0.5
    assert level.target_motion_type[2] == 0


@pytest.mark.spec("docs/specs/level_compiler.md", "JSON Multi-Level Format")
def test_targets_in_multi_level_format():
    """Test targets in multi-level format with both shared and per-level targets."""
    level_json = {
        "levels": [
            {
                "ascii": ["###", "#S#", "###"],
                "name": "level_1",
                "targets": [{"position": [1.0, 1.0, 1.0], "motion_type": "static"}],
            },
            {"ascii": ["#####", "#S..#", "#####"], "name": "level_2"},
        ],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "targets": [
            {
                "position": [10.0, 10.0, 1.0],
                "motion_type": "figure8",
                "params": {
                    "omega_x": 1.0,
                    "omega_y": 1.0,
                    "center": [10.0, 10.0, 1.0],
                    "mass": 1.0,
                    "phase_x": 5.0,
                    "phase_y": 7.0,
                },
            }
        ],
    }

    compiled_levels = compile_level(level_json)
    assert len(compiled_levels) == 2

    # First level should have per-level target
    level1 = compiled_levels[0]
    assert level1.num_targets == 1
    assert level1.target_motion_type[0] == 0  # Static

    # Second level should have shared target
    level2 = compiled_levels[1]
    assert level2.num_targets == 1
    assert level2.target_motion_type[0] == 1  # Figure-8


@pytest.mark.spec("docs/specs/level_compiler.md", "validate_compiled_level")
def test_targets_validation_errors():
    """Test various validation errors for targets configuration."""

    # Test invalid targets field type
    with pytest.raises(ValueError, match="'targets' field must be a list"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": "invalid",
            }
        )

    # Test too many targets
    too_many_targets = [
        {"position": [i, i, 1.0], "motion_type": "static"}
        for i in range(10)  # More than MAX_TARGETS (8)
    ]
    with pytest.raises(ValueError, match="Too many targets"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": too_many_targets,
            }
        )

    # Test missing position field
    with pytest.raises(ValueError, match="Target 0 must have 'position' field"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": [{"motion_type": "static"}],
            }
        )

    # Test invalid position format
    with pytest.raises(ValueError, match="Target 0 'position' must be \\[x, y, z\\] array"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": [{"position": [1.0, 2.0], "motion_type": "static"}],
            }
        )

    # Test missing motion_type field
    with pytest.raises(ValueError, match="Target 0 must have 'motion_type' field"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": [{"position": [1.0, 2.0, 3.0]}],
            }
        )

    # Test invalid motion_type
    with pytest.raises(ValueError, match="Target 0 invalid motion_type 'invalid'"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": [{"position": [1.0, 2.0, 3.0], "motion_type": "invalid"}],
            }
        )

    # Test figure8 without params
    with pytest.raises(
        ValueError, match="Target 0 with motion_type 'figure8' must have 'params' field"
    ):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": [{"position": [1.0, 2.0, 3.0], "motion_type": "figure8"}],
            }
        )

    # Test figure8 with missing params
    with pytest.raises(ValueError, match="Target 0 figure8 params missing 'omega_x' field"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "targets": [
                    {
                        "position": [1.0, 2.0, 3.0],
                        "motion_type": "figure8",
                        "params": {"omega_y": 1.0},
                    }
                ],
            }
        )


@pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
def test_empty_targets_list():
    """Test that empty targets list is handled correctly."""
    level_json = {
        "ascii": ["####", "#S.#", "####"],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "targets": [],
    }

    compiled_levels = compile_level(level_json)
    assert len(compiled_levels) == 1

    level = compiled_levels[0]
    assert level.num_targets == 0


@pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
def test_targets_field_optional():
    """Test that targets field is optional and defaults to empty."""
    level_json = {
        "ascii": ["####", "#S.#", "####"],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        # No targets field
    }

    compiled_levels = compile_level(level_json)
    assert len(compiled_levels) == 1

    level = compiled_levels[0]
    assert level.num_targets == 0
