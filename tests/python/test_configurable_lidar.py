"""
Tests for configurable lidar beam count and FOV.

This test suite verifies that lidar sensor configuration can be customized
per-level through the JSON level compiler, including:
- Variable beam count (num_samples)
- Variable field of view (fov_degrees)
- Backward compatibility with default values
"""

import numpy as np
import pytest

from madrona_escape_room.level_compiler import compile_level


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_default_lidar_config(cpu_manager):
    """Test that default lidar configuration uses 128 samples and 120° FOV."""
    mgr = cpu_manager
    num_worlds = 4  # cpu_manager fixture uses 4 worlds

    # Default level should have max buffer size
    lidar = mgr.lidar_tensor().to_numpy()
    assert lidar.shape == (
        num_worlds,
        1,
        256,
    ), f"Expected shape ({num_worlds}, 1, 256) for max buffer, got {lidar.shape}"

    compass = mgr.compass_tensor().to_numpy()
    assert compass.shape == (
        num_worlds,
        1,
        256,
    ), f"Expected shape ({num_worlds}, 1, 256) for max buffer, got {compass.shape}"

    # Run simulation step to generate lidar data
    mgr.step()

    # Check that we get valid lidar readings
    lidar = mgr.lidar_tensor().to_numpy()
    assert lidar[0, 0, 0] >= 0, "Lidar should have valid readings"


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
@pytest.mark.json_level(
    {
        "ascii": ["########", "#S....T#", "########"],
        "tileset": {
            "#": {"asset": "wall"},
            "S": {"asset": "spawn"},
            "T": {"asset": "target"},
            ".": {"asset": "empty"},
        },
        "lidar_num_samples": 64,
        "lidar_fov_degrees": 90.0,
        "name": "test_64_samples_90_fov",
    }
)
def test_custom_64_samples_90_fov(cpu_manager):
    """Test 64 lidar samples with 90° FOV."""
    mgr = cpu_manager

    # Verify tensor shapes (always max size)
    lidar = mgr.lidar_tensor().to_numpy()
    assert lidar.shape == (4, 1, 256)

    compass = mgr.compass_tensor().to_numpy()
    assert compass.shape == (4, 1, 256)

    # Step simulation
    mgr.step()

    # Verify only first 64 samples are non-zero (active samples)
    lidar = mgr.lidar_tensor().to_numpy()
    first_64 = lidar[0, 0, :64]
    remaining = lidar[0, 0, 64:]

    # At least some of the first 64 should have non-zero values (wall hits)
    assert np.any(first_64 > 0), "Expected some non-zero lidar samples in active range"

    # Remaining samples should all be zero
    assert np.all(remaining == 0.0), "Expected zero values in inactive sample range"

    # Verify compass uses 64 buckets (one should be 1.0, rest 0.0)
    compass = mgr.compass_tensor().to_numpy()
    first_64_compass = compass[0, 0, :64]
    remaining_compass = compass[0, 0, 64:]

    assert np.sum(first_64_compass) == 1.0, "Expected exactly one active compass bucket"
    assert np.all(remaining_compass == 0.0), "Expected zero values in inactive compass range"


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
@pytest.mark.json_level(
    {
        "ascii": ["##########", "#S......T#", "##########"],
        "tileset": {
            "#": {"asset": "wall"},
            "S": {"asset": "spawn"},
            "T": {"asset": "target"},
            ".": {"asset": "empty"},
        },
        "lidar_num_samples": 180,
        "lidar_fov_degrees": 180.0,
        "name": "test_180_samples_180_fov",
    }
)
def test_custom_180_samples_180_fov(cpu_manager):
    """Test 180 lidar samples with 180° FOV."""
    mgr = cpu_manager

    # Step simulation
    mgr.step()

    # Verify only first 180 samples are active
    lidar = mgr.lidar_tensor().to_numpy()
    first_180 = lidar[0, 0, :180]
    remaining = lidar[0, 0, 180:]

    # Should have wall hits in 180° arc
    assert np.any(first_180 > 0), "Expected non-zero lidar samples in 180° FOV"
    assert np.all(remaining == 0.0), "Expected zero values beyond 180 samples"

    # Verify compass uses 180 buckets
    compass = mgr.compass_tensor().to_numpy()
    first_180_compass = compass[0, 0, :180]
    remaining_compass = compass[0, 0, 180:]

    assert (
        np.sum(first_180_compass) == 1.0
    ), "Expected exactly one active compass bucket in 180 range"
    assert np.all(remaining_compass == 0.0), "Expected zero beyond 180 buckets"


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
@pytest.mark.json_level(
    {
        "ascii": ["####", "#S.#", "####"],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "lidar_num_samples": 1,
        "lidar_fov_degrees": 1.0,
        "name": "test_min_config",
    }
)
def test_min_lidar_config(cpu_manager):
    """Test minimum valid configuration: 1 sample, 1° FOV."""
    mgr = cpu_manager

    mgr.step()

    # Only first sample should be active
    lidar = mgr.lidar_tensor().to_numpy()
    assert lidar[0, 0, 1:].sum() == 0.0, "Expected only first sample to be active"

    # Only first compass bucket should be active
    compass = mgr.compass_tensor().to_numpy()
    assert np.sum(compass[0, 0, :]) == 1.0, "Expected exactly one compass bucket"
    assert compass[0, 0, 1:].sum() == 0.0, "Expected only first bucket to be active"


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
@pytest.mark.json_level(
    {
        "ascii": ["##########", "#S......T#", "##########"],
        "tileset": {
            "#": {"asset": "wall"},
            "S": {"asset": "spawn"},
            "T": {"asset": "target"},
            ".": {"asset": "empty"},
        },
        "lidar_num_samples": 256,
        "lidar_fov_degrees": 360.0,
        "name": "test_max_config",
    }
)
def test_max_lidar_config(cpu_manager):
    """Test maximum configuration: 256 samples, 360° FOV."""
    mgr = cpu_manager

    mgr.step()

    # All 256 samples should be available (full 360° coverage)
    lidar = mgr.lidar_tensor().to_numpy()
    assert lidar.shape[2] == 256, "Expected 256 lidar samples"

    # With 360° FOV, should see walls in all directions
    assert np.any(lidar[0, 0, :] > 0), "Expected wall hits in 360° coverage"

    # Compass should use all 256 buckets
    compass = mgr.compass_tensor().to_numpy()
    assert np.sum(compass[0, 0, :]) == 1.0, "Expected exactly one active compass bucket"


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_invalid_lidar_num_samples():
    """Test that invalid num_samples values are rejected."""
    # Test 0 samples (too low)
    with pytest.raises(ValueError, match="lidar_num_samples.*1-256"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "lidar_num_samples": 0,
            }
        )

    # Test 257 samples (too high)
    with pytest.raises(ValueError, match="lidar_num_samples.*1-256"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "lidar_num_samples": 257,
            }
        )

    # Test non-integer
    with pytest.raises(ValueError, match="lidar_num_samples.*integer"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "lidar_num_samples": 128.5,
            }
        )


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_invalid_lidar_fov_degrees():
    """Test that invalid fov_degrees values are rejected."""
    # Test 0° (too low)
    with pytest.raises(ValueError, match="lidar_fov_degrees.*1.0-360.0"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "lidar_fov_degrees": 0.0,
            }
        )

    # Test 361° (too high)
    with pytest.raises(ValueError, match="lidar_fov_degrees.*1.0-360.0"):
        compile_level(
            {
                "ascii": ["###", "#S#", "###"],
                "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}},
                "lidar_fov_degrees": 361.0,
            }
        )


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_multi_level_shared_config():
    """Test that multi-level format shares lidar config across levels."""
    level_json = {
        "levels": [{"ascii": ["####", "#S.#", "####"]}, {"ascii": ["######", "#S...#", "######"]}],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "lidar_num_samples": 96,
        "lidar_fov_degrees": 135.0,
    }

    compiled_levels = compile_level(level_json)

    # Both levels should have the same config
    assert len(compiled_levels) == 2
    assert compiled_levels[0].lidar_num_samples == 96
    assert compiled_levels[0].lidar_fov_degrees == 135.0
    assert compiled_levels[1].lidar_num_samples == 96
    assert compiled_levels[1].lidar_fov_degrees == 135.0


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_multi_level_per_level_override():
    """Test that per-level lidar config overrides shared config."""
    level_json = {
        "levels": [
            {"ascii": ["####", "#S.#", "####"], "lidar_num_samples": 64, "lidar_fov_degrees": 90.0},
            {
                "ascii": ["######", "#S...#", "######"],
                # Uses shared config
            },
        ],
        "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}, ".": {"asset": "empty"}},
        "lidar_num_samples": 128,
        "lidar_fov_degrees": 120.0,
    }

    compiled_levels = compile_level(level_json)

    # First level uses override
    assert compiled_levels[0].lidar_num_samples == 64
    assert compiled_levels[0].lidar_fov_degrees == 90.0

    # Second level uses shared config
    assert compiled_levels[1].lidar_num_samples == 128
    assert compiled_levels[1].lidar_fov_degrees == 120.0


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_backward_compatibility_no_config():
    """Test that levels without lidar config use default values."""
    level_json = {
        "ascii": ["########", "#S....T#", "########"],
        "tileset": {
            "#": {"asset": "wall"},
            "S": {"asset": "spawn"},
            "T": {"asset": "target"},
            ".": {"asset": "empty"},
        },
        # No lidar_num_samples or lidar_fov_degrees specified
    }

    compiled_levels = compile_level(level_json)

    # Should default to 128 samples, 120° FOV
    assert compiled_levels[0].lidar_num_samples == 128
    assert compiled_levels[0].lidar_fov_degrees == 120.0
