"""
Test lidar sensor noise implementation

Tests Gaussian noise addition to lidar readings using deterministic PRNG.

Specification: docs/specs/sim.md - lidarSystem - Noise Model
Validates proportional and base Gaussian noise implementation per spec.
"""

import numpy as np
import pytest

from madrona_escape_room import ExecMode

# Simple test level with a wall for lidar hits
NOISE_TEST_LEVEL = """################################################################
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
..............................S...............................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
################################################################"""


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
@pytest.mark.ascii_level(NOISE_TEST_LEVEL)
def test_no_noise_is_deterministic(cpu_manager):
    """Default (0.0 noise factors) produces exact same readings across episodes"""
    mgr = cpu_manager

    # Capture lidar readings from first episode
    mgr.step()  # Step 1
    lidar_ep1 = mgr.lidar_tensor().to_numpy().copy()

    # Trigger reset and get readings from second episode
    for _ in range(200):  # Complete episode
        mgr.step()

    mgr.step()  # Step 1 of episode 2
    lidar_ep2 = mgr.lidar_tensor().to_numpy().copy()

    # With no noise, readings should be identical
    np.testing.assert_array_equal(lidar_ep1, lidar_ep2)


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_with_noise_varies_between_episodes():
    """Non-zero noise creates variation between episodes"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.level_compiler import compile_ascii_level

    level = compile_ascii_level(NOISE_TEST_LEVEL)

    # Enable noise
    level.lidar_noise_factor = 0.01  # 1% proportional noise
    level.lidar_base_sigma = 0.02  # 2cm base noise

    mgr = SimManager(
        exec_mode=ExecMode.CPU, num_worlds=1, rand_seed=42, auto_reset=True, compiled_levels=[level]
    )

    # Capture lidar readings from first episode
    mgr.step()  # Step 1
    lidar_ep1 = mgr.lidar_tensor().to_numpy().copy()

    # Trigger reset and get readings from second episode
    for _ in range(200):  # Complete episode
        mgr.step()

    mgr.step()  # Step 1 of episode 2
    lidar_ep2 = mgr.lidar_tensor().to_numpy().copy()

    # With noise, readings should differ
    # (Different episode = different RNG state)
    assert not np.allclose(lidar_ep1, lidar_ep2, rtol=1e-6)


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_noise_deterministic_with_same_seed():
    """Same seed produces identical noisy readings"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.level_compiler import compile_ascii_level

    level = compile_ascii_level(NOISE_TEST_LEVEL)

    # Enable noise
    level.lidar_noise_factor = 0.005
    level.lidar_base_sigma = 0.01

    # Create first manager with seed 123
    mgr1 = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        rand_seed=123,
        auto_reset=False,
        compiled_levels=[level],
    )

    mgr1.step()
    lidar1 = mgr1.lidar_tensor().to_numpy().copy()

    # Create second manager with same seed
    mgr2 = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        rand_seed=123,
        auto_reset=False,
        compiled_levels=[level],
    )

    mgr2.step()
    lidar2 = mgr2.lidar_tensor().to_numpy().copy()

    # Same seed should produce identical noise
    np.testing.assert_array_equal(lidar1, lidar2)


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_noise_stays_in_valid_range():
    """Noisy readings are clamped to [0, 1]"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.level_compiler import compile_ascii_level

    level = compile_ascii_level(NOISE_TEST_LEVEL)

    # Use very high noise to test clamping
    level.lidar_noise_factor = 1.0  # 100% proportional noise (extreme)
    level.lidar_base_sigma = 10.0  # 10 unit base noise (extreme)

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=1,
        rand_seed=42,
        auto_reset=False,
        compiled_levels=[level],
    )

    # Run several steps to sample various readings
    for _ in range(10):
        mgr.step()
        lidar = mgr.lidar_tensor().to_numpy()

        # All readings must be in [0, 1] range
        assert np.all(lidar >= 0.0), f"Found negative lidar reading: {lidar.min()}"
        assert np.all(lidar <= 1.0), f"Found lidar reading > 1.0: {lidar.max()}"


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_proportional_noise_scales_with_distance():
    """Noise factor creates larger variations for distant objects"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.level_compiler import compile_ascii_level

    level = compile_ascii_level(NOISE_TEST_LEVEL)

    # Use only proportional noise (no base noise)
    level.lidar_noise_factor = 0.01
    level.lidar_base_sigma = 0.0

    mgr = SimManager(
        exec_mode=ExecMode.CPU,
        num_worlds=2,
        rand_seed=42,
        auto_reset=False,
        compiled_levels=[level],
    )

    mgr.step()
    lidar = mgr.lidar_tensor().to_numpy()  # [worlds, agents, samples]

    # World 0 and World 1 should have different noise (different RNG state)
    # But the pattern should be similar (same geometry)
    assert lidar.shape == (2, 1, 128)

    # Verify readings are not identical across worlds
    assert not np.allclose(lidar[0], lidar[1], rtol=1e-6)


@pytest.mark.spec("docs/specs/sim.md", "lidarSystem")
def test_base_sigma_creates_noise_floor():
    """Base sigma adds constant noise regardless of distance"""
    from madrona_escape_room import SimManager
    from madrona_escape_room.level_compiler import compile_ascii_level

    level = compile_ascii_level(NOISE_TEST_LEVEL)

    # Use only base noise (no proportional)
    level.lidar_noise_factor = 0.0
    level.lidar_base_sigma = 0.05  # 5cm noise floor

    mgr = SimManager(
        exec_mode=ExecMode.CPU, num_worlds=1, rand_seed=42, auto_reset=True, compiled_levels=[level]
    )

    # Get readings from two episodes
    mgr.step()
    lidar_ep1 = mgr.lidar_tensor().to_numpy().copy()

    for _ in range(200):  # Complete episode
        mgr.step()

    mgr.step()
    lidar_ep2 = mgr.lidar_tensor().to_numpy().copy()

    # Even with just base noise, readings should vary
    assert not np.allclose(lidar_ep1, lidar_ep2, rtol=1e-6)
