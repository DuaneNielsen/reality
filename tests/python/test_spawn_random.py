import pytest
import torch

import madrona_escape_room


def test_spawn_random_fixed(cpu_manager):
    mgr = cpu_manager
    positions = []

    for reset_num in range(3):
        reset_tensor = mgr.reset_tensor().to_torch()
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        obs = mgr.self_observation_tensor().to_torch()
        agent_pos = obs[0, 0, :3].clone()
        positions.append(agent_pos)

    # All positions should be identical (fixed spawn)
    for i, pos in enumerate(positions[1:], 1):
        assert torch.allclose(positions[0], pos, atol=0.01)


def test_default_level_spawn_random_flag():
    level = madrona_escape_room.create_default_level()
    assert not level.spawn_random


def test_spawn_random_enabled_custom_level():
    level = madrona_escape_room.create_default_level()
    level.spawn_random = True

    mgr = madrona_escape_room.SimManager(
        exec_mode=madrona_escape_room.ExecMode.CPU,
        num_worlds=1,
        compiled_levels=[level],
        gpu_id=0,
        rand_seed=42,
        auto_reset=True,
    )

    positions = []
    for reset_num in range(3):
        reset_tensor = mgr.reset_tensor().to_torch()
        reset_tensor[:] = 1
        mgr.step()
        reset_tensor[:] = 0

        obs = mgr.self_observation_tensor().to_torch()
        agent_pos = obs[0, 0, :3].clone()
        positions.append(agent_pos)

    # Check that not all positions are identical (proves randomness)
    all_same = all(torch.allclose(positions[0], pos, atol=0.01) for pos in positions[1:])
    assert not all_same
