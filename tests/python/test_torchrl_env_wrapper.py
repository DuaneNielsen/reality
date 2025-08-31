"""Unit tests for the TorchRL environment wrapper"""

import pytest
import torch

# Skip all tests in this module - TorchRL integration is a work in progress
pytest.skip("TorchRL integration is a work in progress", allow_module_level=True)

# Import from madrona_escape_room_learn package
from madrona_escape_room_learn import MadronaEscapeRoomEnv
from tensordict import TensorDict

import madrona_escape_room


@pytest.fixture
def cpu_env():
    """Fixture for CPU environment"""
    env = MadronaEscapeRoomEnv(num_worlds=4, gpu_id=-1, rand_seed=42, auto_reset=True)
    yield env
    env.close()


@pytest.fixture
def small_env():
    """Fixture for small environment (2 worlds) for quick tests"""
    env = MadronaEscapeRoomEnv(num_worlds=2)
    yield env
    env.close()


def test_env_creation_cpu():
    """Test environment creation on CPU"""
    env = MadronaEscapeRoomEnv(num_worlds=4, gpu_id=-1, rand_seed=42, auto_reset=True)

    assert env.num_worlds == 4
    assert env.num_agents == madrona_escape_room.consts.numAgents
    assert env.batch_size == torch.Size([4])
    assert env.device == torch.device("cpu")

    env.close()


def test_action_spec(small_env):
    """Test action specification"""
    # Check action spec structure
    assert "move_amount" in small_env.action_spec
    assert "move_angle" in small_env.action_spec
    assert "rotate" in small_env.action_spec

    # Check action dimensions
    assert small_env.action_spec["move_amount"].n == 4
    assert small_env.action_spec["move_angle"].n == 8
    assert small_env.action_spec["rotate"].n == 5

    # Check that we can sample valid actions
    action = small_env.action_spec.rand()
    assert action["move_amount"].shape == torch.Size([2])
    assert action["move_angle"].shape == torch.Size([2])
    assert action["rotate"].shape == torch.Size([2])


def test_observation_spec(small_env):
    """Test observation specification"""
    # Check observation spec
    assert "observation" in small_env.observation_spec
    obs_spec = small_env.observation_spec["observation"]

    # Check nested structure
    assert "self_obs" in obs_spec

    # Check individual component shapes
    assert obs_spec["self_obs"].shape == torch.Size([2, madrona_escape_room.consts.numAgents, 5])

    # Check dtypes
    assert obs_spec["self_obs"].dtype == torch.float32


def test_reset(cpu_env):
    """Test environment reset"""
    # Reset environment
    td = cpu_env.reset()

    # Check output structure
    assert isinstance(td, TensorDict)
    assert "observation" in td

    # Check nested observation structure
    obs = td["observation"]
    assert isinstance(obs, TensorDict)
    assert "self_obs" in obs

    # Check individual component shapes
    assert obs["self_obs"].shape == torch.Size([4, madrona_escape_room.consts.numAgents, 5])

    # Check that observations are valid
    assert not torch.isnan(obs["self_obs"]).any()
    assert not torch.isinf(obs["self_obs"]).any()


def test_step(small_env):
    """Test environment step"""
    # Reset first
    td = small_env.reset()

    # Create action
    action = TensorDict(
        {
            "move_amount": torch.tensor([1, 2]),
            "move_angle": torch.tensor([0, 4]),
            "rotate": torch.tensor([2, 2]),
        },
        batch_size=[2],
    )

    td["action"] = action

    # Step environment
    td_out = small_env.step(td)

    # Check output structure - TorchRL puts step results in "next"
    assert "next" in td_out
    assert "observation" in td_out["next"]
    assert "reward" in td_out["next"]
    assert "done" in td_out["next"]
    assert "terminated" in td_out["next"]
    assert "truncated" in td_out["next"]

    # Check observation structure
    next_obs = td_out["next"]["observation"]
    assert isinstance(next_obs, TensorDict)
    assert "self_obs" in next_obs

    # Check info structure for steps_remaining
    assert "info" in td_out["next"]
    info = td_out["next"]["info"]
    assert "steps_remaining" in info

    # Check shapes
    assert next_obs["self_obs"].shape == torch.Size([2, madrona_escape_room.consts.numAgents, 5])
    assert info["steps_remaining"].shape == torch.Size([2, madrona_escape_room.consts.numAgents, 1])
    assert td_out["next"]["reward"].shape == torch.Size(
        [2, madrona_escape_room.consts.numAgents, 1]
    )
    assert td_out["next"]["done"].shape == torch.Size([2, madrona_escape_room.consts.numAgents, 1])

    # Check types
    assert td_out["next"]["reward"].dtype == torch.float32
    assert td_out["next"]["done"].dtype == torch.bool


def test_batch_consistency():
    """Test that batched environments maintain consistency"""
    num_worlds = 8
    env = MadronaEscapeRoomEnv(num_worlds=num_worlds)

    # Reset
    td = env.reset()

    # Run several steps with different actions per world
    for step in range(10):
        # Each world gets different actions
        action = TensorDict(
            {
                "move_amount": torch.randint(0, 4, (num_worlds,)),
                "move_angle": torch.randint(0, 8, (num_worlds,)),
                "rotate": torch.randint(0, 5, (num_worlds,)),
            },
            batch_size=[num_worlds],
        )

        td["action"] = action
        td = env.step(td)

        # Verify batch dimensions are preserved
        assert td["next"]["observation"].shape[0] == num_worlds
        assert td["next"]["reward"].shape[0] == num_worlds
        assert td["next"]["done"].shape[0] == num_worlds

    env.close()


def test_auto_reset():
    """Test automatic reset functionality"""
    env = MadronaEscapeRoomEnv(num_worlds=2, auto_reset=True)

    td = env.reset()

    # Run many steps to ensure some episodes complete
    episode_ended = False
    for _ in range(300):  # More than episode length
        # Always move forward fast
        action = TensorDict(
            {
                "move_amount": torch.tensor([3, 3]),
                "move_angle": torch.tensor([0, 0]),
                "rotate": torch.tensor([2, 2]),
            },
            batch_size=[2],
        )

        td["action"] = action
        td = env.step(td)

        if td["next"]["done"].any():
            episode_ended = True
            # Next step should have reset the done environments
            # This is handled automatically by Madrona with auto_reset=True

    assert episode_ended, "No episode ended in 300 steps"
    env.close()


def test_stateful_behavior():
    """Test that the environment maintains state correctly"""
    env = MadronaEscapeRoomEnv(num_worlds=1)

    # Reset and get initial observation
    td1 = env.reset()
    obs1 = td1["observation"].clone()
    obs1_self = obs1["self_obs"].clone()

    # Take a specific action
    action = TensorDict(
        {
            "move_amount": torch.tensor([3]),  # Fast forward
            "move_angle": torch.tensor([0]),  # Forward
            "rotate": torch.tensor([2]),  # No rotation
        },
        batch_size=[1],
    )

    td1["action"] = action
    td2 = env.step(td1)
    obs2 = td2["next"]["observation"]
    obs2_self = obs2["self_obs"]

    # Position should have changed (Y coordinate increases when moving forward)
    # self_obs format: [globalX, globalY, globalZ, maxY, theta]
    assert not torch.allclose(obs1_self, obs2_self), "Self observation didn't change after movement"

    # Specifically, globalY and maxY should increase
    assert obs2_self[0, 0, 1] > obs1_self[0, 0, 1], "Global Y position didn't increase"
    assert obs2_self[0, 0, 3] >= obs1_self[0, 0, 3], "Max Y didn't update correctly"

    # Steps remaining should be in info and should decrease
    info1 = td1.get("info", None)
    info2 = td2["next"]["info"]
    if info1 is not None:
        assert (
            info2["steps_remaining"][0, 0, 0] < info1["steps_remaining"][0, 0, 0]
        ), "Steps remaining didn't decrease"

    env.close()


def test_device_placement_cpu():
    """Test that tensors are on CPU"""
    env = MadronaEscapeRoomEnv(num_worlds=2, gpu_id=-1, device="cpu")

    td = env.reset()

    # Check all observation components are on correct device
    obs = td["observation"]
    assert obs["self_obs"].device.type == "cpu"

    # Step and check again
    action = env.action_spec.rand()
    td["action"] = action
    td_out = env.step(td)

    next_obs = td_out["next"]["observation"]
    assert next_obs["self_obs"].device.type == "cpu"
    assert td_out["next"]["reward"].device.type == "cpu"
    assert td_out["next"]["done"].device.type == "cpu"

    # Check info is on correct device
    info = td_out["next"]["info"]
    assert info["steps_remaining"].device.type == "cpu"

    env.close()


def test_seed_determinism():
    """Test that setting seed produces deterministic results"""
    # Create two environments with same seed
    env1 = MadronaEscapeRoomEnv(num_worlds=2, rand_seed=123)
    env2 = MadronaEscapeRoomEnv(num_worlds=2, rand_seed=123)

    # Reset both
    td1 = env1.reset()
    td2 = env2.reset()

    # Initial observations should be identical
    obs1 = td1["observation"]
    obs2 = td2["observation"]
    assert torch.allclose(obs1["self_obs"], obs2["self_obs"])

    # Take same actions
    action = TensorDict(
        {
            "move_amount": torch.tensor([2, 1]),
            "move_angle": torch.tensor([3, 5]),
            "rotate": torch.tensor([1, 3]),
        },
        batch_size=[2],
    )

    td1["action"] = action.clone()
    td2["action"] = action.clone()

    td1_out = env1.step(td1)
    td2_out = env2.step(td2)

    # Results should be identical
    next_obs1 = td1_out["next"]["observation"]
    next_obs2 = td2_out["next"]["observation"]
    assert torch.allclose(next_obs1["self_obs"], next_obs2["self_obs"])

    # Check info is identical
    info1 = td1_out["next"]["info"]
    info2 = td2_out["next"]["info"]
    assert torch.allclose(info1["steps_remaining"], info2["steps_remaining"])

    assert torch.allclose(td1_out["next"]["reward"], td2_out["next"]["reward"])
    assert torch.equal(td1_out["next"]["done"], td2_out["next"]["done"])

    env1.close()
    env2.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
