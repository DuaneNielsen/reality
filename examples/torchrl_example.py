"""Example usage of Madrona Escape Room with TorchRL

This script demonstrates how to use the Madrona Escape Room environment
with TorchRL for reinforcement learning tasks.
"""

import torch
from tensordict import TensorDict
from torchrl.envs import TransformedEnv, StepCounter, RewardSum
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.modules import ProbabilisticActor, ValueOperator, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torch import nn
import math

# Import from madrona_escape_room_learn package
from madrona_escape_room_learn import MadronaEscapeRoomEnv


def make_env(num_worlds=32, gpu_id=-1, transforms=True):
    """Create and optionally transform the environment.
    
    Args:
        num_worlds: Number of parallel worlds to simulate
        gpu_id: GPU device ID (-1 for CPU)
        transforms: Whether to apply transforms like StepCounter
    
    Returns:
        Environment instance (possibly transformed)
    """
    # Create base environment
    env = MadronaEscapeRoomEnv(
        num_worlds=num_worlds,
        gpu_id=gpu_id,
        rand_seed=42,
        auto_reset=True
    )
    
    if transforms:
        # Add transforms for tracking episode statistics
        env = TransformedEnv(env)
        env.append_transform(StepCounter())
        env.append_transform(RewardSum())
    
    return env


def make_simple_policy(env):
    """Create a simple policy network for discrete actions.
    
    Args:
        env: The environment instance
        
    Returns:
        Actor module for the policy
    """
    obs_shape = env.observation_spec["observation"].shape[-1]
    
    # Simple MLP backbone
    backbone = nn.Sequential(
        nn.Linear(obs_shape, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
    )
    
    # Separate heads for each discrete action
    move_amount_head = nn.Linear(128, 4)
    move_angle_head = nn.Linear(128, 8)
    rotate_head = nn.Linear(128, 5)
    
    class DiscretePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.move_amount_head = move_amount_head
            self.move_angle_head = move_angle_head
            self.rotate_head = rotate_head
            
        def forward(self, obs):
            features = self.backbone(obs)
            return TensorDict({
                "move_amount_logits": self.move_amount_head(features),
                "move_angle_logits": self.move_angle_head(features),
                "rotate_logits": self.rotate_head(features),
            }, batch_size=obs.shape[:-1])
    
    return DiscretePolicy()


def random_rollout_example():
    """Example 1: Random action rollout"""
    print("=" * 60)
    print("Example 1: Random Action Rollout")
    print("=" * 60)
    
    # Create environment
    env = make_env(num_worlds=4, gpu_id=-1)
    
    # Reset environment
    td = env.reset()
    print(f"Initial observation shape: {td['observation'].shape}")
    print(f"Batch size: {env.batch_size}")
    
    # Run a few steps with random actions
    total_reward = 0
    for i in range(10):
        # Sample random actions
        action = env.action_spec.rand()
        td["action"] = action
        
        # Step environment
        td = env.step(td)
        
        step_reward = td["next"]["reward"].sum().item()
        total_reward += step_reward
        
        if i < 3:  # Print first few steps
            print(f"\nStep {i+1}:")
            print(f"  Actions - move_amount: {action['move_amount'][:4]}")
            print(f"  Rewards: {td['next']['reward'].squeeze()[:4]}")
            print(f"  Done: {td['next']['done'].squeeze()[:4]}")
    
    print(f"\nTotal reward over 10 steps: {total_reward:.2f}")
    env.close()


def rollout_with_policy_example():
    """Example 2: Rollout with a simple policy"""
    print("\n" + "=" * 60)
    print("Example 2: Policy-Based Rollout")
    print("=" * 60)
    
    # Create environment with transforms
    env = make_env(num_worlds=8, gpu_id=-1, transforms=True)
    
    # Create a simple policy
    policy = make_simple_policy(env)
    
    # Reset and get initial observation
    td = env.reset()
    
    # Rollout for one episode
    done = False
    step = 0
    episode_rewards = []
    
    while not done and step < 200:
        # Get observation
        obs = td["observation"]
        
        # Forward through policy
        with torch.no_grad():
            logits = policy(obs)
            
            # Sample actions from logits
            action = TensorDict({
                "move_amount": torch.distributions.Categorical(
                    logits=logits["move_amount_logits"]
                ).sample(),
                "move_angle": torch.distributions.Categorical(
                    logits=logits["move_angle_logits"]
                ).sample(),
                "rotate": torch.distributions.Categorical(
                    logits=logits["rotate_logits"]
                ).sample(),
            }, batch_size=env.batch_size)
        
        td["action"] = action
        td = env.step(td)
        
        # Track rewards
        episode_rewards.append(td["next"]["reward"])
        
        # Check if any environment is done
        done = td["next"]["done"].any().item()
        step += 1
    
    # Report statistics
    total_rewards = torch.stack(episode_rewards).sum(dim=0)
    print(f"Episode lengths: {td.get('step_count', step)}")
    print(f"Episode rewards per world: {total_rewards.squeeze()}")
    print(f"Average reward: {total_rewards.mean().item():.2f}")
    
    env.close()


def data_collection_example():
    """Example 3: Using TorchRL's data collector"""
    print("\n" + "=" * 60)
    print("Example 3: Data Collection with SyncDataCollector")
    print("=" * 60)
    
    # Create environment
    env = make_env(num_worlds=16, gpu_id=-1, transforms=True)
    
    # Create policy
    policy = make_simple_policy(env)
    
    # Wrap policy to output actions properly
    class PolicyWrapper(nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy
            
        def forward(self, td):
            obs = td["observation"]
            logits = self.policy(obs)
            
            # Sample actions
            actions = TensorDict({
                "move_amount": torch.distributions.Categorical(
                    logits=logits["move_amount_logits"]
                ).sample(),
                "move_angle": torch.distributions.Categorical(
                    logits=logits["move_angle_logits"]
                ).sample(),
                "rotate": torch.distributions.Categorical(
                    logits=logits["rotate_logits"]
                ).sample(),
            }, batch_size=obs.shape[:-1])
            
            td["action"] = actions
            return td
    
    policy_wrapper = PolicyWrapper(policy)
    
    # Create data collector
    collector = SyncDataCollector(
        env,
        policy_wrapper,
        frames_per_batch=1000,
        total_frames=5000,
        device="cpu",
    )
    
    # Collect data
    total_episodes = 0
    total_reward = 0
    
    for i, batch in enumerate(collector):
        # Batch contains trajectories from multiple environments
        rewards = batch.get("episode_reward")
        if rewards is not None:
            episode_rewards = rewards[rewards != 0]  # Filter out zeros from padding
            if len(episode_rewards) > 0:
                total_episodes += len(episode_rewards)
                total_reward += episode_rewards.sum().item()
                print(f"Batch {i+1}: Completed {len(episode_rewards)} episodes, "
                      f"avg reward: {episode_rewards.mean().item():.2f}")
    
    print(f"\nTotal episodes collected: {total_episodes}")
    print(f"Average episode reward: {total_reward / max(total_episodes, 1):.2f}")
    
    collector.shutdown()
    env.close()


def batch_inference_example():
    """Example 4: Efficient batch inference"""
    print("\n" + "=" * 60)
    print("Example 4: Efficient Batch Inference")
    print("=" * 60)
    
    # Create large batch environment for inference
    num_worlds = 256  # Large batch
    env = make_env(num_worlds=num_worlds, gpu_id=-1)
    
    # Simple policy
    policy = make_simple_policy(env)
    policy.eval()
    
    # Reset
    td = env.reset()
    
    # Time the inference
    import time
    start_time = time.time()
    
    # Run 100 steps
    with torch.no_grad():
        for _ in range(100):
            obs = td["observation"]
            logits = policy(obs)
            
            # Greedy action selection for inference
            action = TensorDict({
                "move_amount": logits["move_amount_logits"].argmax(dim=-1),
                "move_angle": logits["move_angle_logits"].argmax(dim=-1),
                "rotate": logits["rotate_logits"].argmax(dim=-1),
            }, batch_size=env.batch_size)
            
            td["action"] = action
            td = env.step(td)
    
    elapsed = time.time() - start_time
    total_steps = num_worlds * 100
    
    print(f"Simulated {total_steps} steps across {num_worlds} worlds")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Steps per second: {total_steps / elapsed:.0f}")
    print(f"FPS per world: {100 / elapsed:.1f}")
    
    env.close()


if __name__ == "__main__":
    # Run all examples
    random_rollout_example()
    rollout_with_policy_example()
    data_collection_example()
    batch_inference_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)