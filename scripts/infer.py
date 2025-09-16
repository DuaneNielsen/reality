import argparse
import warnings

import numpy as np
import torch
from madrona_escape_room_learn import LearningState
from madrona_escape_room_learn.moving_avg import EpisodicEMATrackerWithHistogram
from madrona_escape_room_learn.sim_interface_adapter import setup_lidar_training_environment
from policy import make_policy, setup_obs

import madrona_escape_room
from madrona_escape_room.level_io import load_compiled_levels

warnings.filterwarnings("error")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--gpu-id", type=int, default=0)
arg_parser.add_argument("--ckpt-path", type=str, required=True)
arg_parser.add_argument("--action-dump-path", type=str)
arg_parser.add_argument("--recording-path", type=str, help="Path to save recording file")
arg_parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
arg_parser.add_argument("--level-file", type=str, help="Path to compiled .lvl level file")

arg_parser.add_argument("--num-worlds", type=int, required=True)
arg_parser.add_argument("--num-steps", type=int, required=True)

arg_parser.add_argument("--num-channels", type=int, default=256)
arg_parser.add_argument("--separate-value", action="store_true")
arg_parser.add_argument("--fp16", action="store_true")

arg_parser.add_argument("--gpu-sim", action="store_true")

args = arg_parser.parse_args()

torch.manual_seed(args.seed)

exec_mode = madrona_escape_room.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.ExecMode.CPU

# Load custom level if provided
compiled_levels = None
if args.level_file:
    compiled_levels = load_compiled_levels(args.level_file)
    if len(compiled_levels) == 1:
        print(f"Loaded custom level from {args.level_file}")
    else:
        print(f"Loaded multi-level file with {len(compiled_levels)} levels from {args.level_file}")

sim_interface = setup_lidar_training_environment(
    num_worlds=args.num_worlds,
    exec_mode=exec_mode,
    gpu_id=args.gpu_id,
    rand_seed=args.seed,
    compiled_levels=compiled_levels,
)

obs, num_obs_features = setup_obs(sim_interface.obs)
policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

weights = LearningState.load_policy_weights(args.ckpt_path)
policy.load_state_dict(weights)

actions = sim_interface.actions
dones = sim_interface.dones
rewards = sim_interface.rewards

# Keep original tensor shapes:
# - actions: [worlds, 3]
# - dones: [worlds, 1] (1 agent per world)
# - rewards: [worlds, 1] (1 agent per world)

cur_rnn_states = []

for shape in policy.recurrent_cfg.shapes:
    cur_rnn_states.append(
        torch.zeros(
            *shape[0:2],
            actions.shape[0],
            shape[2],
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
    )

if args.action_dump_path:
    action_log = open(args.action_dump_path, "wb")
else:
    action_log = None

# Initialize episode tracker
device = torch.device("cuda" if args.gpu_sim else "cpu")
episode_tracker = EpisodicEMATrackerWithHistogram(
    num_envs=args.num_worlds,
    alpha=0.01,
    device=device,
    reward_bins=[-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0],
    length_bins=[1, 25, 50, 100, 150, 200],
)

# Keep detailed tracking for final statistics
episode_returns = []  # Store completed episode returns
episode_counts = torch.zeros(args.num_worlds, dtype=torch.int32)

# Start recording if recording path is provided
if args.recording_path:
    try:
        sim_interface.manager.start_recording(args.recording_path)
        print(f"Recording started: {args.recording_path}")
    except Exception as e:
        print(f"Failed to start recording: {e}")
        args.recording_path = None  # Disable recording

for i in range(args.num_steps):
    with torch.no_grad():
        action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
        action_dists.best(actions)

        probs = action_dists.probs()

    if action_log:
        actions.numpy().tofile(action_log)

    print(f"\n=== Step {i+1}/{args.num_steps} ===")
    print("Progress:", obs[0])
    print("Compass:", obs[1])
    print("Lidar:", obs[2])

    print("Move Amount Probs")
    print(" ", np.array_str(probs[0][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[0][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Move Angle Probs")
    print(" ", np.array_str(probs[1][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[1][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Rotate Probs")
    print(" ", np.array_str(probs[2][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[2][1].cpu().numpy(), precision=2, suppress_small=True))

    # Grab action removed - only 3 action components now

    print("Actions:\n", actions.cpu().numpy())
    print("Values:\n", values.cpu().numpy())

    # Step the simulation
    sim_interface.step()

    # Process rewards and episode completions
    # Rewards and dones are [worlds, 1] when there's 1 agent per world
    step_rewards = rewards[:, 0] if rewards.dim() == 2 else rewards[:, 0, 0]
    step_dones = dones[:, 0] if dones.dim() == 2 else dones[:, 0, 0]

    # Update episode tracker
    completed_episodes = episode_tracker.step_update(step_rewards, step_dones.bool())

    # Process completed episodes for detailed tracking
    if "completed_episodes" in completed_episodes:
        completed_data = completed_episodes["completed_episodes"]
        completed_rewards = completed_data["rewards"]
        completed_env_indices = completed_data["env_indices"]

        for i, env_idx in enumerate(completed_env_indices):
            episode_return = completed_rewards[i].item()
            episode_returns.append(episode_return)
            episode_counts[env_idx] += 1
            print(f"  Episode completed in world {env_idx}: return = {episode_return:.4f}")

    print(f"Step Rewards: {step_rewards.cpu().numpy()}")

    # Get current episode statistics
    episode_stats = episode_tracker.get_statistics()
    reward_ema = episode_stats["episodes/reward_ema"]
    length_ema = episode_stats["episodes/length_ema"]
    print(f"Episode EMA - Reward: {reward_ema:.3f}, Length: {length_ema:.1f}")

# Stop recording if it was started
if args.recording_path:
    try:
        sim_interface.manager.stop_recording()
        print(f"Recording completed: {args.recording_path}")
    except Exception as e:
        print(f"Failed to stop recording: {e}")

if action_log:
    action_log.close()

# Print final statistics
print("\n=== Final Statistics ===")
episode_stats = episode_tracker.get_statistics()
print(f"Total episodes completed: {episode_stats['episodes/completed']}")
if episode_stats["episodes/completed"] > 0:
    print(f"Average episode return (EMA): {episode_stats['episodes/reward_ema']:.4f}")
    print(f"Average episode length (EMA): {episode_stats['episodes/length_ema']:.1f}")
    print(f"Min episode return: {episode_stats['episodes/reward_min']:.4f}")
    print(f"Max episode return: {episode_stats['episodes/reward_max']:.4f}")
    print(f"Min episode length: {episode_stats['episodes/length_min']}")
    print(f"Max episode length: {episode_stats['episodes/length_max']}")
    if episode_returns:
        print(f"Std episode return: {np.std(episode_returns):.4f}")
        print(f"Episodes per world: {episode_counts.cpu().numpy()}")
else:
    print("No episodes completed")
