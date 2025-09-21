import argparse
import warnings

import numpy as np
import torch
from inference_core import InferenceRunner
from inference_utils import create_inference_config_from_args, print_episode_statistics

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

# Load custom level if provided
compiled_levels = None
if args.level_file:
    compiled_levels = load_compiled_levels(args.level_file)
    if len(compiled_levels) == 1:
        print(f"Loaded custom level from {args.level_file}")
    else:
        print(f"Loaded multi-level file with {len(compiled_levels)} levels from {args.level_file}")

# Create configuration from args
config = create_inference_config_from_args(args)
config.compiled_levels = compiled_levels

# Create and run inference
runner = InferenceRunner(config)

# Keep detailed tracking for final statistics
episode_returns = []  # Store completed episode returns
episode_counts = torch.zeros(args.num_worlds, dtype=torch.int32)

# Track episodes completed this step for detailed logging
last_completed_count = 0


# Custom callback to collect detailed episode data and print verbose output
def detailed_callback(step, obs, actions, rewards, dones, values, probs):
    global last_completed_count

    # Print detailed step information (matching original behavior)
    print(f"\n=== Step {step+1}/{args.num_steps} ===")
    print("Compass:", obs[0])
    print("Lidar:", obs[1])

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

    # Process completed episodes for detailed tracking
    current_stats = runner.episode_tracker.get_statistics()
    current_completed = current_stats["episodes/completed"]

    # Check if any episodes completed this step
    if current_completed > last_completed_count:
        for env_idx in range(args.num_worlds):
            if dones[env_idx]:
                episode_counts[env_idx] += 1
                # We can't get the exact episode return here easily, so we'll use EMA
                print(f"  Episode completed in world {env_idx}")
        last_completed_count = current_completed

    print(f"Step Rewards: {rewards.cpu().numpy()}")

    # Get current episode statistics
    reward_ema = current_stats["episodes/reward_ema"]
    length_ema = current_stats["episodes/length_ema"]
    print(f"Episode EMA - Reward: {reward_ema:.3f}, Length: {length_ema:.1f}")


# Run inference with detailed callback
episode_tracker, _ = runner.run_steps(callback=detailed_callback)

# Print final statistics
print("\n=== Final Statistics ===")
episode_stats = episode_tracker.get_statistics()
print(f"Total episodes completed: {episode_stats['episodes/completed']}")
if episode_stats["episodes/completed"] > 0:
    print(f"Average episode return (EMA): {episode_stats['episodes/reward_ema']:.4f}")
    print(f"Average episode length (EMA): {episode_stats['episodes/length_ema']:.1f}")
    print(f"Episodes per world: {episode_counts.cpu().numpy()}")
else:
    print("No episodes completed")
