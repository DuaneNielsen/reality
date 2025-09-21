#!/usr/bin/env python3
"""
Script to run inference from a wandb run hash.
Finds the latest checkpoint and creates a recording with the same name but .rec extension.
"""

import argparse
import sys
from pathlib import Path

from inference_core import InferenceRunner
from inference_utils import create_inference_config_from_wandb, print_episode_statistics

# Import wandb utilities
from wandb_utils import find_wandb_run_dir, get_run_object, lookup_run

import wandb


def main():
    parser = argparse.ArgumentParser(description="Run inference from wandb run name or hash")
    parser.add_argument(
        "wandb_run_identifier",
        nargs="?",
        help="Wandb run name or hash (e.g., 'my-training-run' or 'iecno14m')",
    )
    parser.add_argument(
        "--list",
        nargs="?",
        const="",
        help="List available wandb runs (optionally filter with regex pattern)",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    parser.add_argument("--gpu-sim", action="store_true", help="Use GPU simulation")
    parser.add_argument("--num-worlds", type=int, default=4, help="Number of worlds")
    parser.add_argument("--num-steps", type=int, default=1000, help="Number of steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--num-channels",
        type=int,
        default=None,
        help="Number of channels (auto-detected from wandb config)",
    )
    parser.add_argument("--separate-value", action="store_true", help="Use separate value network")
    parser.add_argument("--sim-seed", type=int, default=0, help="Seed for simulation")
    parser.add_argument("--level-file", type=str, help="Path to compiled .lvl level file")
    parser.add_argument(
        "--project", type=str, default="madrona-escape-room", help="Wandb project name"
    )
    parser.add_argument(
        "--lookup", type=str, help="Look up run hash for given human readable name and exit"
    )
    parser.add_argument(
        "--recording-path",
        type=str,
        help="Custom path for recording file (overrides default .rec name)",
    )

    args = parser.parse_args()

    # Handle lookup option
    if args.lookup:
        result = lookup_run(args.lookup, args.project)
        print(result)
        return

    # Handle list option
    if args.list is not None:
        import re

        api = wandb.Api()
        runs = api.runs(args.project)

        # Filter runs if regex pattern provided
        if args.list:  # Non-empty string means regex pattern provided
            pattern = re.compile(args.list, re.IGNORECASE)
            filtered_runs = [run for run in runs if pattern.search(run.name)]
            print(f"Wandb runs matching '{args.list}':")
        else:
            filtered_runs = runs
            print("All wandb runs:")

        for i, run in enumerate(filtered_runs):
            print(f"{i+1:3d}. {run.name} ({run.id}) - {run.state}")

        print(f"Total: {len(filtered_runs)} runs")
        if args.list and len(filtered_runs) != len(runs):
            print(f"(Filtered from {len(runs)} total runs)")
        return

    # Require run identifier if not listing
    if not args.wandb_run_identifier:
        parser.error("wandb_run_identifier is required unless --list is used")

    # Create config from wandb run
    try:
        # Get the wandb run to extract model parameters
        from wandb_utils import find_wandb_run_dir, get_run_object

        from madrona_escape_room.level_io import load_compiled_levels

        run = get_run_object(args.wandb_run_identifier, args.project)
        if isinstance(run, str) and run.startswith("ERROR"):
            raise ValueError(run)

        # Build model kwargs dict - extract from wandb config and override with command line
        model_kwargs = {}

        # Get from wandb config
        if hasattr(run, "config"):
            if "num_channels" in run.config:
                model_kwargs["num_channels"] = run.config["num_channels"]
            if "separate_value" in run.config:
                model_kwargs["separate_value"] = run.config["separate_value"]

        # Override with command line args if provided
        if args.num_channels is not None:
            model_kwargs["num_channels"] = args.num_channels
        if args.separate_value:
            model_kwargs["separate_value"] = args.separate_value

        # Determine level file - command line override takes precedence
        level_file = None
        compiled_levels = []
        num_worlds = args.num_worlds  # Use command line value

        if args.level_file:
            # Use command line override
            level_file = args.level_file
            compiled_levels = load_compiled_levels(level_file)
            # Override num_worlds to match level file unless explicitly set
            if args.num_worlds == 4:  # default value
                num_worlds = len(compiled_levels)
        else:
            # Use default from wandb directory
            run_dir_result = find_wandb_run_dir(run.id)
            if run_dir_result != "NOT_FOUND" and not run_dir_result.startswith("ERROR:"):
                run_dir = Path(run_dir_result)
                checkpoints_dir = run_dir / "files" / "checkpoints"
                if checkpoints_dir.exists():
                    lvl_files = list(checkpoints_dir.glob("*.lvl"))
                    if lvl_files:
                        level_file = str(lvl_files[0])
                        compiled_levels = load_compiled_levels(level_file)
                        # Override num_worlds to match level file unless explicitly set
                        if args.num_worlds == 4:  # default value
                            num_worlds = len(compiled_levels)

        # Build overrides with resolved values
        overrides = {
            "model_kwargs": model_kwargs,
            "level_file": level_file,
            "compiled_levels": compiled_levels,
            "num_worlds": num_worlds,
            "num_steps": args.num_steps,
            "exec_mode": "CUDA" if args.gpu_sim else "CPU",
            "gpu_id": args.gpu_id,
            "sim_seed": args.sim_seed,
            "fp16": args.fp16,
        }

        config = create_inference_config_from_wandb(
            args.wandb_run_identifier,
            args.project,
            overrides=overrides,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Set exec_mode properly
    import madrona_escape_room

    if args.gpu_sim:
        config.exec_mode = madrona_escape_room.ExecMode.CUDA
    else:
        config.exec_mode = madrona_escape_room.ExecMode.CPU

    print(f"Using checkpoint: {config.ckpt_path}")

    # Create recording path - use custom path if provided, otherwise default to checkpoint.rec
    if args.recording_path:
        recording_path = Path(args.recording_path)
        config.recording_path = str(recording_path)
        print(f"Recording will be saved to: {recording_path} (custom path)")
    else:
        ckpt_path = Path(config.ckpt_path)
        recording_path = ckpt_path.with_suffix(".rec")
        config.recording_path = str(recording_path)
        print(f"Recording will be saved to: {recording_path} (default path)")

    # Set up for detailed output like infer_from_wandb.py
    config.verbose = True
    config.print_interval = 100  # Print every 100 steps instead of every step

    # Keep detailed tracking for per-world analysis
    # TODO: Implement per-world analysis if needed

    def detailed_wandb_callback(step, obs, actions, rewards, dones, values, probs):
        # Only print every 100 steps for wandb-style output
        if step % 100 == 0:
            stats = runner.episode_tracker.get_statistics()
            episodes_completed = stats["episodes/completed"]
            print(f"Step {step+1}/{config.num_steps} - Episodes completed: {episodes_completed}")
            print("Compass:", obs[0])
            print("Lidar:", obs[1])
            print("Actions:", actions.cpu().numpy())
            print(f"Step Rewards: {rewards.cpu().numpy()}")
            reward_ema = stats["episodes/reward_ema"]
            length_ema = stats["episodes/length_ema"]
            print(f"Episode EMA - Reward: {reward_ema:.3f}, Length: {length_ema:.1f}")
            print()

    # Create and run inference
    runner = InferenceRunner(config)
    print(f"Model kwargs: {config.model_kwargs}")

    # Run inference
    episode_tracker, _ = runner.run_steps(callback=detailed_wandb_callback)

    # Print episode statistics in wandb style
    print("\n" + "=" * 60)
    print("EPISODE STATISTICS")
    print("=" * 60)

    stats = episode_tracker.get_statistics()
    total_episodes = stats.get("episodes/completed", 0)

    if total_episodes > 0:
        print(f"Total episodes completed: {total_episodes}")
        print()

        # Overall statistics
        print("OVERALL STATISTICS:")
        reward_ema = stats["episodes/reward_ema"]
        length_ema = stats["episodes/length_ema"]
        print(f"Episode Return - Mean: {reward_ema:.3f}")
        print(f"Episode Length - Mean: {length_ema:.3f}")
        print()

        # Show termination reasons if available
        if "episodes/termination_time_limit_prob" in stats:
            print("Termination Reasons:")
            time_limit = stats["episodes/termination_time_limit_prob"]
            progress_complete = stats["episodes/termination_progress_complete_prob"]
            collision_death = stats["episodes/termination_collision_death_prob"]
            print(f"  Time Limit: {time_limit:.1%}")
            print(f"  Progress Complete: {progress_complete:.1%}")
            print(f"  Collision Death: {collision_death:.1%}")

        # Note: Per-world statistics would require more complex tracking
        # that the current EMA tracker doesn't provide
        print("\nNote: Per-world statistics not available with EMA tracker")
    else:
        print("No episodes completed during inference")


if __name__ == "__main__":
    main()
