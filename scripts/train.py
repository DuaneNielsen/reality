import argparse
import shutil
from pathlib import Path

import torch
from madrona_escape_room_learn import (
    PPOConfig,
    SimInterface,
    TrainConfig,
    profile,
    train,
)
from madrona_escape_room_learn.sim_interface_adapter import setup_lidar_training_environment
from policy import make_policy, setup_obs

import madrona_escape_room
from madrona_escape_room.generated_constants import ExecMode
from madrona_escape_room.level_io import load_compiled_levels

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class LearningCallback:
    def __init__(
        self,
        ckpt_dir,
        profile_report,
        training_config=None,
        level_file_path=None,
        additional_tags=None,
    ):
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = profile_report
        self.training_config = training_config or {}

        # Episode reward tracking
        self.episode_reward_max = float("-inf")
        self.episode_reward_min = float("inf")

        # Initialize wandb
        self.use_wandb = False
        if WANDB_AVAILABLE:
            try:
                # Create tags based on level
                level_tag = (
                    training_config.get("level_name", "default_16x16_room")
                    if training_config
                    else "default_16x16_room"
                )
                tags = ["lidar-training", level_tag]
                if training_config and training_config.get("level_file"):
                    tags.append("custom-level")

                # Add additional tags from command line
                if additional_tags:
                    tags.extend(additional_tags)

                wandb.init(
                    project="madrona-escape-room",
                    config=training_config or {},
                    tags=tags,
                )
                self.use_wandb = True
                print("✓ wandb initialized")
            except Exception as e:
                print(f"Warning: wandb initialization failed: {e}")
        else:
            print("Warning: wandb not available - metrics will only be printed to console")

        # Set checkpoint directory
        if ckpt_dir is not None:
            self.ckpt_dir = ckpt_dir
            print(f"Using provided checkpoint directory: {self.ckpt_dir}")
        elif self.use_wandb:
            self.ckpt_dir = Path(wandb.run.dir) / "checkpoints"
            print(f"Using wandb checkpoint directory: {self.ckpt_dir}")
        else:
            self.ckpt_dir = Path("./checkpoints")
            print(f"Using default checkpoint directory: {self.ckpt_dir}")

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Copy level file to checkpoint directory if provided
        if level_file_path:
            level_file_src = Path(level_file_path)
            if level_file_src.exists():
                level_file_dst = self.ckpt_dir / level_file_src.name
                try:
                    shutil.copy2(level_file_src, level_file_dst)
                    print(f"Copied level file to checkpoint directory: {level_file_dst}")
                except Exception as e:
                    print(f"Warning: Failed to copy level file: {e}")

    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        fps = args.num_worlds * args.steps_per_update / update_time
        self.mean_fps += (fps - self.mean_fps) / update_id

        # Always calculate performance metrics for wandb
        reserved_gb = (
            torch.cuda.memory_reserved() / 1024 / 1024 / 1024 if torch.cuda.is_available() else 0
        )
        current_gb = (
            torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
            if torch.cuda.is_available()
            else 0
        )

        if update_id != 1 and update_id % 10 != 0:
            # Log performance metrics to wandb even when not printing
            if self.use_wandb:
                wandb.log(
                    {
                        "performance/fps": fps,
                        "performance/avg_fps": self.mean_fps,
                        "performance/update_time": update_time,
                        "performance/memory_reserved_gb": reserved_gb,
                        "performance/memory_current_gb": current_gb,
                    },
                    step=update_id,
                )
            return

        ppo = update_results.ppo_stats

        with torch.no_grad():
            reward_mean = update_results.rewards.mean().cpu().item()
            reward_min = update_results.rewards.min().cpu().item()
            reward_max = update_results.rewards.max().cpu().item()

            value_mean = update_results.values.mean().cpu().item()
            value_min = update_results.values.min().cpu().item()
            value_max = update_results.values.max().cpu().item()

            advantage_mean = update_results.advantages.mean().cpu().item()
            advantage_min = update_results.advantages.min().cpu().item()
            advantage_max = update_results.advantages.max().cpu().item()

            bootstrap_value_mean = update_results.bootstrap_values.mean().cpu().item()
            bootstrap_value_min = update_results.bootstrap_values.min().cpu().item()
            bootstrap_value_max = update_results.bootstrap_values.max().cpu().item()

            # Use entropy from PPO stats (entropy_loss is negative entropy scaled by coef)
            # Convert back to actual entropy by dividing by negative entropy coefficient
            actual_entropy = (
                -ppo.entropy_loss / args.entropy_loss_coef if args.entropy_loss_coef != 0 else 0.0
            )
            action_entropy_mean = actual_entropy
            action_entropy_min = actual_entropy  # Single aggregate value
            action_entropy_max = actual_entropy  # Single aggregate value

            # Only extract normalizer stats if normalization is enabled
            value_normalization_enabled = self.training_config.get(
                "value_normalization_enabled", True
            )
            if value_normalization_enabled:
                vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
                vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

        # Log to wandb
        if self.use_wandb:
            wandb_data = {
                # PPO losses
                "losses/total_loss": ppo.loss,
                "losses/action_loss": ppo.action_loss,
                "losses/value_loss": ppo.value_loss,
                "losses/entropy_loss": ppo.entropy_loss,
                # Rewards
                "rewards/mean": reward_mean,
                "rewards/min": reward_min,
                "rewards/max": reward_max,
                # Values
                "values/mean": value_mean,
                "values/min": value_min,
                "values/max": value_max,
                # Advantages
                "advantages/mean": advantage_mean,
                "advantages/min": advantage_min,
                "advantages/max": advantage_max,
                # Bootstrap values
                "bootstrap_values/mean": bootstrap_value_mean,
                "bootstrap_values/min": bootstrap_value_min,
                "bootstrap_values/max": bootstrap_value_max,
                # Action entropy (exploration measure)
                "action_entropy/mean": action_entropy_mean,
                "action_entropy/min": action_entropy_min,
                "action_entropy/max": action_entropy_max,
                # Returns
                "returns/mean": ppo.returns_mean,
                "returns/stddev": ppo.returns_stddev,
                # Episode tracking
                "episodes/length_ema": update_results.episode_length_ema,
                "episodes/reward_ema": update_results.episode_reward_ema,
                "episodes/reward_max": update_results.episode_reward_max,
                "episodes/reward_min": update_results.episode_reward_min,
                # Performance metrics
                "performance/fps": fps,
                "performance/avg_fps": self.mean_fps,
                "performance/update_time": update_time,
                "performance/memory_reserved_gb": reserved_gb,
                "performance/memory_current_gb": current_gb,
            }

            # Only log value normalizer stats if normalization is enabled
            if value_normalization_enabled:
                wandb_data["value_normalizer/mu"] = vnorm_mu
                wandb_data["value_normalizer/sigma"] = vnorm_sigma

            wandb.log(wandb_data, step=update_id)

        # Keep original console output
        if self.use_wandb and wandb.run is not None:
            level_info = self.training_config.get("level_name", "unknown")
            print(f"\nUpdate: {update_id} [{wandb.run.name}] ({wandb.run.id}) [{level_info}]")
        else:
            level_info = self.training_config.get("level_name", "unknown")
            print(f"\nUpdate: {update_id} [{level_info}]")
        print(
            f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, "
            f"V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}"
        )
        print()
        print(
            f"    Rewards          => Avg: {reward_mean:.3f}, "
            f"Min: {reward_min:.3f}, Max: {reward_max:.3f}"
        )
        print(
            f"    Values           => Avg: {value_mean:.3f}, "
            f"Min: {value_min:.3f}, Max: {value_max:.3f}"
        )
        print(
            f"    Advantages       => Avg: {advantage_mean:.3f}, "
            f"Min: {advantage_min:.3f}, Max: {advantage_max:.3f}"
        )
        print(
            f"    Bootstrap Values => Avg: {bootstrap_value_mean:.3f}, "
            f"Min: {bootstrap_value_min:.3f}, Max: {bootstrap_value_max:.3f}"
        )
        print(
            f"    Action Entropy   => Avg: {action_entropy_mean:.3f}, "
            f"Min: {action_entropy_min:.3f}, Max: {action_entropy_max:.3f}"
        )
        print(f"    Returns          => Avg: {ppo.returns_mean:.3f}, σ: {ppo.returns_stddev:.3f}")
        print(f"    Episode Length   => EMA: {update_results.episode_length_ema:.1f} steps")
        print(
            f"    Episode Reward   => EMA: {update_results.episode_reward_ema:.3f}, "
            f"Max: {update_results.episode_reward_max:.3f}, "
            f"Min: {update_results.episode_reward_min:.3f}"
        )

        # Only show value normalizer stats if normalization is enabled
        value_normalization_enabled = self.training_config.get("value_normalization_enabled", True)
        if value_normalization_enabled:
            print(f"    Value Normalizer => Mean: {vnorm_mu:.3f}, σ: {vnorm_sigma:.3f}")
        else:
            print("    Value Normalizer => DISABLED")

        if self.profile_report:
            print()
            print(
                f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}"
            )
            print(
                f"    PyTorch Memory Usage: {reserved_gb:.3f}GB (Reserved), "
                f"{current_gb:.3f}GB (Current)"
            )
            profile.report()

        if update_id % 100 == 0:
            learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")

    def finish(self):
        """Clean up wandb if it was initialized"""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as e:
                print(f"Warning: wandb cleanup failed: {e}")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--gpu-id", type=int, default=0)
arg_parser.add_argument("--ckpt-dir", type=str, default=None)
arg_parser.add_argument("--restore", type=int)

arg_parser.add_argument("--num-worlds", type=int, required=True)
arg_parser.add_argument("--num-updates", type=int, required=True)
arg_parser.add_argument("--steps-per-update", type=int, default=40)
arg_parser.add_argument("--num-bptt-chunks", type=int, default=8)

arg_parser.add_argument("--lr", type=float, default=1e-4)
arg_parser.add_argument("--gamma", type=float, default=0.998)
arg_parser.add_argument("--entropy-loss-coef", type=float, default=0.01)
arg_parser.add_argument("--value-loss-coef", type=float, default=0.5)
arg_parser.add_argument("--clip-value-loss", action="store_true")

arg_parser.add_argument("--num-channels", type=int, default=256)
arg_parser.add_argument("--separate-value", action="store_true")
arg_parser.add_argument("--fp16", action="store_true")

arg_parser.add_argument("--gpu-sim", action="store_true")
arg_parser.add_argument("--profile-report", action="store_true")
arg_parser.add_argument(
    "--enable-value-normalization",
    action="store_true",
    help="Enable value normalization during training",
)
arg_parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
arg_parser.add_argument("--level-file", type=str, help="Path to compiled .lvl level file")
arg_parser.add_argument(
    "--tag", action="append", dest="tags", help="Add tags to wandb run (can be used multiple times)"
)
arg_parser.add_argument(
    "--record",
    type=str,
    help="Enable recording and save to specified filepath (e.g., training_run.bin)",
)

args = arg_parser.parse_args()

torch.manual_seed(args.seed)

# Load custom level if provided
compiled_levels = None
level_name = "default_16x16_room"
if args.level_file:
    compiled_levels = load_compiled_levels(args.level_file)

    if len(compiled_levels) == 1:
        # Single level file - extract level name from the level
        level_name = compiled_levels[0].level_name.decode("utf-8", errors="ignore").strip("\x00")
        if not level_name:
            level_name = Path(args.level_file).stem
        print(f"Loaded custom level: {level_name} from {args.level_file}")
    else:
        # Multi-level file - use filename for display
        level_name = Path(args.level_file).stem
        print(f"Loaded multi-level file: {len(compiled_levels)} levels from {args.level_file}")
        print(f"Using curriculum name: {level_name}")

# Setup training environment with 128-beam lidar sensor (distance values only)
exec_mode = ExecMode.CUDA if args.gpu_sim else ExecMode.CPU

sim_interface = setup_lidar_training_environment(
    num_worlds=args.num_worlds,
    exec_mode=exec_mode,
    gpu_id=args.gpu_id,
    rand_seed=args.seed,
    compiled_levels=compiled_levels,
)

ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else None

# Prepare training configuration for wandb logging
training_config = {
    "num_worlds": args.num_worlds,
    "num_updates": args.num_updates,
    "steps_per_update": args.steps_per_update,
    "num_bptt_chunks": args.num_bptt_chunks,
    "learning_rate": args.lr,
    "gamma": args.gamma,
    "entropy_loss_coef": args.entropy_loss_coef,
    "value_loss_coef": args.value_loss_coef,
    "clip_value_loss": args.clip_value_loss,
    "num_channels": args.num_channels,
    "separate_value": args.separate_value,
    "fp16": args.fp16,
    "gpu_sim": args.gpu_sim,
    "value_normalization_enabled": args.enable_value_normalization,
    "exec_mode": "CUDA" if args.gpu_sim else "CPU",
    "level_name": level_name,  # Dynamic level name
    "level_file": args.level_file if args.level_file else None,
    "sensor_type": "lidar_128_beam",
    "random_seed": args.seed,
    "recording_enabled": bool(args.record),
    "recording_filepath": args.record
    if args.record and args.record != "wandb"
    else "train.rec"
    if args.record
    else None,
}

learning_cb = LearningCallback(
    ckpt_dir, args.profile_report, training_config, args.level_file, args.tags
)

# Start recording if requested
if args.record:
    if args.record == "wandb" and learning_cb.use_wandb:
        # Save recording to checkpoint directory for wandb runs
        record_path = learning_cb.ckpt_dir / "train.rec"
        sim_interface.manager.start_recording(str(record_path))
        print(f"✓ Started recording to wandb checkpoint directory: {record_path}")
    else:
        # Use the provided path directly
        sim_interface.manager.start_recording(args.record)
        print(f"✓ Started recording to: {args.record}")

if torch.cuda.is_available():
    dev = torch.device(f"cuda:{args.gpu_id}")
else:
    dev = torch.device("cpu")


# Setup observations from [progress, compass, lidar] tensor list
obs, num_obs_features = setup_obs(sim_interface.obs)
policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

if args.restore:
    restore_ckpt = ckpt_dir / f"{args.restore}.pth"
else:
    restore_ckpt = None

# Use the sim_interface directly - it already has everything configured!
try:
    train(
        dev,
        sim_interface,
        TrainConfig(
            num_updates=args.num_updates,
            steps_per_update=args.steps_per_update,
            num_bptt_chunks=args.num_bptt_chunks,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=0.95,
            ppo=PPOConfig(
                num_mini_batches=1,
                clip_coef=0.2,
                value_loss_coef=args.value_loss_coef,
                entropy_coef=args.entropy_loss_coef,
                max_grad_norm=0.5,
                num_epochs=2,
                clip_value_loss=args.clip_value_loss,
            ),
            normalize_values=args.enable_value_normalization,
            value_normalizer_decay=0.999,
            mixed_precision=args.fp16,
        ),
        policy,
        learning_cb,
        restore_ckpt,
    )
finally:
    # Stop recording if active
    if args.record and sim_interface.manager.is_recording():
        try:
            sim_interface.manager.stop_recording()
            if args.record == "wandb" and learning_cb.use_wandb:
                record_path = learning_cb.ckpt_dir / "train.rec"
                print(f"✓ Recording saved to wandb checkpoint directory: {record_path}")
            else:
                print(f"✓ Recording saved to: {args.record}")
        except Exception as e:
            print(f"⚠ Warning: Failed to stop recording: {e}")

    # Clean up wandb
    learning_cb.finish()
