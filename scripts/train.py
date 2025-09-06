import argparse
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

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

torch.manual_seed(0)


class LearningCallback:
    def __init__(self, ckpt_dir, profile_report, training_config=None):
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = profile_report

        # Initialize wandb
        self.use_wandb = False
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project="madrona-escape-room",
                    config=training_config or {},
                    tags=["lidar-training", "default_16x16_room"],
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

            vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
            vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

        # Log to wandb
        if self.use_wandb:
            wandb.log(
                {
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
                    # Returns
                    "returns/mean": ppo.returns_mean,
                    "returns/stddev": ppo.returns_stddev,
                    # Value normalizer
                    "value_normalizer/mu": vnorm_mu,
                    "value_normalizer/sigma": vnorm_sigma,
                    # Performance metrics
                    "performance/fps": fps,
                    "performance/avg_fps": self.mean_fps,
                    "performance/update_time": update_time,
                    "performance/memory_reserved_gb": reserved_gb,
                    "performance/memory_current_gb": current_gb,
                },
                step=update_id,
            )

        # Keep original console output
        print(f"\nUpdate: {update_id}")
        print(
            f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, "
            f"V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}"
        )
        print()
        print(
            f"    Rewards          => Avg: {reward_mean: .3e}, "
            f"Min: {reward_min: .3e}, Max: {reward_max: .3e}"
        )
        print(
            f"    Values           => Avg: {value_mean: .3e}, "
            f"Min: {value_min: .3e}, Max: {value_max: .3e}"
        )
        print(
            f"    Advantages       => Avg: {advantage_mean: .3e}, "
            f"Min: {advantage_min: .3e}, Max: {advantage_max: .3e}"
        )
        print(
            f"    Bootstrap Values => Avg: {bootstrap_value_mean: .3e}, "
            f"Min: {bootstrap_value_min: .3e}, Max: {bootstrap_value_max: .3e}"
        )
        print(f"    Returns          => Avg: {ppo.returns_mean}, σ: {ppo.returns_stddev}")
        print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma:.3e}")

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

args = arg_parser.parse_args()

# Setup training environment with 128-beam lidar sensor (distance values only)
exec_mode = ExecMode.CUDA if args.gpu_sim else ExecMode.CPU

sim_interface = setup_lidar_training_environment(
    num_worlds=args.num_worlds, exec_mode=exec_mode, gpu_id=args.gpu_id, rand_seed=5
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
    "exec_mode": "CUDA" if args.gpu_sim else "CPU",
    "level_name": "default_16x16_room",  # Known level name
    "sensor_type": "lidar_128_beam",
    "random_seed": 5,  # From sim_interface setup
}

learning_cb = LearningCallback(ckpt_dir, args.profile_report, training_config)

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
            value_normalizer_decay=0.999,
            mixed_precision=args.fp16,
        ),
        policy,
        learning_cb,
        restore_ckpt,
    )
finally:
    # Clean up wandb
    learning_cb.finish()
