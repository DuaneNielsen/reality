import argparse
import warnings

import numpy as np
import torch
from madrona_escape_room_learn import LearningState
from madrona_escape_room_learn.sim_interface_adapter import setup_lidar_training_environment
from policy import make_policy, setup_obs

import madrona_escape_room

warnings.filterwarnings("error")


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--gpu-id", type=int, default=0)
arg_parser.add_argument("--ckpt-path", type=str, required=True)
arg_parser.add_argument("--action-dump-path", type=str)
arg_parser.add_argument("--recording-path", type=str, help="Path to save recording file")
arg_parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
arg_parser.add_argument("--recording-seed", type=int, default=5, help="Seed for recording")

arg_parser.add_argument("--num-worlds", type=int, required=True)
arg_parser.add_argument("--num-steps", type=int, required=True)

arg_parser.add_argument("--num-channels", type=int, default=256)
arg_parser.add_argument("--separate-value", action="store_true")
arg_parser.add_argument("--fp16", action="store_true")

arg_parser.add_argument("--gpu-sim", action="store_true")

args = arg_parser.parse_args()

torch.manual_seed(args.seed)

exec_mode = madrona_escape_room.ExecMode.CUDA if args.gpu_sim else madrona_escape_room.ExecMode.CPU

sim_interface = setup_lidar_training_environment(
    num_worlds=args.num_worlds, exec_mode=exec_mode, gpu_id=args.gpu_id, rand_seed=args.seed
)

obs, num_obs_features = setup_obs(sim_interface.obs)
policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

weights = LearningState.load_policy_weights(args.ckpt_path)
policy.load_state_dict(weights)

actions = sim_interface.actions
dones = sim_interface.dones
rewards = sim_interface.rewards

# Flatten N, A, ... tensors to N * A, ... for rewards and dones (still per-agent)
# Actions are now per-world, so no flattening needed
dones = dones.view(-1, *dones.shape[2:])
rewards = rewards.view(-1, *rewards.shape[2:])

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

# Start recording if recording path is provided
if args.recording_path:
    try:
        sim_interface.manager.start_recording(args.recording_path, args.recording_seed)
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

    print()
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
    sim_interface.step()
    print("Rewards:\n", rewards)

# Stop recording if it was started
if args.recording_path:
    try:
        sim_interface.manager.stop_recording()
        print(f"Recording completed: {args.recording_path}")
    except Exception as e:
        print(f"Failed to stop recording: {e}")

if action_log:
    action_log.close()
