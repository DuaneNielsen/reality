#!/usr/bin/env python3
"""Test the instrumented wrapper to see timing breakdown."""

import torch
import os
import argparse
from madrona_escape_room_learn import MadronaEscapeRoomEnv
from step_timer import FPSCounter

os.environ['MADRONA_DISABLE_CUDA_HEAP_SIZE'] = '1'


def main():
    parser = argparse.ArgumentParser(description="Test instrumented TorchRL wrapper timing")
    parser.add_argument('--gpu-id', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')
    parser.add_argument('--num-worlds', type=int, default=1024,
                       help='Number of parallel worlds')
    parser.add_argument('--num-steps', type=int, default=100,
                       help='Number of steps to collect timing statistics')
    
    args = parser.parse_args()
    
    print("Testing instrumented TorchRL wrapper")
    print("="*60)
    print(f"Configuration:")
    print(f"  Device: {'CUDA:' + str(args.gpu_id) if args.gpu_id >= 0 else 'CPU'}")
    print(f"  Num worlds: {args.num_worlds}")
    print(f"  Steps: {args.num_steps}")
    
    # Create environment
    env = MadronaEscapeRoomEnv(
        num_worlds=args.num_worlds,
        gpu_id=args.gpu_id,
        rand_seed=5,
        auto_reset=False,  # Avoid AutoResetEnv wrapper overhead
    )
    
    # Create FPS counter
    fps_counter = FPSCounter(args.num_worlds)
    
    # Reset
    print("\nRunning benchmark...")
    td = env.reset()
    
    # Run timing steps
    fps_counter.start()
    for i in range(args.num_steps):
        action = env.action_spec.rand()
        td["action"] = action
        
        # Start timing TorchRL entry overhead
        env._runtime_timer.start("torchrl_entry_overhead")
        td = env.step(td)
        # Stop timing TorchRL exit overhead (it was started inside _step)
        env._runtime_timer.stop("torchrl_exit_overhead")
        
        fps_counter.frame()
    
    # Print timing reports
    fps_counter.report()
    env._init_timer.report("Initialization Timing")
    env._runtime_timer.report("Runtime Timing")
    
    env.close()


if __name__ == "__main__":
    main()