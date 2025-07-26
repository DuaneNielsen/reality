import torch
import madrona_escape_room
import argparse
from pyinstrument import Profiler
from step_timer import FPSCounter

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, default=1024)
arg_parser.add_argument('--num-steps', type=int, default=1000)
arg_parser.add_argument('--profile-renderer', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=-1,
                       help='GPU device ID (-1 for CPU)')

args = arg_parser.parse_args()

# Determine execution mode
if args.gpu_id >= 0:
    exec_mode = madrona_escape_room.madrona.ExecMode.CUDA
else:
    exec_mode = madrona_escape_room.madrona.ExecMode.CPU

# Create simulator
sim = madrona_escape_room.SimManager(
    exec_mode = exec_mode,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
    rand_seed = 5,
    enable_batch_renderer = args.profile_renderer,
)

# Get action tensor
actions = sim.action_tensor().to_torch()

# Warm up
for _ in range(10):
    actions[..., 0] = torch.randint_like(actions[..., 0], 0, 4)
    actions[..., 1] = torch.randint_like(actions[..., 1], 0, 8)
    actions[..., 2] = torch.randint_like(actions[..., 2], 0, 5)
    sim.step()

# Print configuration
print(f"\nBenchmark Configuration:")
print(f"{'='*60}")
print(f"Worlds: {args.num_worlds}")
print(f"Steps: {args.num_steps}")
print(f"Device: {'CPU' if args.gpu_id < 0 else f'GPU {args.gpu_id}'}")
print(f"Renderer: {'Enabled' if args.profile_renderer else 'Disabled'}")

# Main benchmark loop with profiling
fps_counter = FPSCounter(args.num_worlds)
profiler = Profiler(interval=0.00001)  # 1ms interval for better sampling

print(f"\nProfiling {args.num_steps} steps...")
profiler.start()
fps_counter.start()

for i in range(args.num_steps):
    actions[..., 0] = torch.randint_like(actions[..., 0], 0, 4)
    actions[..., 1] = torch.randint_like(actions[..., 1], 0, 8)
    actions[..., 2] = torch.randint_like(actions[..., 2], 0, 5)

    sim.step()
    
    # Read all observation tensors like a real training loop
    rewards = sim.reward_tensor().to_torch()
    dones = sim.done_tensor().to_torch()
    self_obs = sim.self_observation_tensor().to_torch()
    steps_remaining = sim.steps_remaining_tensor().to_torch()
    
    fps_counter.frame()

profiler.stop()

# Print results
print("\nPerformance Summary:")
print("="*60)
fps_counter.report()

print("\nProfile Results:")
print("="*60)
print(profiler.output_text(unicode=True, show_all=True))

# Save HTML output
html_file = f"/tmp/sim_bench_profile_{'gpu' if args.gpu_id >= 0 else 'cpu'}.html"
with open(html_file, 'w') as f:
    f.write(profiler.output_html())
print(f"\nDetailed HTML profile saved to: {html_file}")
