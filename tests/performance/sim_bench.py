import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from pyinstrument import Profiler
from pyinstrument.renderers import JSONRenderer
from step_timer import FPSCounter

import madrona_escape_room
from madrona_escape_room.generated_constants import ExecMode

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--num-worlds", type=int, default=1024)
arg_parser.add_argument("--num-steps", type=int, default=1000)
arg_parser.add_argument("--profile-renderer", action="store_true")
arg_parser.add_argument("--gpu-id", type=int, default=-1, help="GPU device ID (-1 for CPU)")
arg_parser.add_argument(
    "--check-baseline", action="store_true", help="Check against performance baselines"
)
arg_parser.add_argument("--save-profile", action="store_true", help="Save profiling data to files")
arg_parser.add_argument("--output-dir", type=str, help="Directory to save results")

args = arg_parser.parse_args()

# Determine execution mode
if args.gpu_id >= 0:
    exec_mode = ExecMode.CUDA
else:
    exec_mode = ExecMode.CPU

# Create simulator
sim = madrona_escape_room.SimManager(
    exec_mode=exec_mode,
    gpu_id=args.gpu_id,
    num_worlds=args.num_worlds,
    auto_reset=True,
    rand_seed=5,
    enable_batch_renderer=args.profile_renderer,
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
print("\nBenchmark Configuration:")
print(f"{'=' * 60}")
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
    steps_taken = sim.steps_taken_tensor().to_torch()

    fps_counter.frame()

profiler.stop()
end_time = time.time()

# Calculate FPS from frame counter
total_frames = len(fps_counter._frame_times)
if fps_counter._last_frame_time and fps_counter._start_time:
    total_time = fps_counter._last_frame_time - fps_counter._start_time
    fps = (total_frames * args.num_worlds) / total_time
else:
    # Fallback calculation
    total_time = time.time() - fps_counter._start_time if fps_counter._start_time else 0
    fps = (total_frames * args.num_worlds) / total_time if total_time > 0 else 0

# Print results
print("\nPerformance Summary:")
print("=" * 60)
fps_counter.report()

# Baseline checking
exit_code = 0
baseline_status = None
if args.check_baseline:
    # Load baselines
    baseline_file = Path(__file__).parent / "performance_baselines.json"
    if baseline_file.exists():
        with open(baseline_file, "r") as f:
            baselines = json.load(f)

        # Determine which baseline to use
        if args.gpu_id >= 0 and args.num_worlds == 8192:
            baseline_key = "gpu_8192"
        elif args.gpu_id < 0 and args.num_worlds == 1024:
            baseline_key = "cpu_1024"
        else:
            baseline_key = None
            device = "GPU" if args.gpu_id >= 0 else "CPU"
            print(f"\n⚠ No baseline for configuration: {device} with {args.num_worlds} worlds")

        if baseline_key and baseline_key in baselines:
            baseline = baselines[baseline_key]
            min_fps = baseline["min_fps"]
            warn_fps = baseline["warn_fps"]

            print("\nBaseline Check:")
            print("=" * 60)
            if fps >= warn_fps:
                print(f"✓ PASS: {fps:,.0f} FPS >= {warn_fps:,} FPS (warning threshold)")
                baseline_status = "PASS"
            elif fps >= min_fps:
                print(f"⚠ WARN: {fps:,.0f} FPS < {warn_fps:,} FPS (warning threshold)")
                print(f"  but >= {min_fps:,} FPS (minimum threshold)")
                baseline_status = "WARN"
                exit_code = 2
            else:
                print(f"✗ FAIL: {fps:,.0f} FPS < {min_fps:,} FPS (minimum threshold)")
                baseline_status = "FAIL"
                exit_code = 1

            print(f"\nDescription: {baseline['description']}")
    else:
        print(f"\n⚠ Baseline file not found: {baseline_file}")

print("\nProfile Results:")
print("=" * 60)
print(profiler.output_text(unicode=True, show_all=True))

# Save profiling data if requested
if args.save_profile or args.output_dir:
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        device = "gpu" if args.gpu_id >= 0 else "cpu"
        output_dir = Path(__file__).parent / f"perf_results/{timestamp}/{device}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save HTML profile
    html_file = output_dir / "profile.html"
    with open(html_file, "w") as f:
        f.write(profiler.output_html())
    print(f"\nHTML profile saved to: {html_file}")

    # Save JSON profile
    json_file = output_dir / "profile.json"
    with open(json_file, "w") as f:
        f.write(profiler.output(JSONRenderer()))
    print(f"JSON profile saved to: {json_file}")

    # Save results summary
    results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "num_worlds": args.num_worlds,
            "num_steps": args.num_steps,
            "device": "GPU" if args.gpu_id >= 0 else "CPU",
            "gpu_id": args.gpu_id,
            "renderer": args.profile_renderer,
        },
        "performance": {
            "fps": fps,
            "total_frames": total_frames * args.num_worlds,
            "total_time": total_time,
            "frame_time_ms": (total_time / total_frames) * 1000 if total_frames > 0 else 0,
        },
    }

    if args.check_baseline and baseline_status:
        results["baseline_check"] = {
            "status": baseline_status,
            "min_fps": min_fps if "min_fps" in locals() else None,
            "warn_fps": warn_fps if "warn_fps" in locals() else None,
        }

    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results summary saved to: {results_file}")
else:
    # Default HTML output for backward compatibility
    html_file = f"/tmp/sim_bench_profile_{'gpu' if args.gpu_id >= 0 else 'cpu'}.html"
    with open(html_file, "w") as f:
        f.write(profiler.output_html())
    print(f"\nDetailed HTML profile saved to: {html_file}")

# Exit with appropriate code
sys.exit(exit_code)
