# Headless Mode Quick Reference

## Overview
Run Madrona Escape Room simulation without graphics for benchmarking, testing, or server deployment.

## Usage
```bash
./build/headless [OPTIONS]
```

### Required Options
- `--mode MODE`: Execution mode (`CPU` or `CUDA`)
- `--num-worlds N`: Number of parallel worlds (e.g., 1, 1024, 8192)
- `--num-steps N`: Simulation steps to run (e.g., 100, 1000, 10000)

### Options
- `--rand-actions`: Generate random actions (for benchmarking)
- `--track`: Enable trajectory tracking (default: world 0, agent 0)
- `--track-world N`: Specify world to track (default: 0)
- `--track-agent N`: Specify agent to track (default: 0)
- `--track-file FILE`: Save trajectory to file

## Examples

```bash
# Basic CPU run
./build/headless --mode cpu --num-worlds 1 --num-steps 1000

# GPU benchmark with random actions
./build/headless --mode cuda --num-worlds 8192 --num-steps 1000 --rand-actions

# Track agent 0 in world 5
./build/headless --mode cpu --num-worlds 100 --num-steps 1000 --track --track-world 5

# Track agent 1 in world 5
./build/headless --mode cpu --num-worlds 100 --num-steps 1000 --track-world 5 --track-agent 1
```

## Output
- **FPS**: (worlds × steps) / time - higher is better
- **Trajectory**: Position data if tracking enabled

## Key Settings
- No auto-reset between episodes
- Fixed random seed (5)
- No rendering overhead

## Performance Tips
- **CPU**: Use world count ≤ CPU cores
- **GPU**: Use 1000+ worlds for efficiency
- Check `nvidia-smi` for GPU memory