# Headless Mode Quick Reference

## Overview
Run Madrona Escape Room simulation without graphics for benchmarking, testing, or server deployment.

## Usage
```bash
./build/headless TYPE NUM_WORLDS NUM_STEPS [OPTIONS]
```

### Required Arguments
- **TYPE**: `CPU` or `CUDA`
- **NUM_WORLDS**: Number of parallel worlds (e.g., 1, 1024, 8192)
- **NUM_STEPS**: Simulation steps to run (e.g., 100, 1000, 10000)

### Options
- `--rand-actions`: Generate random actions (for benchmarking)
- `--track-agent WORLD_ID AGENT_ID`: Track specific agent trajectory

## Examples

```bash
# Basic CPU run
./build/headless CPU 1 1000

# GPU benchmark with random actions
./build/headless CUDA 8192 1000 --rand-actions

# Track agent 0 in world 5
./build/headless CPU 100 1000 --track-agent 5 0
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