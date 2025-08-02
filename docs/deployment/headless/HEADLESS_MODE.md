# Headless Mode Documentation

## Overview

The headless executable provides a way to run the Madrona Escape Room simulation without any graphical output. This is useful for:
- Performance benchmarking
- Training reinforcement learning agents on servers without displays
- Running automated tests
- Generating action sequences for later visualization
- Debugging simulation logic without rendering overhead

the `headless` executable will be in the `build/` directory.

## Command Line Usage

### Basic Syntax

```bash
./build/headless TYPE NUM_WORLDS NUM_STEPS [OPTIONS]
```

### Required Arguments

1. **TYPE**: Execution mode
   - `CPU` - Run simulation on CPU
   - `CUDA` - Run simulation on GPU

2. **NUM_WORLDS**: Number of parallel worlds to simulate
   - Integer value (e.g., 1, 1024, 8192)
   - Higher values provide better throughput but require more memory

3. **NUM_STEPS**: Number of simulation steps to execute
   - Integer value (e.g., 100, 1000, 10000)

### Optional Arguments

- `--rand-actions`: Generate random actions for all agents
  - Useful for performance benchmarking
  - Actions are uniformly sampled from the action space

- `--track-agent WORLD_ID AGENT_ID`: Enable trajectory tracking for a specific agent
  - `WORLD_ID`: Zero-based world index (0 to NUM_WORLDS-1)
  - `AGENT_ID`: Zero-based agent index (0 to numAgents-1)
  - Stores position history for later analysis

## Examples

### Basic CPU Execution
```bash
# Run 1 world for 1000 steps on CPU
./build/headless CPU 1 1000

# Run 64 worlds for 5000 steps on CPU
./build/headless CPU 64 5000
```

### GPU Execution
```bash
# Run 1024 worlds for 1000 steps on GPU
./build/headless CUDA 1024 1000

# Run 8192 worlds for 10000 steps on GPU (large-scale)
./build/headless CUDA 8192 10000
```

### Performance Benchmarking
```bash
# Benchmark with random actions on CPU
./build/headless CPU 1024 1000 --rand-actions

# Benchmark with random actions on GPU
./build/headless CUDA 8192 1000 --rand-actions
```

### Agent Tracking
```bash
# Track agent 0 in world 5 while running 100 worlds
./build/headless CPU 100 1000 --track-agent 5 0

# Track agent 1 in world 0 with random actions
./build/headless CUDA 1024 500 --rand-actions --track-agent 0 1
```

## Output

The headless executable provides minimal output to maximize performance:

1. **FPS (Frames Per Second)**: Total simulation throughput
   - Calculated as: (NUM_WORLDS × NUM_STEPS) / elapsed_time
   - Higher is better for performance benchmarking

2. **Trajectory Data** (if tracking enabled): Stored internally for analysis

## Common Use Cases


### 1. Automated Testing
```bash
# Run deterministic test with fixed seed
./build/headless CPU 1 200  # Uses randSeed=5 internally
```

### 2. Action Recording (Developer Feature)
The code includes a commented-out `saveWorldActions` function that can be enabled to save action sequences to disk for replay in the viewer.


## Troubleshooting

### Low FPS on CPU
- Reduce NUM_WORLDS to match CPU core count
- Ensure no other CPU-intensive processes are running
- Check memory bandwidth limitations

### CUDA Errors
- Verify GPU has sufficient memory for world count
- Check CUDA installation with `nvidia-smi`
- Reduce NUM_WORLDS if out of memory

### Tracking Not Working
- Ensure WORLD_ID < NUM_WORLDS
- Verify AGENT_ID < numAgents (currently 2)
- Check trajectory data is being accessed correctly

## Implementation Details

The headless executable:
1. Creates a `Manager` instance with specified parameters
2. Optionally enables trajectory tracking via `enableAgentTrajectory()`
3. Generates random actions if `--rand-actions` is specified
4. Calls `mgr.step()` for the specified number of steps
5. Calculates and reports total FPS

Key configuration:
- `autoReset`: Disabled (episodes don't auto-restart)
- `enableBatchRenderer`: Disabled (no rendering)
- `randSeed`: Fixed at 5 for reproducibility