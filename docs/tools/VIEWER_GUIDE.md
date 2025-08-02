# Viewer Guide

## Overview

The viewer provides an interactive 3D visualization of the Madrona Escape Room environment. It supports real-time simulation viewing, agent control, trajectory tracking, action recording, and replay functionality.

## Building the Viewer

The viewer is built automatically with the project:

```bash
mkdir build
/opt/cmake/bin/cmake -B build
make -C build -j$(nproc)
```

This creates the `viewer` executable in the `build/` directory.

## Command Line Usage

### Basic Syntax

```bash
./build/viewer [num_worlds] [--cpu|--cuda] [options]
```

### Arguments

1. **num_worlds** (optional): Number of parallel worlds to simulate
   - Default: 1
   - Example: `./build/viewer 4` runs 4 worlds

2. **Execution mode** (optional): 
   - `--cpu`: Use CPU execution (default)
   - `--cuda`: Use GPU execution

### Options

- **`--track <world_id> <agent_id>`**: Enable trajectory tracking for a specific agent
  - Prints agent position, rotation, and progress at each step
  - `world_id`: Zero-based world index (0 to num_worlds-1)
  - `agent_id`: Zero-based agent index (0 to numAgents-1)
  - If indices omitted, defaults to world 0, agent 0

- **`--record <path>`**: Record actions to file
  - Creates a binary file with agent actions
  - Press SPACE during simulation to start recording
  - Useful for creating demonstrations or debugging

- **`<replay_file>`**: Path to action file for replay
  - Replays previously recorded actions
  - Must use same number of worlds as when recorded

### Examples

```bash
# Basic usage - 1 world on CPU
./build/viewer

# Multiple worlds on CPU
./build/viewer 4 --cpu

# GPU execution with 8 worlds
./build/viewer 8 --cuda

# Track specific agent's trajectory
./build/viewer 1 --cuda --track 0 0        # Track agent 0 in world 0
./build/viewer 4 --cpu --track 2 1         # Track agent 1 in world 2

# Record actions to file
./build/viewer 2 --cpu --record demo.bin   # Press SPACE to start recording

# Replay recorded actions
./build/viewer 2 --cpu demo.bin            # Must match world count from recording
```

## Keyboard Controls

### Camera Controls
- **Mouse**: Look around (when controlling an agent)
- **Number Keys (1-9)**: Switch between different world views
- **0**: Overview camera showing all worlds

### Agent Control (when in agent view)
- **W/A/S/D**: Move forward/left/backward/right
- **Q/E**: Rotate left/right
- **Shift**: Move faster (works with WASD and Q/E)
- **G**: Grab/release objects (if grab system enabled)

### Simulation Controls
- **R**: Reset current world
- **T**: Toggle trajectory tracking for current world
- **SPACE**: Start recording (when using `--record` flag)
- **TAB**: Switch between worlds (in multi-world mode)
- **ESC**: Exit viewer (stops recording and saves if recording)

## Features

### Multi-World Visualization
The viewer can display multiple worlds simultaneously in a grid layout. Use number keys to focus on individual worlds or press 0 for an overview.

### Trajectory Tracking
When enabled with `--track` or toggled with T key, the viewer prints detailed agent state at each timestep:
```
Step   0: World 0 Agent 0: pos=(0.00,0.05,0.00) rot=-45.0° progress=0.05
Step   1: World 0 Agent 0: pos=(0.23,0.05,0.15) rot=-42.5° progress=0.20
```

This is useful for:
- Debugging agent behavior
- Understanding navigation patterns
- Analyzing training progress

### Action Recording and Replay

#### Recording Mode
Record agent actions for later analysis:

1. Start viewer with `--record <output_file>`
2. Set up your scene as desired
3. Press SPACE to begin recording (automatically resets world for clean start)
4. Control agents using WASD/QE keys
5. ESC to stop recording and save

Example:
```bash
# Record single world on CPU
./build/viewer 1 --cpu --record my_recording.bin
# Press SPACE to start recording

# Record 4 worlds on GPU
./build/viewer 4 --cuda --record multi_world_recording.bin
# Press SPACE to start recording
```

**Recording Benefits:**
- Delayed start: Set up scene before recording begins
- Clean episodes: World automatically resets when recording starts
- Demonstration capture: Create expert trajectories for imitation learning

#### Replay Mode
Replay previously recorded actions:

```bash
# Replay single world recording
./build/viewer 1 --cpu my_recording.bin

# Replay multi-world recording
./build/viewer 4 --cuda multi_world_recording.bin
```

**Important**: Replay requires the same number of worlds as the original recording.

#### File Format
Recordings are stored as binary files containing sequential int32_t values:
- 3 values per world per timestep: [move_amount, move_angle, rotate]
- Actions are interleaved by world: world0_actions, world1_actions, ..., worldN_actions
- Compatible with training pipeline action format

### Visual Indicators
- **Origin Markers**: XYZ axis gizmos mark agent spawn positions
- **Agent Orientation**: Visual indicators show agent facing direction
- **Progress Tracking**: Trajectory system tracks maximum forward progress

## Performance Considerations

### CPU Mode
- Best for interactive debugging (1-10 worlds)
- Allows direct agent control
- Lower latency for input

### CUDA Mode  
- Optimal for observing many worlds (10+ worlds)
- Higher throughput
- May have slight input lag

### Frame Rate
The viewer targets 60 FPS. If performance drops:
- Reduce number of worlds
- Switch from CUDA to CPU for small world counts
- Close other GPU applications

## Troubleshooting

### Common Issues

**"Unknown option" error**
- Check command spelling (e.g., use `--track`, not `--trajectory`)
- Ensure flags start with exactly two dashes (`--`)

**Replay file errors**
- Verify replay file exists and is readable
- Ensure world count matches recording
- Check file wasn't corrupted during transfer

**Black screen or crashes**
- Verify GPU drivers are up to date
- Try CPU mode instead of CUDA
- Check available GPU memory

**Trajectory not printing**
- Ensure `--track` indices are valid
- World index must be < num_worlds
- Agent index must be < numAgents (currently 2)

### Debug Output

The viewer prints diagnostic information on startup:
```
Viewer Controls:
  R: Reset current world
  T: Toggle trajectory tracking for current world
```

When recording is enabled:
```
Recording mode enabled. Press SPACE to start recording to: demo.bin
```

## Integration with Training

The viewer uses the same `Manager` class as training scripts, ensuring consistent behavior. You can:

1. Record expert demonstrations for imitation learning
2. Visualize trained policy behavior
3. Debug environment dynamics
4. Create test scenarios

Example workflow:
```bash
# Train a policy
uv run python scripts/train.py --num-worlds 1024 --num-updates 1000

# Visualize trained behavior
uv run python scripts/infer.py --ckpt-path checkpoints/1000.pth --action-dump-path actions.bin

# Replay in viewer
./build/viewer 1 --cpu actions.bin
```

## Advanced Usage

### Custom Scenarios
Create specific test cases by:
1. Recording a sequence of actions
2. Modifying the binary action file
3. Replaying to test edge cases

### Benchmarking
Compare different execution modes:
```bash
# Benchmark CPU rendering
time ./build/viewer 100 --cpu --record bench_cpu.bin

# Benchmark GPU rendering  
time ./build/viewer 100 --cuda --record bench_gpu.bin
```

### Multi-Agent Coordination
Track multiple agents simultaneously by running multiple terminals:
```bash
# Terminal 1
./build/viewer 4 --cpu --track 0 0 > agent0.log

# Terminal 2  
./build/viewer 4 --cpu --track 0 1 > agent1.log
```