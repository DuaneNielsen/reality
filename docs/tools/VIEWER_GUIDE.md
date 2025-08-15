# Viewer Guide

## Overview

The viewer provides an interactive 3D visualization of the Madrona Escape Room environment. It supports real-time simulation viewing, agent control, trajectory tracking, action recording with embedded levels, and replay functionality.

## Building the Viewer

The viewer is built automatically with the project:

```bash
mkdir build
cmake -B build
make -C build -j8
```

This creates the `viewer` executable in the `build/` directory.

## Command Line Usage

### Basic Syntax

```bash
./build/viewer --load <level.lvl> [options]          # Load and run level
./build/viewer --load <level.lvl> --record <rec.rec> # Load level and record
./build/viewer --replay <recording.rec> [options]    # Replay recording
```

### Required Arguments

The viewer requires either a level file or a replay file:

- **`--load <file.lvl>`**: Load a binary level file
- **`--replay <file.rec>`**: Replay a recording (contains embedded level)

You cannot specify both `--replay` and `--record` together.

### Options

#### Execution Mode
- **`--cuda <device_id>`**: Use GPU execution on specified device (default: CPU)
- **`--num-worlds <n>`, `-n <n>`**: Number of parallel worlds (default: 1)

#### Tracking & Logging
- **`--track`, `-t`**: Enable trajectory tracking (default: world 0, agent 0)
- **`--track-world <n>`**: Specify world to track (default: 0)
- **`--track-agent <n>`**: Specify agent to track (default: 0)
- **`--track-file <file>`**: Save trajectory to CSV file

#### Recording & Replay
- **`--record <file.rec>`, `-r <file.rec>`**: Record session to file
  - Recording starts PAUSED - press SPACE to begin
  - Level data is embedded in the recording
- **`--replay <file.rec>`**: Replay a recording
  - Automatically extracts embedded level
  - Must use same number of worlds as recording

#### Other Options
- **`--seed <value>`, `-s <value>`**: Set random seed (default: 5)
- **`--hide-menu`**: Hide ImGui menu (useful for screenshots)
- **`--help`, `-h`**: Print usage information

### File Formats

- **`.lvl` files**: Binary compiled level data
- **`.rec` files**: Recordings with metadata, embedded level, and action frames

### Examples

```bash
# Load and run a level
./build/viewer --load levels/tutorial.lvl

# Load level with 4 worlds on GPU
./build/viewer --load levels/main.lvl --cuda 0 -n 4

# Track specific agent's trajectory
./build/viewer --load level.lvl --track --track-world 2 --track-agent 1

# Track and save trajectory to file
./build/viewer --load level.lvl --track --track-file trajectory.csv

# Record a session (starts paused, press SPACE to begin)
./build/viewer --load level.lvl --record demo.rec -n 2

# Replay a recording (level embedded, worlds auto-matched)
./build/viewer --replay demo.rec

# Set custom seed for deterministic behavior
./build/viewer --load level.lvl --seed 42 -n 4

# Hide menu for clean screenshots
./build/viewer --load level.lvl --hide-menu
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
- **SPACE**: Pause/Resume simulation (or start recording when in record mode)
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
Record agent actions and level data for later analysis:

1. Start viewer with `--load <level.lvl> --record <output.rec>`
2. Simulation starts PAUSED
3. Press SPACE to begin recording (world automatically resets)
4. Control agents using WASD/QE keys
5. Press SPACE to pause/resume during recording
6. ESC to stop recording and save

Example:
```bash
# Record single world
./build/viewer --load level.lvl --record demo.rec
# Press SPACE to start recording

# Record 4 worlds on GPU
./build/viewer --load level.lvl --cuda 0 -n 4 --record multi.rec
# Press SPACE to start recording
```

**Recording Features:**
- **Embedded Levels**: Level data is stored in the recording file
- **Metadata Storage**: Seed and world count preserved for exact replay
- **Pause Control**: Can pause/resume during recording with SPACE
- **Clean Start**: World automatically resets when recording begins

#### Replay Mode
Replay previously recorded sessions:

```bash
# Replay a recording (level automatically extracted)
./build/viewer --replay demo.rec

# Replay with tracking enabled
./build/viewer --replay demo.rec --track
```

**Replay Features:**
- **Automatic Level Loading**: Embedded level is extracted and used
- **World Count Matching**: Automatically uses the recorded world count
- **Seed Preservation**: Uses the same seed as the original recording
- **Pause Control**: Can pause/resume replay with SPACE

#### File Format (.rec files)
Recording files contain:
1. **Metadata Header**: Version, seed, world count, timesteps
2. **Embedded Level**: Complete CompiledLevel structure
3. **Action Frames**: Sequential int32_t values
   - 3 values per agent per timestep: [move_amount, move_angle, rotate]
   - Actions interleaved by world: world0_actions, world1_actions, ..., worldN_actions

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

**"Must provide either --replay OR --load" error**
- The viewer now requires explicit level loading
- Use `--load <level.lvl>` to load a level file
- Use `--replay <recording.rec>` to replay a recording

**"Cannot specify both --replay and --record" error**
- These modes are mutually exclusive
- For recording: use `--load <level.lvl> --record <output.rec>`
- For replay: use `--replay <recording.rec>`

**File extension warnings**
- Use `.lvl` extension for level files
- Use `.rec` extension for recording files
- Viewer will warn if extensions don't match expected format

**Replay world count mismatch**
- Viewer automatically adjusts to match recording's world count
- You'll see a warning but replay will proceed with correct count

**Black screen or crashes**
- Verify GPU drivers are up to date
- Try CPU mode instead of CUDA
- Check available GPU memory
- Ensure level file is valid

**Trajectory not printing**
- Ensure `--track` indices are valid
- World index must be < num_worlds
- Agent index must be < numAgents (currently 2)
- Use `--track-file` to save to CSV if console output is missing

### Debug Output

The viewer prints diagnostic information on startup:
```
Loaded level from levels/main.lvl: 16x16 grid, 256 tiles
Viewer Controls:
  R: Reset current world
  T: Toggle trajectory tracking for current world
```

When recording:
```
Recording mode: Starting PAUSED (press SPACE to start recording)
```

When replaying:
```
Extracted embedded level from demo.rec: 16x16 grid, 256 tiles
Using seed 42 from replay file
```

## Integration with Training

The viewer uses the same `Manager` class as training scripts, ensuring consistent behavior. You can:

1. Record expert demonstrations for imitation learning
2. Visualize trained policy behavior
3. Debug environment dynamics
4. Create test scenarios with specific levels

Example workflow:
```bash
# Create a level using Python
uv run python scripts/create_level.py --output levels/custom.lvl

# Record expert demonstration
./build/viewer --load levels/custom.lvl --record expert_demo.rec -n 1
# Control agent to demonstrate optimal behavior

# Train a policy
uv run python scripts/train.py --num-worlds 1024 --num-updates 1000

# Visualize trained behavior on same level
uv run python scripts/infer.py --level levels/custom.lvl --ckpt-path checkpoints/1000.pth --action-dump-path actions.rec

# Compare expert vs trained policy
./build/viewer --replay expert_demo.rec    # Expert demonstration
./build/viewer --replay actions.rec         # Trained policy
```

## Advanced Usage

### Architecture Overview

The viewer consists of several key components:

1. **ViewerCore**: Handles simulation logic, input processing, and state management
   - Manages recording/replay state machine
   - Processes keyboard input into agent actions
   - Coordinates with Manager for simulation stepping

2. **RecordReplayStateMachine**: Controls recording and replay states
   - Recording starts paused, requiring SPACE to begin
   - Replay starts immediately (not paused)
   - Handles pause/resume during both modes

3. **FrameActionManager**: Batches actions for all worlds per frame
   - Maintains action buffer for all worlds
   - Resets to defaults between frames
   - Provides consistent action format for recording

### Custom Scenarios
Create specific test cases:
```bash
# Create a challenging level
uv run python scripts/create_level.py --difficulty hard --output levels/test.lvl

# Record specific behavior patterns
./build/viewer --load levels/test.lvl --record pattern1.rec --seed 100

# Test with different seeds for variation
./build/viewer --load levels/test.lvl --seed 200
```

### Benchmarking
Compare different execution modes:
```bash
# Benchmark CPU performance
./build/viewer --load level.lvl -n 100 --seed 42 --track-file cpu_bench.csv

# Benchmark GPU performance
./build/viewer --load level.lvl -n 100 --cuda 0 --seed 42 --track-file gpu_bench.csv

# Compare trajectory files to verify determinism
diff cpu_bench.csv gpu_bench.csv
```

### Multi-Agent Coordination
Track different agents in the same world:
```bash
# Save trajectories for both agents
./build/viewer --load level.lvl --track --track-agent 0 --track-file agent0.csv
./build/viewer --load level.lvl --track --track-agent 1 --track-file agent1.csv

# Analyze coordination patterns
uv run python scripts/analyze_trajectories.py agent0.csv agent1.csv
```

### Deterministic Replay
Ensure exact reproducibility:
```bash
# Record with specific seed
./build/viewer --load level.lvl --seed 12345 --record demo.rec

# Replay maintains exact behavior
./build/viewer --replay demo.rec  # Uses seed 12345 from recording
```