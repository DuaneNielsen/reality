# Viewer Recording Feature

The viewer now supports recording action trajectories during manual gameplay, which can be replayed later for analysis or demonstrations.

## Usage

### Recording Mode
Start the viewer in recording mode to capture actions:
```bash
./build/viewer <num_worlds> <exec_mode> --record <output_file>
```

Press **SPACE** to begin recording after the viewer starts. This allows you to:
- Set up your scene before recording
- Start recording at the exact moment you want
- Get a clean episode start (automatically resets world when recording begins)

Example:
```bash
# Record single world on CPU
./build/viewer 1 --cpu --record my_recording.bin
# Then press SPACE to start recording

# Record 4 worlds on GPU
./build/viewer 4 --cuda --record multi_world_recording.bin
# Then press SPACE to start recording
```

### Replay Mode
Replay previously recorded actions:
```bash
./build/viewer <num_worlds> <exec_mode> <recording_file>
```

Example:
```bash
# Replay single world recording
./build/viewer 1 --cpu my_recording.bin

# Replay multi-world recording
./build/viewer 4 --cuda multi_world_recording.bin
```

## Features

- **Binary Format**: Recordings use the same efficient binary format as inference dumps
- **Multi-World Support**: Records actions for all worlds simultaneously
- **Default Actions**: Non-controlled worlds receive default actions (no movement, no rotation)
- **Progress Display**: Shows recording progress every 100 frames
- **Auto-Reset**: Automatically enables episode auto-reset during recording

## File Format

Recordings are stored as binary files containing sequential int32_t values:
- 3 values per world per timestep: [move_amount, move_angle, rotate]
- Actions are interleaved by world: world0_actions, world1_actions, ..., worldN_actions

## Controls During Recording

- **SPACE**: Start recording (when in recording mode but not yet recording)
- **WASD**: Move agent (W=forward, S=back, A=left, D=right)
- **Q/E**: Rotate agent (Q=left, E=right, hold Shift for faster rotation)
- **Shift+WASD**: Fast movement
- **TAB**: Switch between worlds (in multi-world mode)
- **R**: Reset current world
- **ESC**: Stop recording and save

## Testing

Run the test script to verify recording functionality:
```bash
./test_recording.sh
```

This will test:
1. Single world recording and replay
2. Multi-world recording and replay
3. File creation and size verification