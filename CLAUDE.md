# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Madrona Escape Room - a high-performance 3D multi-agent reinforcement learning environment built on the Madrona Engine. It implements a cooperative puzzle game where agents must navigate rooms by stepping on buttons or pushing blocks to open doors.

**Tech Stack:**
- C++ (core simulation using Entity Component System pattern)
- Python (PyTorch-based PPO training)
- CMake build system
- CUDA (optional GPU acceleration)

## Essential Commands

### Building the Project
```bash
# Initial setup (from repo root)
mkdir build
cd build
cmake ..
make -j$(nproc)
cd ..

# Install Python package
pip install -e .
```

### Running the Simulation
```bash
# Interactive viewer
./build/viewer

# Benchmark performance
python scripts/sim_bench.py --num-worlds 1024 --num-steps 1000 --gpu-id 0
```

### Training
```bash
# Basic CPU training
python scripts/train.py --num-worlds 1024 --num-updates 100 --ckpt-dir build/checkpoints

# Full GPU training with optimizations
python scripts/train.py --num-worlds 8192 --num-updates 5000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints/
```

### Inference
```bash
# Run trained policy
python scripts/infer.py --num-worlds 1 --num-steps 1000 --fp16 --ckpt-path build/checkpoints/5000.pth --action-dump-path build/dumped_actions

# Replay in viewer
./build/viewer 1 --cpu build/dumped_actions
```

### Development
```bash
# Rebuild after C++ changes
cd build && make -j$(nproc) && cd ..

# No linting/testing commands provided - verify with training scripts
```

## Architecture

### Core Concepts
The simulator uses Madrona's Entity Component System (ECS) pattern:
- **Components**: Data containers (Position, Action, Reward, etc.)
- **Archetypes**: Entity templates grouping components (Agent, Wall, Button, etc.)
- **Systems**: Functions that operate on components (movementSystem, physicsSystem, etc.)
- **Task Graph**: Defines execution order of systems each simulation step

### Key Files
- `src/types.hpp`: All ECS component and archetype definitions
- `src/sim.cpp/hpp`: Core simulation logic and task graph setup
- `src/level_gen.cpp`: Procedural level generation
- `src/mgr.cpp/hpp`: Manager class handling Python/PyTorch integration
- `scripts/train.py`: PPO training entry point
- `scripts/policy.py`: Neural network policy definitions

### Simulation Flow
1. **Level Generation**: `generateWorld()` creates random puzzle layouts
2. **Per-Step Execution** (defined in `setupTasks()`):
   - `movementSystem`: Converts discrete actions to physics forces
   - `physicsSystem`: Rigid body simulation
   - `gameLogicSystem`: Button/door interactions
   - `collectObservationSystem`: Builds agent observations
   - `computeRewardSystem`: Calculates rewards based on progress

### Python Integration
- Manager class exports PyTorch tensors directly mapped to ECS components
- Zero-copy data transfer between simulation and training
- Supports both CPU and GPU backends

### Modifying the Simulator
1. **Adding new observations**: 
   - Add component to `src/types.hpp`
   - Update `Agent` archetype
   - Populate in `collectObservationSystem`
   - Export in `src/mgr.cpp`

2. **Changing game logic**:
   - Modify systems in `src/sim.cpp`
   - Update level generation in `src/level_gen.cpp`

3. **Adjusting physics**:
   - Tune parameters in movement/physics systems
   - Modify collision responses

### Performance Considerations
- Batch simulation across thousands of worlds simultaneously
- GPU backend provides massive speedup for large batches
- Use `--profile-report` flag to identify bottlenecks
- Adjust `--num-worlds` based on available GPU memory