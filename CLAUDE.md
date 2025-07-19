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
/opt/cmake/bin/cmake ..
make -j$(nproc)
cd ..

# Install Python package
pip install -e .

# Alternative headless executable (no visualization)
./build/headless
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
- `src/consts.hpp`: Game parameters and constants

### System Execution Order
The task graph defines precise system dependencies:
1. **Input**: Action processing
2. **Physics Pipeline**: 
   - `movementSystem` → `broadphase_setup` → `grabSystem` → `physicsSystem` → `agentZeroVel`
3. **Game Logic Pipeline**:
   - `doorOpenSystem` → `rewardSystem` → `bonusRewardSystem` → `doneSystem` → `resetSystem`
4. **Output**: Observation collection

### Data Flow
1. **Actions** (Python) → `Action` components
2. **Forces** → Physics simulation → Position updates
3. **Game state** → Logic systems → Rewards/Done flags
4. **World state** → Observation systems → PyTorch tensors

### Python Integration
- Manager class exports PyTorch tensors directly mapped to ECS components
- Zero-copy data transfer between simulation and training
- Tensors shaped as (num_worlds × agents_per_world, ...)
- Components marked with `exportColumn` become accessible from Python

### GPU Optimization Patterns
- **Warp-level dispatch**: Lidar uses 32 threads per agent
- **Entity sorting**: Entities grouped by world ID for coalesced access
- **Conditional compilation**: `#ifdef MADRONA_GPU_MODE` for GPU-specific code
- **Memory layout**: Components packed for cache efficiency

### Observation Space Details
- **Egocentric coordinates**: All positions relative to agent
- **Normalization**: Distances by world size, angles by π
- **Fixed arrays**: Padded to `maxEntitiesPerRoom`
- **Lidar**: 30 samples in circle around agent

### Modifying the Simulator
1. **Adding new observations**: 
   - Add component to `src/types.hpp`
   - Update `Agent` archetype
   - Populate in `collectObservationSystem`
   - Export in `src/mgr.cpp` with `exportColumn`

2. **Changing game logic**:
   - Modify systems in `src/sim.cpp`
   - Update level generation in `src/level_gen.cpp`
   - Adjust constants in `src/consts.hpp`

3. **Adjusting physics**:
   - Tune parameters in movement/physics systems
   - Modify collision responses
   - Note: Velocities zeroed each frame for controllability

4. **Adding new entity types**:
   - Define archetype in `src/types.hpp`
   - Add to `SimArchetypes` enum
   - Update observation collection if visible to agents

### Performance Considerations
- Batch simulation across thousands of worlds simultaneously
- GPU backend provides massive speedup for large batches (8192+ worlds)
- Use `--profile-report` flag to identify bottlenecks
- Adjust `--num-worlds` based on available GPU memory
- Fixed timestep: 0.04s with 4 physics substeps
- Component access patterns critical for cache performance

## Initialization Process

### Manager Creation Flow
The Manager constructor performs crucial initialization:
```cpp
Manager::Manager(const Config &cfg) {
    // 1. Initialize implementation (CPU/CUDA)
    // 2. Force reset all worlds
    // 3. Execute one step to populate observations
}
```

### Execution Mode Initialization

#### CPU Mode:
1. Creates `PhysicsLoader` for CPU execution
2. Loads collision meshes for all objects
3. Initializes `ThreadPoolExecutor` with:
   - Auto-detected worker threads (0 = num CPU cores)
   - Exported component memory allocation
   - Per-world initialization data
4. Maps exported buffer pointers

#### CUDA Mode:
1. Initializes CUDA context for specified GPU
2. Creates GPU-based `PhysicsLoader`
3. Initializes `MWCudaExecutor` with:
   - JIT compilation of GPU kernels
   - Device memory allocation
   - CUDA graph optimization
4. Maps device pointers to exported buffers

### Asset Loading

#### Physics Assets:
- **Collision Meshes**: Cube, Wall, Door, Agent, Button
- **Mass Properties**:
  - Cube: 0.075 inverse mass (movable)
  - Walls/Doors: 0 inverse mass (static)
  - Agent: 1.0 inverse mass, Z-axis rotation only
- **Friction**: μs=0.5, μd=0.5-0.75 for most objects

#### Render Assets:
- **Meshes**: Visual representations with material assignments
- **Materials**: Orange cubes, gray walls, red doors, white agents, yellow buttons
- **Textures**: Grid pattern, smile emoji
- **Lighting**: Single directional light

### Memory Layout
1. **World Data**: Array of `Sim` instances
2. **Exported Tensors**:
   - Actions: `[numWorlds × numAgents × 4]`
   - Rewards: `[numWorlds × numAgents × 1]`
   - Multiple observation tensors
3. **Physics Data**: Collision geometry, rigid body metadata
4. **Render Buffers**: GPU memory for meshes, textures, outputs

### Initialization Sequence
1. **Manager Construction**:
   - Select CPU/CUDA implementation
   - Load all assets
   - Create execution backend
   
2. **Per-World Sim Initialization**:
   - Register ECS components/archetypes
   - Create persistent entities (floor, walls, agents)
   - Generate initial level
   - Setup task graph
   
3. **First Step Execution**:
   - Trigger reset for all worlds
   - Run one simulation step
   - Populate initial observations
   - Ensure valid starting state

### Key Configuration Parameters
- `execMode`: CPU or CUDA execution
- `gpuID`: Target GPU device
- `numWorlds`: Parallel simulation count
- `randSeed`: RNG initialization
- `autoReset`: Automatic episode restart
- `enableBatchRenderer`: GPU rendering toggle

### Thread/GPU Parallelism
- **CPU**: Thread pool with automatic core detection
- **CUDA**: One thread block per world, warp-level optimizations
- Task graph ensures correct system execution order
- Zero-copy memory mapping for Python integration
