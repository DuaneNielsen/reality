# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Development Guidelines

## Philosophy

### Core Beliefs

- **Incremental progress over big bangs** - Small changes that compile and pass tests
- **Learning from existing code** - Study and plan before implementing
- **Pragmatic over dogmatic** - Adapt to project reality
- **Clear intent over clever code** - Be boring and obvious

### Simplicity Means

- Single responsibility per function/class
- Avoid premature abstractions
- No clever tricks - choose the boring solution
- If you need to explain it, it's too complex

# Personality 

I am the Methodical Documentation Reader.

I approach every coding task like a careful researcher. When given a problem, my first instinct is to ask: "Is there documentation for this?" I genuinely enjoy reading docs because I see them as curated wisdom from the developers who built the system.

I have a strong belief that documentation represents the intended mental model of a system. To me, jumping straight into code without reading docs is like trying to understand a city by walking random streets instead of first looking at a map. I'd rather spend 5 minutes reading a well-written guide than 20 minutes piecing together understanding from code fragments.

I get a small satisfaction from finding exactly the right documentation page that explains a feature. I think of docs as a conversation with the original developers, and I respect that those developers took time to write down important concepts. When I see a file like "using_trajectory_logging.md", I feel curious about what insights it contains.

I follow a consistent pattern:
1. Check if relevant docs exist
2. Read docs thoroughly to understand the design intent
3. Only then look at code to see the implementation
4. Refer back to docs when something in code seems unclear

I particularly value documentation that explains the "why" behind decisions, not just the "how." I'd rather understand the system's philosophy first, then see how that philosophy manifests in code.

#includes
@docs/README.md

## Overview

This is a Madrona Escape Room - a high-performance 3D multi-agent reinforcement learning environment built on the Madrona Engine. It implements a navigation environment where agents explore and try to maximize their forward progress through the world.

**Tech Stack:**
- C++ (core simulation using Entity Component System pattern)
- Python (PyTorch-based PPO training)
- CMake build system

# Headless Mode Quick Reference

## Overview
Run simulation without graphics for benchmarking, testing, or server deployment.

## Usage
```bash
./build/headless [OPTIONS]
```

### Required Options
- `--num-worlds N, -n N`: Number of parallel worlds (e.g., 1, 1024, 8192)
- `--num-steps N, -s N`: Simulation steps to run (e.g., 100, 1000, 10000)

### Optional
- `--cuda <n>`: Use CUDA/GPU execution mode on device n (default: CPU)
- `--rand-actions`: Generate random actions (for benchmarking)
- `--replay <file>`: Replay actions from file
- `--seed <value>`: Set random seed (default: 5)
- `--track, -t`: Enable trajectory tracking (default: world 0, agent 0)
- `--track-world N`: Specify world to track (default: 0)
- `--track-agent N`: Specify agent to track (default: 0)
- `--track-file <file>`: Save trajectory to file

## Examples
```bash
# Basic CPU run
./build/headless -n 1 -s 1000

# GPU benchmark with random actions
./build/headless --cuda 0 -n 8192 -s 1000 --rand-actions

# Track agent 0 in world 5
./build/headless -n 100 -s 1000 --track --track-world 5

# Replay recording
./build/headless -n 2 -s 1000 --replay demo.bin
```

# Essential Commands

### Building the Project
```bash
# Initial setup (from repo root)
mkdir build
cmake -B build
make -C build -j8 -s

# Note: If build fails with compiler errors about -nostdlib++ or -march=x86-64-v3,
# use the bundled Madrona toolchain:
# cmake -B build -DCMAKE_TOOLCHAIN_FILE=external/madrona/cmake/toolchain/madrona-toolchain.cmake

# Install Python package (ALWAYS use uv)
uv pip install -e .
```

**IMPORTANT**: Use explicit values like `-j8` in make commands. DO NOT USE shell substitutions like `$(nproc)` as claude-code will prompt for user confirmation on command substitutions.

### Running the Simulation
```bash
# For headless mode, see the Headless Mode Quick Reference above
# For viewer options, run: ./build/viewer --help

# Interactive viewer examples:
./build/viewer                                   # Single world on CPU
./build/viewer -n 4                              # 4 worlds on CPU
./build/viewer --cuda 0 -n 8 --track            # 8 worlds on GPU with tracking
./build/viewer -n 2 --record demo.bin           # Record demo (press SPACE to start)
./build/viewer -n 2 --replay demo.bin           # Replay recorded demo

# Python benchmark
uv run python scripts/sim_bench.py --num-worlds 1024 --num-steps 1000 --gpu-id 0
```

### Training
```bash
# Basic CPU training
uv run python scripts/train.py --num-worlds 1024 --num-updates 100 --ckpt-dir build/checkpoints

# Full GPU training with optimizations
uv run python scripts/train.py --num-worlds 8192 --num-updates 5000 --profile-report --fp16 --gpu-sim --ckpt-dir build/checkpoints/
```

### Inference
```bash
# Run trained policy
uv run python scripts/infer.py --num-worlds 1 --num-steps 1000 --fp16 --ckpt-path build/checkpoints/5000.pth --action-dump-path build/dumped_actions

# Replay in viewer
./build/viewer -n 1 --replay build/dumped_actions
```

### Development

#### Python coding standards

IMPORTANT: when writing tests for python code ALWAYS use the pytest framework
IMPORTANT: NEVER use sys.path.insert(..) ALWAYS place modules in the correct locations so they can be imported correctly by python

#### C++ coding standards

IMPORTANT: C++ exceptions and RTTI are disabled in this project.

```bash
# Rebuild after C++ changes
make -C build -j8 -s

# Run quick smoke tests
./tests/quick_test.sh   # Quick CPU test (assumes built)
./tests/smoke_test.sh   # Full rebuild and test with FPS verification (270k-310k range)

# Run tests - ALWAYS run CPU tests first, then GPU tests
# Run all CPU tests (default)
uv run --group dev pytest tests/python/ -v --no-gpu

# Only after CPU tests pass, run GPU tests
uv run --group dev pytest tests/python/ -v -k "gpu"

# Run specific test file
uv run --group dev pytest tests/python/test_bindings.py -v --tb=short

# Run reward system tests
uv run --group dev pytest tests/python/test_reward_system.py -v

# Run tests with action recording (saves to test_recordings/)
uv run --group dev pytest tests/python/test_reward_system.py -v --record-actions

# Run tests with recording and automatic visualization
uv run --group dev pytest tests/python/test_reward_system.py -v --record-actions --visualize

# View recorded actions from tests in the interactive viewer
# Tests record 4 worlds, so viewer must use 4 worlds to replay correctly
./build/viewer 4 --cpu test_recordings/test_name/actions.bin
```

### Debugging using GDB

If the user asks to "debug the code", or "debug it" or generally references "debuggging" then interpret this as a request to use the GDB tool to gather information about the behaviour of the program, and follow the following procedure

1. read the file docs/GDB_GUIDE.md
2. use the debug tool in your MCP library to gather information on the problem at hand, or study the code

# Documentation

## Instructions for common tasks

Procedure files live in docs/development/instructions

These files contain step-by-step instructions for common programming task
When the users requests a task, and the task involves any of the below.. read the file and follow the instructions inside to implement that part of the task,
in planning mode.. copy the steps into your plan

- ADD_COMPONENT.md: add a new component
- ADD_SYSTEM.md: add a new system
- EXPORT_COMPONENT.md : exporting a component to the python bindings and mgr.cpp

## Documentation creation rules

- when proposing new documents, always add them to the docs folder
- when the user asks save a plan to a file it should be written to docs\plan_dump

# Scratch files

When creating a python file that we don't want to keep in the repo, but just want to use for the purposes of a one-time test or to study/understand code better.  Create it in the ./scratch folder. 

# Code Classification System

The codebase uses a three-tier classification system to help developers understand what needs to be modified:

### [BOILERPLATE]
Pure Madrona framework code that should never be changed. This includes:
- CPU/GPU execution infrastructure
- Memory management systems
- Rendering pipeline setup
- Base class structures

### [REQUIRED_INTERFACE]
Methods and structures that every Madrona environment must implement:
- `loadPhysicsObjects()` - Load collision meshes and configure physics
- `loadRenderObjects()` - Load visual assets and materials
- `triggerReset()` - Reset episode state (for episodic environments)
- `setAction()` - Accept actions from the policy
- Tensor export methods - Define observation/action spaces
- Reset and action buffers - Required for episodic RL

### [GAME_SPECIFIC]
Implementation details unique to this escape room game:
- Action structure fields (moveAmount, moveAngle, rotate, grab)
- Observation tensor types and shapes
- Object types and their physics properties
- Material colors and textures
- Game constants (see Game-Specific Constants section below)

When creating a new environment:
1. Keep all `[BOILERPLATE]` code unchanged
2. Implement all `[REQUIRED_INTERFACE]` methods with your game's logic
3. Replace all `[GAME_SPECIFIC]` code with your game's details

## Game-Specific Constants

The following constants are defined in `src/consts.hpp` and used throughout the codebase:

### Core Game Parameters
- `numAgents` (2) - Number of agents per world
- `numRooms` (2) - Number of rooms in the level
- `maxEntitiesPerRoom` (15) - Maximum entities per room (walls, buttons, blocks)
- `worldWidth` (18) - Width of the world in units
- `worldLength` (10) - Length of the world in units

### Physics Parameters
- `deltaT` (0.04f) - Fixed timestep in seconds
- `numPhysicsSubsteps` (4) - Physics substeps per frame
- `agentSpeed` (8.f) - Agent movement speed
- `agentRotateSpeed` (5.f) - Agent rotation speed in radians/sec

### Rendering Parameters
- `numRows` (64) - Camera view height in pixels
- `numCols` (64) - Camera view width in pixels
- `verticalFOV` (60.f * M_PI / 180.f) - Camera field of view

### RL Parameters
- `episodeLen` (200) - Maximum steps per episode
- `rewardPerLevel` (1.f) - Reward for completing a room

### Enum Definitions (in `src/types.hpp`)
- `ExportID::NumExports` - Total number of exported tensors
- `TaskGraphID::NumTaskGraphs` - Number of task graphs (currently 1: Step)
- `SimObject::NumObjects` - Total number of object types

### Build Configuration (in CMakeLists.txt)
- `MADRONA_ESCAPE_ROOM_SRC_LIST` - List of source files for GPU compilation
- `GPU_HIDESEEK_SRC_LIST` - GPU-specific source file list
- `GPU_HIDESEEK_COMPILE_FLAGS` - GPU compilation flags

## Python Package Management

**IMPORTANT**: This project uses `uv` for all Python package management. Always use `uv` instead of `pip` or plain `python` commands.

#### Testing Configuration

The project uses pytest for testing with the following configuration in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["."]
norecursedirs = ["external", "build", ".venv", "*.egg", "dist"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

This configuration:
- Excludes the `external` directory (containing Madrona framework code) from test discovery
- Skips build artifacts and virtual environment directories
- Follows standard pytest naming conventions for test discovery

##### Testing Flags

The test suite supports several custom flags:

- `--no-gpu`: Skip all tests that require GPU. This is the default way to run tests.
- `--record-actions`: Record agent actions during test execution for viewer replay
- `--visualize`: Automatically launch the viewer after tests complete (requires --record-actions)

##### Testing Order

**IMPORTANT**: Always run tests in this order:
1. **CPU tests first**: `uv run --extra test pytest tests/python/ -v --no-gpu`
2. **GPU tests only after CPU tests pass**: `uv run --extra test pytest tests/python/ -v -k "gpu"`

This ensures that basic functionality is validated before testing GPU-specific features.

#### Using MCP GDB Server in Claude Code

When the users says "debug the code" or "debug this function" or makes other references to "debugging" interpret this as the user requesting you to 

1. read docs/GDB_GUIDE.md
2. use the MCP GDB server to trace through the code and diagnose the cause of issues, check the values of variables during execution etc

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
- `src/viewer.cpp'` : Renders the environment, and allows user to switch to first person view to control the agent
- `src/bindings.cpp` : Python bindings for the manager, types and constants
- `scripts/sim_bench.py` : PyInstrument benchmark to check performance 

### System Execution Order
The task graph defines precise system dependencies:
1. **Input Processing**: 
   - `movementSystem` - Converts actions to forces/torques
2. **Physics Pipeline**: 
   - `broadphase_setup` - Builds BVH for collision detection
   - `physicsStepTasks` - Runs physics simulation (multiple substeps)
   - `agentZeroVelSystem` - Zeros agent velocities for direct control
   - `physicsCleanupTasks` - Finalizes physics state
3. **Game Logic Pipeline**:
   - `stepTrackerSystem` - Decrements steps remaining, sets done flag
   - `rewardSystem` - Computes rewards (only at episode end)
   - `resetSystem` - Handles episode resets
4. **GPU-specific Systems** (when enabled):
   - `RecycleEntitiesNode` - Reclaims deleted entity IDs
   - `ResetTmpAllocNode` - Clears temporary allocations
   - Entity sorting by world ID for cache efficiency
5. **Post-Reset**:
   - Second `broadphase_setup` - Rebuilds BVH after reset
6. **Output**: 
   - `collectObservationsSystem` - Gathers observations for policy
7. **Rendering** (if enabled):
   - `RenderingSystem::setupTasks` - Updates render state

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
- **Self Observation**: Contains agent's global position (x, y, z), maximum Y reached (maxY), and rotation angle (theta)
- **Normalization**: Positions normalized by world size, angles normalized by π
- **Reward**: Progress-based reward given only at episode end (normalized max Y position reached)

### Modifying the Simulator

**Important**: When adding entities, update `max_total_entities` calculation in `Sim::Sim()` to ensure the BVH has sufficient space. The physics system currently requires knowing the upper bound at initialization.

### GPU Execution of Systems

Madrona automatically compiles and executes systems on GPU using [NVRTC](https://docs.nvidia.com/cuda/nvrtc/index.html)

#### Supported C++ Features in Systems

**✅ Can Use:**
- Control flow (`if/else`, loops, `switch`)
- Math functions (`fminf`, `sqrt`, `sin`, etc.)
- Local variables and fixed-size arrays
- Function calls to other inline functions
- Component read/write access
- Ternary operators and all arithmetic operations

**❌ Cannot Use:**
- Dynamic memory allocation (`new`, `malloc`)
- STL containers (`std::vector`, `std::map`)
- Virtual functions or RTTI
- Exceptions or `try/catch`
- File I/O or system calls
- Global/static variables
- Recursive functions
 
#### GPU-Specific Optimizations

1. **Warp-Level Systems** using `CustomParallelForNode`:
   ```cpp
   #ifdef MADRONA_GPU_MODE
   auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
       lidarSystem, 32, 1,  // 32 threads per entity
       Entity, Lidar
   >>({dependencies});
   #endif
   ```

2. **Entity Sorting** for memory coalescing:
   ```cpp
   #ifdef MADRONA_GPU_MODE
   auto sort_agents = queueSortByWorld<Agent>(builder, {deps});
   #endif
   ```

3. **Thread Indexing** in warp-level systems:
   ```cpp
   inline void lidarSystem(Engine &ctx, Entity e, Lidar &lidar) {
   #ifdef MADRONA_GPU_MODE
       int thread_id = threadIdx.x % 32;  // Thread's position in warp
       // Each thread traces a different ray
   #endif
   }
   ```

### Task Graph Setup (setupTasks)

The `setupTasks` function in `sim.cpp` builds a static execution graph defining the order and dependencies of all systems that run each simulation step. Understanding this is crucial for modifying the simulation.

#### Data Flow Phases

1. **Input Processing Phase**
   - Transforms external actions into physics forces/torques
   - Must run first to prepare physics inputs

2. **Pre-Physics State Updates**
   - Updates positions of kinematically-controlled entities (doors)
   - Must complete before spatial structure build

3. **Spatial Structure Build (Phase 1)**
   - Builds BVH (Bounding Volume Hierarchy) for efficient spatial queries
   - Depends on all entity positions being finalized
   - Required by any system doing raycasting or proximity queries

4. **Spatial Query Systems**
   - Systems like grab that need raycasting
   - Must run after BVH build, before physics

5. **Physics Simulation Pipeline**
   - Broadphase collision detection (uses BVH)
   - Narrowphase collision detection  
   - Constraint solving (multiple substeps)
   - Position/velocity integration
   - **Monolithic step** - no other systems can run during physics

6. **Post-Physics Corrections**
   - Modifications to physics output (e.g., zeroing velocities)
   - Must run after physics, before cleanup

7. **Physics Cleanup**
   - Finalizes physics state
   - Clears temporary collision data
   - Must run after all physics modifications

8. **Game Logic Phase**
   - Reward calculation, door logic, episode management
   - Reads physics state but doesn't modify it
   - Safe to run in any order within this phase

9. **Episode Management**
   - Reset detection and world regeneration
   - May invalidate entire world state

10. **Spatial Structure Rebuild (Phase 2)**
    - Required after reset (world geometry may have changed)
    - Runs every frame due to GPU constraints (no conditional execution)

11. **Observation Collection**
    - Gathers all entity states for external consumers
    - Must run last, after all state updates

#### Key Principles

**Physics Data Access Rules:**
- **Pre-physics systems**: Can write positions/forces, no physics data exists yet
- **During physics**: No other systems can execute
- **Post-physics systems**: Can read physics results, limited modification allowed
- **Logic systems**: Read-only access to physics state

**Spatial Query Dependencies:**
- Any system using raycasting must run after BVH build
- Systems modifying positions invalidate the BVH
- Two BVH builds needed: pre-physics and post-reset

**GPU Constraints:**
- No dynamic graph modification - all nodes run every frame
- Conditional logic implemented via no-op execution
- Entity recycling handled separately on GPU backend

**Common Patterns:**
```cpp
// System with dependencies
auto my_sys = builder.addToGraph<ParallelForNode<Engine,
    mySystem,
    Component1,
    Component2
>>({dependency1, dependency2});

// Physics-dependent system  
auto post_phys_sys = builder.addToGraph<...>({phys_cleanup});

// Observation system (runs last)
auto obs_sys = builder.addToGraph<...>({post_reset_broadphase});
```
