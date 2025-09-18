# Manager Specification

## Overview
The Manager component serves as the primary interface between the Python API and the Madrona simulation engine. It handles simulation lifecycle management, asset loading, tensor exports for reinforcement learning, and provides recording/replay functionality for the Madrona Escape Room simulation.

## Key Files

### Source Code
Primary implementation files:

- `src/mgr.cpp` - Manager implementation with CPU/GPU execution support
- `src/mgr.hpp` - Public interface and configuration structures
- `src/level_io.cpp` - Level I/O utilities for loading and saving levels
- `src/level_io.hpp` - Level I/O interface definitions

### Test Files

#### C++ Tests
- `tests/cpp/unit/test_manager_gpu.cpp` - GPU manager functionality tests
- `tests/cpp/unit/test_manager_gpu_stress.cpp` - GPU manager stress tests with large world counts
- `tests/cpp/unit/test_recording_utilities.cpp` - Recording and replay utility functions
- `tests/cpp/integration/test_manager_from_replay.cpp` - Manager creation from replay files
- `tests/cpp/unit/test_direct_cpp.cpp` - Direct C++ Manager API tests (CPU focus)

#### Python Tests
- `tests/python/test_native_recording.py` - Native recording functionality tests
- `tests/python/test_native_recording_gpu.py` - GPU-specific recording tests
- `tests/python/test_native_recording_replay_roundtrip.py` - Recording/replay roundtrip validation
- `tests/python/test_multi_level_recording_replay.py` - Multi-level recording and replay tests
- `tests/python/test_compass_tensor.py` - Compass observation tensor tests

## Architecture

### System Integration
The Manager acts as the bridge between high-level Python training code and the low-level ECS simulation. It integrates with:
- Madrona's TaskGraphExecutor for CPU execution
- MWCudaExecutor for GPU execution
- RenderManager for batch rendering and visualization
- PhysicsLoader for collision object management
- CompiledLevel system for world initialization

### GPU/CPU Code Separation
- **GPU (NVRTC) Code**: None - Manager is CPU-only orchestration layer
- **CPU-Only Code**: All Manager code (mgr.cpp, mgr.hpp)
- **Shared Headers**: types.hpp, sim.hpp (for ExportID enum)

## Data Structures

### Primary Structure

#### Manager::Config
```cpp
struct Config {
    // Execution configuration
    ExecMode execMode;                      // CPU or CUDA execution
    int gpuID;                              // GPU device ID for CUDA mode
    uint32_t numWorlds;                     // Number of parallel worlds

    // Level configuration
    std::vector<std::optional<CompiledLevel>> perWorldCompiledLevels;  // Per-world level data

    // Rendering configuration
    bool enableBatchRenderer;               // Enable batch rendering for RL
    uint32_t batchRenderViewWidth;          // Width of rendered views
    uint32_t batchRenderViewHeight;         // Height of rendered views
    uint32_t renderMode;                     // RGBD or Depth only

    // External rendering (optional)
    render::APIBackend *extRenderAPI;       // External render API
    render::GPUDevice *extRenderDev;        // External GPU device

    // Simulation parameters
    uint32_t randSeed;                      // Random seed
    bool autoReset;                         // Auto-reset episodes
    float customVerticalFov;                // Custom camera FOV
};
```

**Key Points:**
- Must provide compiled levels for all worlds (no defaults allowed)
- External rendering allows integration with existing render pipelines

### Supporting Structures

#### Manager::Impl
```cpp
struct Impl {
    Config cfg;                             // Configuration
    PhysicsLoader physicsLoader;            // Physics asset manager
    WorldReset *worldResetBuffer;           // Reset control buffer
    LidarVisControl *lidarVisBuffer;        // Lidar visualization control
    Action *agentActionsBuffer;             // Agent action buffer

    // Rendering state
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    // Recording/replay state
    bool isRecordingActive;
    std::ofstream recordingFile;
    ReplayMetadata recordingMetadata;
    Optional<ReplayData> replayData;
    uint32_t currentReplayStep;

    // Trajectory logging
    bool enableTrajectoryLogging;
    int32_t trackWorldIdx;
    int32_t trackAgentIdx;
    FILE* trajectoryLogFile;
};
```

**Purpose:** Internal implementation hiding execution-specific details (CPU vs GPU)

#### Invariants
- Recording can only start from step 0 of a fresh simulation
- Replay data must match current world configuration
- All tensor exports maintain consistent shape across steps

## Module Interface

### Manager

#### Manager (constructor)

**Purpose:** Initialize the simulation manager with specified configuration

**Parameters:**
- `cfg`: Configuration structure with execution mode, world count, levels, etc.

**Returns:** N/A (constructor)

**Preconditions:**
- Valid compiled levels must be provided for all worlds
- GPU ID must be valid if using CUDA mode
- Render configuration must be consistent

**Specs:**
- Loads physics and render assets during initialization
- Creates CPU or GPU executor based on exec mode
- Forces initial reset and step to populate observations
- Initializes all exported tensor buffers
- **Multi-Level Support:**
  - Accepts per-world level configuration via `compiled_levels` parameter
  - Each world can have a different CompiledLevel structure
  - Level deduplication during recording (stores unique levels only)
  - Supports heterogeneous world configurations for varied training scenarios

**Error Handling:**
- **Invalid Levels:** Aborts if no compiled levels provided
- **CUDA Not Available:** Fatal error if CUDA requested but not compiled
- **Asset Load Failure:** Fatal error if assets cannot be loaded

#### Manager::fromReplay (static factory)

**Purpose:** Create a Manager instance from a replay file with embedded levels

**Parameters:**
- `filepath`: Path to replay file containing embedded levels
- `execMode`: Execution mode (CPU or CUDA)
- `gpuID`: GPU device ID for CUDA mode (default: -1)

**Returns:** std::unique_ptr<Manager> (nullptr on failure)

**Preconditions:**
- Valid replay file with embedded levels
- File must exist and be readable

**Specs:**
- Extracts embedded levels from replay file
- Uses replay metadata for configuration:
  - Number of worlds from metadata
  - Random seed from metadata
  - Embedded compiled levels
- Automatically loads replay data for playback
- Creates fully initialized Manager ready for replayStep()

**Error Handling:**
- **Non-existent File:** Returns nullptr
- **Invalid Format:** Returns nullptr
- **Corrupt Metadata:** Returns nullptr

#### Manager::readReplayMetadata (static)

**Purpose:** Read metadata from replay file without creating Manager

**Parameters:**
- `filepath`: Path to replay file

**Returns:** std::optional<ReplayMetadata>

**Preconditions:**
- File should exist (returns empty optional if not)

**Specs:**
- Extracts metadata without loading full replay
- Returns structure containing:
  - num_worlds: Number of parallel worlds
  - seed: Random seed used
  - num_steps: Total steps recorded
- Non-destructive read (file remains unchanged)

**Error Handling:**
- **Missing File:** Returns empty optional
- **Invalid Format:** Returns empty optional

#### stopRecording (Python: stop_recording)

**Purpose:** Stop active recording and finalize file

**Parameters:** None

**Returns:** void

**Preconditions:**
- Recording should be active (but safe to call when not recording)

**Specs:**
- Flushes remaining data to file
- Updates final metadata with total steps
- Closes recording file
- Marks recording as inactive
- Can be called multiple times safely (idempotent)

**Error Handling:**
- **No Active Recording:** Safe no-op - does not raise error

#### isRecording (Python: is_recording)

**Purpose:** Check if recording is currently active

**Parameters:** None

**Returns:** bool - True if recording is active, False otherwise

**Preconditions:** None

**Specs:**
- Thread-safe query of recording state
- Does not modify any state
- Available in both C++ and Python APIs

**Error Handling:**
- None - always returns valid bool

#### step

**Purpose:** Execute one simulation timestep across all worlds

**Parameters:** None

**Returns:** void

**Preconditions:**
- Manager must be fully initialized
- Actions should be set via setAction() or action tensor

**Specs:**
- Records actions if recording is active
- Executes task graph for all worlds in parallel
- Updates render manager if batch rendering enabled
- Logs trajectory if tracking is enabled
- Marks that simulation has progressed (affects recording eligibility)

**Error Handling:**
- **CUDA Errors:** Handled internally by CUDA executor
- **Render Failures:** Logged but don't stop simulation

#### setAction

**Purpose:** Set discrete actions for a specific world

**Parameters:**
- `world_idx`: World index (0 to numWorlds-1)
- `move_amount`: Movement speed (0-3)
- `move_angle`: Movement direction (0-7)
- `rotate`: Rotation action (0-4)

**Returns:** void

**Preconditions:**
- World index must be valid
- Action values must be within valid ranges

**Specs:**
- Writes action to buffer at world offset
- Handles CPU/GPU memory transfer automatically
- Actions persist until overwritten

**Error Handling:**
- **Invalid World Index:** Undefined behavior (no bounds checking)
- **Invalid Actions:** Clamped or wrapped by simulation

#### triggerReset

**Purpose:** Reset a specific world to start a new episode

**Parameters:**
- `world_idx`: World index to reset

**Returns:** void

**Preconditions:**
- World index must be valid

**Specs:**
- Sets reset flag for specified world
- Reset occurs at next step() call
- Resets agent positions, rewards, and episode state

**Error Handling:**
- **Invalid World Index:** Undefined behavior (no bounds checking)

### Tensor Exports

#### actionTensor

**Purpose:** Get tensor for setting agent actions

**Parameters:** None

**Returns:** Tensor of shape [numWorlds * numAgents, 3] with Float32 elements

**Preconditions:**
- Manager must be initialized

**Specs:**
- Returns writable tensor for action input
- **Flattened format**: [numWorlds * numAgents, actionDims]
  - With single agent per world: effectively [numWorlds, 3]
- Action dimensions: [move_amount, move_angle, rotate]
- Direct memory access via devicePtr() for both CPU and GPU
- Changes take effect on next step()
- GPU tensors have gpuID >= 0 and non-null devicePtr
- Tensor shape remains consistent after world resets
- Supports world counts: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024+

**Error Handling:**
- None - always returns valid tensor

#### selfObservationTensor

**Purpose:** Get agent self-observation data

**Parameters:** None

**Returns:** Tensor of shape [numWorlds, numAgents, 5] with Float32 elements

**Preconditions:**
- Manager must be initialized

**Specs:**
- Shape: [numWorlds, numAgents, 5]
  - With single agent per world: [numWorlds, 1, 5]
- Observation features (5 dimensions):
  - position x, y, z (3 floats)
  - maxY (1 float) - maximum Y coordinate reached
  - theta (1 float) - agent rotation/heading
- Direct memory access via devicePtr()
- Updated after each step()
- Read-only tensor (simulation writes, user reads)
- GPU tensors have gpuID >= 0 and non-null devicePtr
- Tensor shape remains consistent after world resets
- Values are valid floats (no NaN values under normal operation)

**Error Handling:**
- None - always returns valid tensor

#### rewardTensor

**Purpose:** Get per-world rewards

**Parameters:** None

**Returns:** Tensor of shape [numWorlds, 1] with Float32 elements

**Preconditions:**
- Manager must be initialized

**Specs:**
- Contains current step reward for each world
- Reset to 0 on episode reset
- Updated after each step()
- GPU tensors have gpuID >= 0 and non-null devicePtr
- Tensor shape remains consistent after world resets

**Error Handling:**
- None - always returns valid tensor

#### doneTensor

**Purpose:** Get episode done flags

**Parameters:** None

**Returns:** Tensor of shape [numWorlds, 1] with UInt8 elements

**Preconditions:**
- Manager must be initialized

**Specs:**
- Contains done flag (0 or 1) for each world
- Set to 1 when episode ends (max steps or termination condition)
- Reset to 0 on episode reset
- GPU tensors have gpuID >= 0 and non-null devicePtr
- Tensor shape remains consistent after world resets

**Error Handling:**
- None - always returns valid tensor

#### compassTensor (Python: compass_tensor)

**Purpose:** Get agent compass observations (one-hot heading encoding)

**Parameters:** None

**Returns:** Tensor of shape [numWorlds, numAgents, 128] with Float32 elements

**Preconditions:**
- Manager must be initialized

**Specs:**
- Shape: [numWorlds, numAgents, 128]
  - With single agent per world: [numWorlds, 1, 128]
- One-hot encoding of agent heading direction
  - Exactly one element is 1.0, rest are 0.0
  - 128 buckets representing full 360° rotation
  - Each bucket represents ~2.8125 degrees (360/128)
- Updated after each step based on agent rotation
- Active bucket index = argmax(compass_tensor[world, agent, :])

**Error Handling:**
- None - always returns valid tensor

### Replay Determinism Guarantees

#### Fundamental Guarantee
Given the same initial seed and action sequence, replay MUST produce:
- Identical episode lengths
- Identical reset points (same step numbers)
- Identical entity positions at every step
- Identical observations and rewards
- Bit-for-bit identical simulation state

#### Requirements for Determinism
1. **PRNG Consistency**: Use only the initial seed - no reseeding or external randomness
2. **Action Fidelity**: Apply recorded actions at exact step indices
3. **Reset Detection**: Episode termination conditions must be purely deterministic
4. **No Hidden State**: All simulation state must derive from seed + actions

#### Verification
Determinism can be verified by:
- Recording trajectory during original run
- Replaying with same seed + actions
- Comparing trajectories - they must match exactly

### Recording/Replay

#### startRecording (Python: start_recording)

**Purpose:** Begin recording simulation actions to file

**Parameters:**
- `filepath`: Path to output recording file

**Returns:** Result enum (Success or error code) / Python: raises RuntimeError on error

**Preconditions:**
- Must be called before any steps (at initialization)
- No recording already active

**Specs:**
- Records all world actions at each step
- Embeds compiled levels in recording file (v3 format)
- Updates metadata incrementally for crash recovery
- File format: [Metadata][Levels][Actions]
- **Determinism Note**: Recording only captures seed + actions. Full determinism is guaranteed by the simulation's PRNG.
- **Version 3 Metadata structure** (192 bytes total):
  - `magic`: uint32_t (4 bytes) - Magic number for validation
  - `version`: uint32_t (4 bytes) - Format version (currently 3)
  - `sim_name`: char[64] - Simulation name (null-terminated ASCII)
  - `level_name`: char[64] - Level name (null-terminated ASCII)
  - `num_worlds`: uint32_t (4 bytes) - Number of parallel worlds
  - `num_agents_per_world`: uint32_t (4 bytes) - Agents per world
  - `num_steps`: uint32_t (4 bytes) - Total steps recorded
  - `actions_per_step`: uint32_t (4 bytes) - Actions per step
  - `timestamp`: uint64_t (8 bytes) - Recording timestamp
  - `seed`: uint32_t (4 bytes) - Random seed used
  - `reserved`: uint32_t[7] (28 bytes) - Reserved for future use
- Embedded level: Full MER_CompiledLevel structure
- Actions: num_steps × num_worlds × 3 int32_t values
- Works on both CPU and GPU execution modes
- **Multi-Level Recording Support:**
  - Handles heterogeneous level configurations across worlds
  - Deduplicates similar levels during storage
  - Maintains per-world level assignment for replay
  - Supports different maze layouts, spawn points, and world scales
  - Records level-specific trajectory differences

**Error Handling:**
- **Already Recording:** Returns ErrorRecordingAlreadyActive / Python: RuntimeError("Recording already in progress")
- **Not at Step 0:** Returns ErrorRecordingNotAtStepZero
- **File IO Error:** Returns ErrorFileIO / Python: may not raise (implementation dependent)
- **Invalid Path:** C++ prints error but may not raise exception

#### loadReplay

**Purpose:** Load replay data from file

**Parameters:**
- `filepath`: Path to replay file

**Returns:** bool (success/failure)

**Preconditions:**
- Valid replay file in v3 format
- Number of worlds must match

**Specs:**
- Loads embedded levels and action sequence
- Validates metadata magic and version
- Prepares for replay playback via replayStep()
- Extracts metadata:
  - Number of worlds (updates viewer configuration)
  - Number of steps
  - Random seed for deterministic replay
- Extracts embedded MER_CompiledLevel data
- Validates file size matches expected format
- Detects malformed files (too small or corrupted)
- **Multi-Level Replay Support:**
  - Reconstructs per-world level assignments from recording
  - Validates that current Manager configuration matches recorded setup
  - Ensures deterministic replay across heterogeneous level configurations
  - Maintains level-specific behavior during replay

**Error Handling:**
- **Invalid Format:** Returns false with error message
- **Version Mismatch:** Only v3 format supported
- **Corrupt File:** Returns false
- **Missing File:** Returns false with invalid metadata
- **Malformed File:** Detected by size validation
- **Level Mismatch:** Validation error if worlds/levels don't match recording

#### replayStep (Python: replay_step)

**Purpose:** Apply next frame of replay actions

**Parameters:** None

**Returns:** bool (true if replay finished)

**Preconditions:**
- Replay must be loaded via loadReplay() or Manager created via from_replay()

**Specs:**
- Applies recorded actions to all worlds
- Advances replay counter
- Returns true when all steps consumed
- Updates action tensors with recorded values
- Maintains exact action reproduction for deterministic replay
- Must be called before step() to load actions for that step
- **Determinism Guarantee**: Maintains deterministic PRNG sequence - no reseeding occurs during replay
- Supports round-trip verification:
  - Record session → Replay session → Compare trajectories
  - Action sequences match exactly during replay
  - Observation values match exactly during replay
- Handles edge cases:
  - Single-step recordings (returns true after first call)
  - Zero-step recordings (returns true immediately)
  - Partial/truncated files (graceful degradation)
- Works across episode resets (maintains determinism)
- Action validation during replay:
  - Move amount: 0-3 (STOP, SLOW, MEDIUM, FAST)
  - Move angle: 0-7 (8 directional movement)
  - Rotate: 0-4 (rotation actions)

**Error Handling:**
- **No Replay Loaded:** Returns true (no-op)
- **Past End:** Returns true (replay complete)
- **Corrupted Action Data:** May load invalid values (detection required at application level)
- **Partial Files:** Continues until available data exhausted

#### SimManager.from_replay (Python static method)

**Purpose:** Create a SimManager instance from a replay file

**Parameters:**
- `filepath`: Path to replay file
- `exec_mode`: Execution mode (ExecMode.CPU or ExecMode.CUDA)
- `gpu_id`: GPU device ID for CUDA mode (default: 0)

**Returns:** SimManager instance configured for replay

**Preconditions:**
- Valid replay file must exist
- GPU must be available if CUDA mode requested

**Specs:**
- Python wrapper for C++ Manager::fromReplay
- Automatically loads replay data
- Configures manager with embedded levels from recording
- Ready for replay_step() calls immediately

**Error Handling:**
- **Invalid File:** Returns None or raises exception
- **GPU Not Available:** Falls back to CPU or raises exception

#### hasReplay (Python: has_replay)

**Purpose:** Check if replay data is loaded

**Parameters:** None

**Returns:** bool - True if replay data is available

**Preconditions:** None

**Specs:**
- Query method to check replay availability
- Returns true after successful loadReplay() or from_replay creation
- Returns false for normal (non-replay) managers

**Error Handling:**
- None - always returns valid bool

#### getReplayStepCount (Python: get_replay_step_count)

**Purpose:** Get current and total replay step counts

**Parameters:** None

**Returns:** tuple (current_step, total_steps)

**Preconditions:**
- Replay must be loaded

**Specs:**
- Returns (0, 0) if no replay loaded
- current_step increments with each replay_step() call
- total_steps reflects number of steps in recording

**Error Handling:**
- **No Replay:** Returns (0, 0)

### Recording/Replay Utilities

#### Trajectory Comparison

**Purpose:** Verify deterministic replay by comparing trajectory files

**Functionality:**
- Parse CSV trajectory files with format:
  ```
  Step [step]: World [world] Agent [agent]: pos=(x,y,z) rot=[degrees]° progress=[value]
  ```
- Extract trajectory data points with fields:
  - step: uint32_t
  - world, agent: int
  - x, y, z: float coordinates
  - rotation: float degrees
  - progress: float value
- Compare trajectories for exact match (determinism verification)
- Support per-world trajectory files for multi-world recordings
- **Round-trip Testing Support:**
  - Automated record → replay → compare workflow
  - Text-based trajectory file comparison (line-by-line)
  - Difference detection for validation
  - Integration with pytest fixtures for automated testing
  - Debug session support with persistent file paths

#### Level Comparison

**Purpose:** Verify embedded levels match source levels

**Functionality:**
- Extract embedded MER_CompiledLevel from recording files
- Compare level dimensions (width, height, num_tiles)
- Validate level data integrity after embedding

#### Recording Format Validation

**Purpose:** Ensure recording files conform to expected binary format

**Functionality:**
- Validate metadata structure (3 × uint32_t)
- Verify embedded level data completeness
- Check action data size matches metadata
- Calculate and verify expected file size
- **File Integrity Validation:**
  - Magic number verification
  - Version validation (currently version 3)
  - Boundary checking for metadata/level/action sections
  - Action data corruption detection
  - Partial file handling and truncation detection
  - File size consistency checks:
    ```
    expected_size = metadata_size + level_size + (num_steps × num_worlds × 3 × 4)
    ```
- **Action Data Validation:**
  - Range checking for action values:
    - move_amount: 0-3 (valid range)
    - move_angle: 0-7 (valid range)
    - rotate: 0-4 (valid range)
  - Binary structure validation
  - Little-endian int32_t format verification

### Trajectory Logging

#### enableTrajectoryLogging

**Purpose:** Enable detailed trajectory logging for a specific agent

**Parameters:**
- `world_idx`: World index to track (0 to numWorlds-1)
- `agent_idx`: Agent index to track (0 to numAgents-1)
- `filename`: Optional filename for log output (nullopt for stdout)

**Returns:** void

**Preconditions:**
- Valid world and agent indices
- File path must be writable if provided

**Specs:**
- Logs position, rotation, compass, progress, reward, done state each step
- Creates/overwrites log file or uses stdout
- Logs initial state immediately (step 0)
- Format: "Episode step X: World Y Agent Z: pos=(x,y,z) rot=θ° compass=C progress=P reward=R done=D term=T"
- **Round-trip verification support:**
  - Produces identical output for record vs replay sessions
  - Enables deterministic trajectory comparison
  - Line-by-line comparison validates exact reproduction
  - Detects trajectory differences when actions differ
  - Compatible with automated testing frameworks
- File operations:
  - Overwrites existing files (no append mode)
  - Flushes data immediately for crash recovery
  - Works with temporary files for testing

**Error Handling:**
- **Invalid Indices:** Prints error and returns without enabling
- **File Open Failed:** Prints error and returns without enabling

#### disableTrajectoryLogging

**Purpose:** Stop trajectory logging and close log file

**Parameters:** None

**Returns:** void

**Preconditions:** None

**Specs:**
- Stops logging trajectory data
- Closes log file if not stdout
- Resets tracking indices to -1

**Error Handling:**
- **No Active Logging:** Safe no-op

#### getAgentPosition

**Purpose:** Get current world position of a specific agent

**Parameters:**
- `world_idx`: World index (0 to numWorlds-1)
- `agent_idx`: Agent index (0 to numAgents-1)

**Returns:** math::Vector3 with agent position

**Preconditions:**
- Valid world and agent indices

**Specs:**
- Extracts position from AgentPosition export tensor
- Handles CPU/GPU memory transfer automatically
- Returns position in world coordinates

**Error Handling:**
- **Invalid Indices:** Undefined behavior (no bounds checking)
- **CUDA Errors:** Returns zero vector if CUDA unavailable

### Visualization Control

#### toggleLidarVisualization

**Purpose:** Toggle lidar ray visualization for a specific world

**Parameters:**
- `world_idx`: World index to toggle

**Returns:** void

**Preconditions:**
- Valid world index

**Specs:**
- Reads current state from lidar control buffer
- Toggles enabled flag (0→1 or 1→0)
- Writes updated state back to buffer
- Prints status message to stdout

**Error Handling:**
- **Invalid World Index:** Undefined behavior (no bounds checking)

#### toggleLidarVisualizationGlobal

**Purpose:** Toggle lidar visualization for all worlds simultaneously

**Parameters:** None

**Returns:** void

**Preconditions:** None

**Specs:**
- Uses first world's state to determine target toggle state
- Applies same toggle to all worlds
- Prints global status message

**Error Handling:**
- **CUDA Errors:** Handled internally

## ViewerCore Integration

### Overview
The Manager can be integrated with ViewerCore for interactive visualization and trajectory tracking.

#### ViewerCore::toggleTrajectoryTracking

**Purpose:** Enable/disable trajectory tracking for a specific world

**Parameters:**
- `world_idx`: World index to toggle tracking

**Functionality:**
- Toggles trajectory tracking state for the specified world
- When enabled, logs trajectory data to stdout:
  - Prints "Trajectory logging enabled for world X"
  - Logs agent position, rotation, and state each step
- When disabled, prints "Trajectory logging disabled for world X"
- Tracking state can be queried with isTrackingTrajectory()

#### ViewerCore::stepSimulation

**Purpose:** Step the underlying Manager simulation

**Functionality:**
- Calls Manager::step() internally
- Updates trajectory logs if tracking is enabled
- Maintains synchronization between viewer and simulation state

## Configuration

### Build Configuration
```cmake
# Manager sources are added to CPU compilation only
set(MANAGER_SRCS
    mgr.cpp
    madrona_escape_room_c_api.cpp
)
```

### Runtime Configuration
Configuration options set via Manager::Config:
- `execMode`: Choose CPU or CUDA execution
- `numWorlds`: Parallel world count (affects memory usage)
  - GPU optimal: Power of 2 values (32, 64, 128, 256, 512, 1024)
  - Tested up to 1024 worlds on GPU
- `enableBatchRenderer`: Enable/disable batch rendering
- `randSeed`: Deterministic simulation seed
- `autoReset`: Automatic episode reset on termination

### GPU-Specific Behavior

#### Performance Characteristics
- Power-of-2 world counts are optimal for GPU execution
- Supports stress testing with repeated create/destroy cycles
- NVRTC compilation occurs on first Manager creation (can be expensive)
- Shared managers can reuse compiled kernels for faster test execution

#### Tensor Properties on GPU
- All tensor exports have `gpuID >= 0` when using CUDA execution
- Tensor device pointers are non-null for GPU tensors
- Tensor shapes remain stable across resets and multiple steps
- Direct GPU memory access available via `devicePtr()`

#### Error Handling Limitations
- Invalid GPU ID causes process abort (CUDA runtime behavior)
- No graceful error handling for invalid GPU devices currently
- Future improvement: std::expected factory pattern for Manager creation