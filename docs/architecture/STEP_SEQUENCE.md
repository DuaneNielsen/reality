# Manager Step Sequence

## Overview
The `Manager::step()` function advances the Madrona simulation by one timestep, executing the task graph and optionally updating rendering state.

```
Manager::step()
├─ Simulation Update Phase
│  ├─ Step 1: Execute task graph
│  │  ├─ CPU: CPUImpl::run()
│  │  └─ GPU: CUDAImpl::run()
│  └─ Step 2: Run registered systems
│     └─ Executes Sim::setupTasks() graph
│
├─ Render State Update Phase [if rendering enabled]
│  ├─ Step 1: Synchronize ECS data
│  │  └─ renderMgr->readECS()
│  └─ Step 2: Update instance buffers
│     ├─ Copy transform components
│     └─ Update visible entities
│
└─ Batch Rendering Phase [if batch renderer enabled]
   ├─ Step 1: Execute batch render
   │  └─ renderMgr->batchRender()
   └─ Step 2: Output to render targets
```

## Input

### Input Sources
- **Action Tensors**: Agent actions from policy network via `mgr.actionTensor()`
- **Reset Flags**: Episode reset signals via `mgr.resetTensor()`
- **Configuration**: Simulation parameters from Manager::Config

### Input Data Format

#### Simple Values
- **step_idx** (`int64_t`): Current simulation step counter
- **num_worlds** (`uint32_t`): Number of parallel worlds being simulated
- **exec_mode** (`ExecMode`): CPU or CUDA execution mode

#### Structured Data
**Action Structure**
```cpp
struct Action {
    int32_t moveAmount;     // Movement speed [0-3]
    int32_t moveAngle;      // Movement direction [0-7]
    int32_t rotate;         // Rotation amount [0-4]
};
```

## Processing

### Processing Pipeline
```
Actions → Task Graph Execution → Physics Update → Observation Generation → Render Update
```

### Detailed Sequence

#### Phase 1: Simulation Update
Executes the ECS task graph to advance simulation state

#### Step 1: Polymorphic Dispatch
**Function:** `impl_->run()`
**Location:** `src/mgr.cpp:450`
**Purpose:** Routes execution to appropriate backend

**Details:**
- CPU Mode dispatches to ThreadPoolExecutor
- CUDA Mode dispatches to MWCudaExecutor
- Maintains zero-copy tensor access

#### Step 2: Task Graph Execution
**Function:** `cpuExec.run()` or `gpuExec.run(stepGraph)`
**Location:** `external/madrona/src/mw/cpu_exec.cpp:100` or `cuda_exec.cpp:200`
**Purpose:** Executes registered ECS systems

**Details:**
- Runs systems defined in Sim::setupTasks()
- Processes physics, actions, observations
- Updates all entity components

### Phase 2: Render State Synchronization
Updates render buffers with latest ECS state

#### Step 1: Read ECS Data
**Function:** `renderMgr->readECS()`
**Location:** `external/madrona/src/viz/viewer_renderer.cpp:500`
**Purpose:** Copies ECS component data to render buffers

**Details:**
- Reads Position, Rotation, Scale components
- Updates per-instance transform matrices
- Filters visible entities

### Phase 3: Batch Rendering
Performs GPU rendering of all worlds

#### Step 1: Execute Batch Render
**Function:** `renderMgr->batchRender()`
**Location:** `external/madrona/src/viz/viewer_renderer.cpp:600`
**Purpose:** Renders all worlds in single GPU operation

**Details:**
- Batches draw calls across worlds
- Applies view and projection matrices
- Outputs to configured render targets

## Output

### Output Data

#### Direct Outputs
Data available immediately after step() returns:

- **Observations** (`float tensor`): Agent observations via `mgr.selfObservationTensor()`
- **Rewards** (`float tensor`): Step rewards via `mgr.rewardTensor()`
- **Done Flags** (`int32_t tensor`): Episode termination via `mgr.doneTensor()`
- **Lidar Data** (`float tensor`): Distance measurements via `mgr.lidarTensor()`
- **Depth Images** (`uint8_t tensor`): Depth buffers via `mgr.depthTensor()` [if rendering enabled]
- **RGB Images** (`uint8_t tensor`): Color buffers via `mgr.rgbTensor()` [if rendering enabled]

#### Side Effects
State changes from step execution:

- **Entity Positions**: Updated based on physics and actions
- **Collision State**: Resolved contacts and overlaps
- **Episode State**: Reset if done flags triggered
- **Render Buffers**: Updated if rendering enabled

### Output Format
```cpp
// Observation tensor layout
struct ObservationLayout {
    float self_observation[13];  // Position, rotation, velocity, etc.
    float lidar[128];            // Distance measurements
    float pad[15];               // Padding for alignment
};
```