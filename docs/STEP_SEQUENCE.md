### Step Sequence

The `Manager::step()` function is the main entry point for advancing the simulation by one timestep. It's called from the main loop after initialization and orchestrates all simulation updates.

#### **Manager::step() Implementation**

The step function executes three main phases:

1. **Simulation Update**
    - Calls `impl_->run()` which polymorphically dispatches to:
        - CPU Mode: `CPUImpl::run()` → `cpuExec.run()` → `ThreadPoolExecutor::run()`
        - CUDA Mode: `CUDAImpl::run()` → `gpuExec.run(stepGraph)` → `MWCudaExecutor::run()`
    - This executes the task graph defined in `Sim::setupTasks()`

2. **Render State Update** (if rendering enabled)
    - Calls `renderMgr->readECS()` to synchronize render state with ECS components
    - Copies transform data (Position, Rotation, Scale) to render buffers
    - Updates instance data for all visible entities

3. **Batch Rendering** (if batch renderer enabled)
    - Calls `renderMgr->batchRender()` to perform actual rendering
    - Renders all worlds in a single batch operation
    - Outputs to configured render targets

**Key Points:**
- **Zero-Copy Design**: Simulation data is directly accessible to Python without copying
- **Polymorphic Execution**: Same interface for CPU and GPU execution modes
- **Optional Rendering**: Render updates only occur when visualization is enabled
- **Synchronous Execution**: Each step completes before returning to caller

**Usage Example:**
```cpp
// Main simulation loop
for (int64_t i = 0; i < num_steps; i++) {
    mgr.step();  // Advance simulation by one timestep
    
    // Python can now read updated observations via tensor methods
    // e.g., mgr.rewardTensor(), mgr.doneTensor(), etc.
}
```