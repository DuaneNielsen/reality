# Fix Camera Tracking Plan

## Problem
The tracking camera is not properly tracking the agent because `getAgentPosition()` is returning normalized/rescaled coordinates instead of actual world coordinates.

## Solution
Export the actual Position component from the simulation and use it directly.

## Files to Modify

### 1. `src/sim.hpp`
- Add `AgentPosition` to the `ExportID` enum (DONE)
- This creates a new export slot for the actual Position component

### 2. `src/sim.cpp`
- Add export registration for Position component:
  ```cpp
  registry.exportColumn<Agent, Position>(
      (uint32_t)ExportID::AgentPosition);
  ```
- Location: After line 101, after the Progress export

### 3. `src/mgr.cpp`
- Completely rewrite `getAgentPosition()` to use the Position export instead of denormalizing SelfObservation
- New implementation:
  ```cpp
  math::Vector3 Manager::getAgentPosition(int32_t world_idx, int32_t agent_idx) const
  {
      int32_t idx = world_idx * madEscape::consts::numAgents + agent_idx;
      
      if (impl_->cfg.execMode == ExecMode::CUDA) {
  #ifdef MADRONA_CUDA_SUPPORT
          CUDAImpl *cuda_impl = static_cast<CUDAImpl*>(impl_.get());
          void *export_ptr = cuda_impl->gpuExec.getExported((uint32_t)ExportID::AgentPosition);
          
          static Position host_pos;
          const Position* device_pos = (const Position*)export_ptr;
          cudaMemcpy(&host_pos, &device_pos[idx], sizeof(Position), cudaMemcpyDeviceToHost);
          
          return math::Vector3{host_pos.x, host_pos.y, host_pos.z};
  #else
          return math::Vector3{0.0f, 0.0f, 0.0f};
  #endif
      } else {
          CPUImpl *cpu_impl = static_cast<CPUImpl*>(impl_.get());
          void *export_ptr = cpu_impl->cpuExec.getExported((uint32_t)ExportID::AgentPosition);
          const Position* pos_data = ((const Position*)export_ptr) + idx;
          return math::Vector3{pos_data->x, pos_data->y, pos_data->z};
      }
  }
  ```

### 4. `src/camera_controller.cpp`
- Remove the hardcoded camera position and target
- Restore the original tracking logic:
  ```cpp
  // Remove lines 213-217 (the HARDCODED comments and fixed position/target)
  // Replace with:
  state_.position = targetPosition_ + offset_;
  Vector3 toTarget = targetPosition_ - state_.position;
  ```

### 5. `src/viewer.cpp`
- The code for initializing tracking camera on switch is already fixed (lines 538-540)
- No additional changes needed

## Build and Test
1. Run `./build.sh` to rebuild
2. Test with `./build/viewer`
3. Press 'C' to switch to tracking mode
4. Verify camera follows the agent properly

## Expected Result
- `getAgentPosition()` returns actual world coordinates (e.g., around -6.25, 3.0, 0.0 for default spawn)
- Tracking camera properly follows the agent as it moves
- No more looking at origin (0, 0, 0) issue