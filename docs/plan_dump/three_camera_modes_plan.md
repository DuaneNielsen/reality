# Three Camera Modes Implementation Plan

## Pre-Reading Files

**IMPORTANT: Read these files first to understand the current implementation:**

1. **Main viewer implementation**: `/home/duane/madrona_escape_room/src/viewer.cpp`
2. **Viewer core logic**: `/home/duane/madrona_escape_room/src/viewer_core.cpp` and `/home/duane/madrona_escape_room/src/viewer_core.hpp`
3. **ECS types with observation structures**: `/home/duane/madrona_escape_room/src/types.hpp` (specifically `SelfObservation` struct)
4. **Level generation (camera attachment)**: `/home/duane/madrona_escape_room/src/level_gen.cpp` (lines 70-85)
5. **Manager class**: `/home/duane/madrona_escape_room/src/mgr.cpp` and `/home/duane/madrona_escape_room/src/mgr.hpp` (for `selfObservationTensor()` method)
6. **Madrona viewer internals**: 
   - `/home/duane/madrona_escape_room/external/madrona/src/viz/viewer.cpp`
   - `/home/duane/madrona_escape_room/external/madrona/src/viz/viewer_common.hpp`
   - `/home/duane/madrona_escape_room/external/madrona/include/madrona/viz/viewer.hpp`

## Current System Understanding

### Camera System Architecture
1. **Madrona Viewer** manages camera internally via `ViewerCam` struct with:
   - `position` (Vector3)
   - `fwd`, `up`, `right` (orientation vectors)
   - Camera selection via `viewIdx`:
     - 0 = Free Camera
     - 1+ = Agent cameras (Agent 0, Agent 1, etc.)

2. **Agent Camera Attachment** (in `level_gen.cpp`):
   ```cpp
   render::RenderingSystem::attachEntityToView(ctx,
       agent,
       fov, consts::rendering::cameraZNear,
       Vector3{0.0f, 1.0f, 1.0f}); // Camera offset
   ```

3. **Self Observation Structure** (in `types.hpp`):
   ```cpp
   struct SelfObservation {
       float globalX;
       float globalY;
       float globalZ;
       float maxY;
       float theta;  // Agent rotation
   };
   ```

## Implementation Plan

### Goal
Add three camera modes to the viewer:
1. **Free Camera** - Existing fly camera (unchanged)
2. **Follow Camera** - Third-person camera that follows agent from behind
3. **Agent View** - First-person view from agent's perspective (existing)

### Challenge
The Madrona viewer's camera is managed internally and doesn't expose a public API to modify camera position dynamically. We need a workaround.

### Solution Approach

#### Option 1: Modify Our Viewer Loop (Recommended)
Since we can't easily modify Madrona's internal camera, we'll track camera mode in our viewer and switch between different viewing strategies.

#### Option 2: Extend Madrona Viewer (More Complex)
Modify Madrona's viewer source to expose camera control methods.

## Detailed Implementation (Option 1)

### 1. Add Camera Mode Management to `src/viewer.cpp`

```cpp
// Add after includes
enum class CameraMode : uint32_t {
    FREE = 0,
    FOLLOW = 1,
    AGENT_VIEW = 2,
    NUM_MODES = 3
};

// Add static variable to track mode
static CameraMode current_camera_mode = CameraMode::FREE;
static const char* camera_mode_names[] = {
    "Free Camera",
    "Follow Camera",
    "Agent View"
};
```

### 2. Add Keyboard Handler for Mode Switching

In the viewer loop's input handler (around line 478-518):

```cpp
// Add F key handling for camera mode switching
if (input.keyHit(Key::F)) {
    current_camera_mode = static_cast<CameraMode>(
        (static_cast<uint32_t>(current_camera_mode) + 1) % 
        static_cast<uint32_t>(CameraMode::NUM_MODES)
    );
    
    std::cout << "Camera mode: " << camera_mode_names[
        static_cast<uint32_t>(current_camera_mode)] << std::endl;
    
    // Handle mode switching
    if (current_camera_mode == CameraMode::AGENT_VIEW) {
        // Switch viewer to agent camera (viewIdx = 1)
        // This requires accessing viewer's internal state
        // We may need to track this separately and apply in UI callback
    }
}
```

### 3. Implement Follow Camera Logic

In the frame update callback (around line 573-589):

```cpp
}, [&viewer_core, &viewer, &mgr]() {
    // Update frame (handles timing and conditionally steps simulation)
    viewer_core.updateFrame();
    
    // Follow camera implementation
    if (current_camera_mode == CameraMode::FOLLOW) {
        // Get agent position from self observation
        auto self_obs = mgr.selfObservationTensor();
        float* obs_data = static_cast<float*>(self_obs.data());
        
        // Extract agent position and rotation
        float agent_x = obs_data[0];  // globalX
        float agent_y = obs_data[1];  // globalY
        float agent_z = obs_data[2];  // globalZ
        float agent_theta = obs_data[4]; // theta (rotation in radians)
        
        // Calculate follow camera position
        // Position camera 10 units behind and 8 units above agent
        float follow_distance = 10.0f;
        float follow_height = 8.0f;
        float follow_x = agent_x - follow_distance * cosf(agent_theta);
        float follow_y = agent_y - follow_distance * sinf(agent_theta);
        float follow_z = agent_z + follow_height;
        
        // Calculate look-at direction (towards agent)
        math::Vector3 cam_pos{follow_x, follow_y, follow_z};
        math::Vector3 agent_pos{agent_x, agent_y, agent_z + 1.0f}; // Look at agent's center
        math::Vector3 look_dir = (agent_pos - cam_pos).normalize();
        
        // TODO: Apply camera position and orientation to viewer
        // This is the main challenge - we need to access viewer's internal camera
        // Possible solutions:
        // 1. Modify Madrona to expose camera control
        // 2. Use reflection/pointer manipulation (hacky)
        // 3. Create our own rendering view that follows agent
    }
    
    // Rest of existing code...
}
```

### 4. Update Help Text

Around line 422-426, add:

```cpp
std::cout << "  F: Cycle camera modes (Free/Follow/Agent)\n";
```

## Alternative Implementation: Modify Madrona Viewer

If we can modify Madrona source, add these methods to `Viewer` class:

### In `/external/madrona/include/madrona/viz/viewer.hpp`:
```cpp
// Add public methods
void setCameraPosition(const math::Vector3& pos);
void setCameraRotation(const math::Quat& rot);
math::Vector3 getCameraPosition() const;
math::Quat getCameraRotation() const;
```

### In `/external/madrona/src/viz/viewer.cpp`:
```cpp
void Viewer::setCameraPosition(const math::Vector3& pos) {
    impl_->vizCtrl.flyCam.position = pos;
}

void Viewer::setCameraRotation(const math::Quat& rot) {
    // Convert quaternion to direction vectors
    impl_->vizCtrl.flyCam.fwd = rot.rotateVec(math::fwd);
    impl_->vizCtrl.flyCam.up = rot.rotateVec(math::up);
    impl_->vizCtrl.flyCam.right = rot.rotateVec(math::right);
}
```

## Testing Plan

1. **Build the project**: `./build.sh`
2. **Test with obstacle course level**: 
   ```bash
   ./build/viewer --load obstacle_course_16x64.lvl --num-worlds 1
   ```
3. **Verify camera modes**:
   - Press F to cycle through modes
   - Free Camera: Should work as before with WASD + mouse
   - Follow Camera: Should track agent from behind
   - Agent View: Should show first-person perspective

## Potential Issues and Solutions

### Issue 1: Cannot Access Viewer's Internal Camera
**Solution**: May need to fork Madrona and add camera control API, or use a different approach like modifying the render camera offset dynamically.

### Issue 2: Camera Jitter in Follow Mode
**Solution**: Add smoothing/interpolation to camera movement:
```cpp
static math::Vector3 smoothed_cam_pos = {0, 0, 0};
smoothed_cam_pos = math::Vector3::lerp(smoothed_cam_pos, target_pos, 0.1f);
```

### Issue 3: Mode Switching Conflicts with Viewer's Internal State
**Solution**: Track our mode separately and coordinate with viewer's viewIdx when needed.

## Benefits

1. **Three distinct viewing modes** for different use cases
2. **Preserves existing functionality** - all current features remain
3. **Follow camera** automatically tracks agent movement and rotation
4. **Easy mode switching** with single key press
5. **Extensible** - can add more modes or adjust follow distance/height

## Next Steps

1. Implement basic mode switching and UI
2. Research best way to control Madrona viewer's camera
3. Implement follow camera logic
4. Add smoothing and polish
5. Test with various levels including obstacle course