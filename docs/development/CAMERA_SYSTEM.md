# Deep Dive: Madrona Escape Room Camera System

The camera system in the Madrona Escape Room viewer is a sophisticated multi-layered architecture that handles both free-flying cameras and entity-attached cameras. This document provides a complete technical overview of how camera transforms flow from CPU input to GPU rendering.

## 1. Camera Initialization

The viewer starts with an initial camera position and rotation configured in `src/viewer.cpp:432-440`:

- **Position**: `{0.0f, -14.0f, 35.0f}` - positioned above and behind the room
- **Rotation**: Calculated from `consts::display::viewerRotationFactor` (0.4) to look down at ~72 degrees
- **Move Speed**: Set to `consts::display::defaultCameraDist` (10.0f units)

## 2. Camera Data Structure

The `ViewerCam` structure (`viewer_common.hpp`) maintains:

```cpp
struct ViewerCam {
    math::Vector3 position;      // World space position
    math::Vector3 fwd;          // Forward direction vector
    math::Vector3 up;           // Up direction vector
    math::Vector3 right;        // Right direction vector
    bool perspective = true;    // Perspective vs orthographic
    float fov = 60.f;          // Field of view in degrees
    float orthoHeight = 5.f;    // Orthographic view height
    math::Vector2 mousePrev;    // Previous mouse position for deltas
};
```

## 3. Camera Control System

The camera uses a dual-mode control system implemented in `handleCamera()`:

### Free Look Mode (Right-click or Shift held)
- Mouse controls rotation via quaternion composition
- WASD moves relative to camera orientation
- Mouse sensitivity: `2e-4f` radians per pixel
- Rotation applied as: `rotation = (around_up * around_right).normalize()`

### Fly Mode (Default)
- WASD moves relative to world axes
- No rotation control
- Simplified navigation for level inspection

## 4. Camera Transform Pipeline

### 4.1 CPU Side Packing

The camera transform is packed into a compact `PackedViewData` structure in `packView()`:

```cpp
// Quaternion from basis vectors (inverse for view transform)
rotation = Quat::fromBasis(right, fwd, up).inv()

// Perspective projection parameters
fov_scale = 1.0f / tan(toRadians(fov * 0.5f))
aspect = width / height
x_scale = fov_scale / aspect
y_scale = -fov_scale  // Negative for Y-axis flip

// Packed into 3 float4 vectors:
data[0] = {position.x, position.y, position.z, rotation.w}
data[1] = {rotation.x, rotation.y, rotation.z, x_scale}
data[2] = {y_scale, z_near, 0, 0}
```

### 4.2 GPU Upload

The packed data flow:
1. Written to staging buffer (`frame.viewStaging`)
2. Copied to GPU via `vkCmdCopyBuffer`
3. Made available to shaders as a uniform buffer
4. Accessed in shaders via `viewDataBuffer` or `flycamBuffer`

## 5. Vertex Transformation

The vertex shader (`viewer_draw.hlsl`) performs the complete transformation:

### 5.1 World to View Space

```hlsl
// Compute relative transform
to_view_translation = rotateVec(cam_rotation_inverse, object_pos - cam_pos)
to_view_rotation = composeQuats(cam_rotation_inverse, object_rotation)

// Transform vertex to view space
view_pos = rotateVec(to_view_rotation, object_scale * vertex_pos) + to_view_translation
```

### 5.2 Perspective Projection

```hlsl
// Madrona uses a unique coordinate system:
// - Y is forward (into screen)
// - Z is up
// - X is right

clip_pos = {
    x_scale * view_pos.x,     // Horizontal with FOV scaling
    y_scale * view_pos.z,     // Vertical (Z becomes Y in clip space)
    z_near,                   // Constant near plane for depth
    view_pos.y                // Forward distance for W division
}
```

## 6. Coordinate System Peculiarities

Madrona uses a **non-standard coordinate system**:

| Axis | Madrona | Standard (OpenGL/DirectX) |
|------|---------|---------------------------|
| X    | Right   | Right                     |
| Y    | Forward | Up                        |
| Z    | Up      | Forward/Back              |

This is why the projection matrix swaps Y and Z:
- `clip.x = view.x * x_scale` (standard horizontal)
- `clip.y = view.z * y_scale` (Z becomes screen Y)
- `clip.w = view.y` (Y is the depth for perspective divide)

## 7. Entity-Attached Cameras

For agent cameras using the `RenderCamera` component:

### 7.1 Attachment (`level_gen.cpp:80`)
```cpp
render::RenderingSystem::attachEntityToView(ctx,
    agent,
    fov,                          // Field of view in degrees
    consts::rendering::cameraZNear,  // Near plane distance
    Vector3{0.0f, 1.0f, 1.0f}     // Offset: 1 unit forward, 1 unit up
);
```

### 7.2 Transform Update (`ecs_system.cpp:275`)
```cpp
// Camera inherits entity position/rotation
Vector3 camera_pos = pos + rot.rotateVec(cam.cameraOffset);

// Pack into PerspectiveCameraData
cam_data.position = camera_pos;
cam_data.rotation = rot.inv();
cam_data.xScale = fov_scale / aspect_ratio;
cam_data.yScale = -fov_scale;
```

### 7.3 View Selection (`viewer_draw.hlsl:171`)
```hlsl
if (push_const.viewIdx == 0) {
    camera_data = unpackViewData(flycamBuffer[0]);  // Free camera
} else {
    int view_idx = (push_const.viewIdx - 1) + viewOffsetsBuffer[push_const.worldIdx];
    camera_data = unpackViewData(viewDataBuffer[view_idx]);  // Entity camera
    
    // Inherit aspect ratio from free camera
    PerspectiveCameraData fly_cam = unpackViewData(flycamBuffer[0]);
    camera_data.xScale = fly_cam.xScale;
    camera_data.yScale = fly_cam.yScale;
}
```

## 8. Dual Renderer Support

The system supports two rendering modes:

### Flycam Mode
- Single viewport with free or entity camera
- Full resolution rendering
- Interactive navigation
- Used for debugging and visualization

### Grid Mode (Batch Renderer)
- Multiple viewports in grid layout
- Renders all worlds simultaneously
- Fixed camera per world
- Used for ML training data generation

## 9. Performance Optimizations

- **Unified Buffer**: All view data stored in single GPU buffer
- **Instanced Drawing**: Single draw call for multiple instances
- **Cached Transforms**: View matrices computed once per frame
- **Quaternion Math**: Avoids expensive matrix operations
- **Staging Buffers**: Efficient CPU-to-GPU data transfer

## 10. Key Implementation Details

### Perspective Division
Happens automatically in GPU hardware after vertex shader output

### Depth Precision
Uses reverse-Z mapping (near plane at `z_near`, far at infinity) for better floating-point precision distribution

### Aspect Ratio Handling
Computed dynamically from framebuffer dimensions:
```cpp
float aspect = float(fb_width) / float(fb_height);
```

### Camera Smoothing
Frame-rate independent movement using time delta:
```cpp
cam.position += translate * cam_move_speed * secondsPerFrame;
```

### View Frustum
The perspective frustum is defined by:
- **FOV**: Field of view angle (default 60°)
- **Aspect**: Width/height ratio
- **Near**: Near clipping plane (`z_near`)
- **Far**: Implicitly at infinity (reverse-Z)

## File References

Key files for camera system implementation:

- `src/viewer.cpp`: Main viewer loop and camera initialization
- `src/viewer_core.cpp`: Camera state management
- `external/madrona/src/viz/viewer.cpp`: Camera controls and UI
- `external/madrona/src/viz/viewer_renderer.cpp`: Camera data packing
- `external/madrona/src/render/shaders/viewer_draw.hlsl`: GPU transformation
- `external/madrona/src/render/ecs_system.cpp`: Entity camera updates

## Summary

This architecture allows seamless switching between multiple camera modes while maintaining consistent rendering performance across both interactive viewing and batch training scenarios. The non-standard coordinate system requires careful attention when implementing new features, but provides efficient integration with the ECS architecture and physics simulation.