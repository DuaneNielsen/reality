# Lidar Ray Visualization Plan

## Pre-Reading List
Before implementing this plan, review these files:
1. `/src/sim.cpp` - Lines 329-384: Current lidarSystem implementation
2. `/src/types.hpp` - Lines 156-160: RenderOnlyEntity archetype example
3. `/src/level_gen.cpp` - Lines 218-234: Origin marker creation pattern
4. `/src/mgr.cpp` - Asset loading and registration
5. `/src/viewer.cpp` - Keyboard input handling
6. `/src/consts.hpp` - Line 41: numLidarSamples = 128

## Overview
Create visual representation of the 128 lidar rays in the 3D viewer to show what the agent "sees" with its 120-degree forward arc sensor.

## Approach
Use thin cylindrical entities to represent lidar rays, similar to how the origin marker axes are rendered in the viewer.

## Implementation Steps

### 1. Add Lidar Ray Entities to Simulation Data
**File: `src/types.hpp`**
- Add to `SimData` struct:
  ```cpp
  Entity lidarRays[consts::numAgents][consts::numLidarSamples];
  bool showLidarRays = false;  // Toggle for visualization
  ```
- Define new archetype:
  ```cpp
  struct LidarRayEntity : public madrona::Archetype<
      Position, Rotation, Scale, ObjectID,
      madrona::render::Renderable
  > {};
  ```

### 2. Create Lidar Ray Asset
**File: `src/mgr.cpp`**
- Add to asset loading (around line 200):
  ```cpp
  // Thin cylinder for lidar ray visualization
  geo_data.cylinderGeo(0.02f, 0.02f, 1.0f, 8, 1);  // Very thin, unit length
  ```
- Register as new asset ID in `src/asset_ids.hpp`:
  ```cpp
  LIDAR_RAY = 9,  // Add to AssetIDs enum
  ```
- Set material properties:
  - Color: Semi-transparent green (0, 255, 0, 128)
  - Emissive for visibility in dark areas

### 3. Initialize Ray Entities
**File: `src/sim.cpp`** (in world generation, around line 700)
```cpp
// Create lidar ray entities for visualization
for (int32_t agent_idx = 0; agent_idx < consts::numAgents; agent_idx++) {
    for (int32_t ray_idx = 0; ray_idx < consts::numLidarSamples; ray_idx++) {
        Entity ray = ctx.makeRenderableEntity<LidarRayEntity>();
        ctx.data().lidarRays[agent_idx][ray_idx] = ray;
        
        // Initially hidden (scale to zero)
        ctx.get<Position>(ray) = Vector3{0, 0, -1000};  // Off-screen
        ctx.get<Rotation>(ray) = Quat{1, 0, 0, 0};
        ctx.get<Scale>(ray) = Diag3x3{0, 0, 0};  // Hidden
        ctx.get<ObjectID>(ray) = ObjectID{(int32_t)AssetIDs::LIDAR_RAY};
    }
}
```

### 4. Update Ray Positions in Lidar System
**File: `src/sim.cpp`** (modify lidarSystem, lines 329-384)
```cpp
inline void lidarSystem(Engine &ctx, Entity e, Lidar &lidar)
{
    // ... existing code ...
    
    // Get agent index for ray entity lookup
    int32_t agent_idx = 0;  // Assuming single agent per world
    bool visualize = ctx.data().showLidarRays && ctx.data().enableRender;
    
    auto traceRay = [&](int32_t idx) {
        // ... existing ray tracing code ...
        
        // Update visualization entity if enabled
        if (visualize) {
            Entity ray_entity = ctx.data().lidarRays[agent_idx][idx];
            
            if (hit_entity != Entity::none()) {
                // Position ray at agent location
                ctx.get<Position>(ray_entity) = pos + 0.5f * math::up;
                
                // Create rotation to align cylinder with ray direction
                // Cylinder default orientation is along Y axis, need to rotate
                Vector3 up = Vector3{0, 1, 0};
                Quat rot_to_ray = Quat::fromBasis(ray_dir, up);
                ctx.get<Rotation>(ray_entity) = rot_to_ray;
                
                // Scale: length = hit distance, width/height = thin
                float ray_length = hit_t;
                ctx.get<Scale>(ray_entity) = Diag3x3{0.02f, ray_length, 0.02f};
            } else {
                // No hit - extend to max distance or hide
                ctx.get<Scale>(ray_entity) = Diag3x3{0, 0, 0};  // Hide
            }
        }
    };
    
    // ... rest of existing code ...
}
```

### 5. Add Viewer Toggle
**File: `src/viewer.cpp`** (in keyboard handler, around line 500)
```cpp
// Toggle lidar visualization with 'L' key
if (input.keyHit(Key::L)) {
    // Access simulation data and toggle flag
    bool& show_lidar = /* access ctx.data().showLidarRays */;
    show_lidar = !show_lidar;
    
    if (show_lidar) {
        printf("Lidar visualization: ON\n");
    } else {
        printf("Lidar visualization: OFF\n");
        // Hide all ray entities when turned off
        // Set all ray scales to zero
    }
}
```

### 6. Performance Optimizations
- **Conditional Updates**: Only update ray entities when `showLidarRays` is true
- **Subset Visualization**: Show every 4th ray for cleaner view:
  ```cpp
  if (idx % 4 == 0 && visualize) {
      // Update ray entity
  }
  ```
- **Color Coding**: 
  - Green rays: Hit walls/obstacles
  - Yellow rays: Hit other agents
  - Red rays: No hit (max distance)
- **Distance Fade**: Make rays more transparent based on distance

## Alternative Approaches

### A. Single Mesh Approach (More Efficient)
Instead of 128 separate entities, create a single mesh with 128 line segments:
- Pro: Much more efficient, single draw call
- Con: Requires custom mesh generation and update logic

### B. Debug Overlay Approach
Draw rays as 2D overlay on screen using ImGui:
- Pro: No impact on 3D scene performance
- Con: Less intuitive, harder to see spatial relationships

### C. Particle System Approach
Use particle system to show lidar "pulses":
- Pro: Visually interesting, shows sensor actively scanning
- Con: More complex to implement

## Testing Plan
1. Enable visualization with 'L' key
2. Verify rays appear in 120-degree arc
3. Check ray lengths match lidar depth values
4. Test with different obstacle configurations
5. Profile performance impact (target <5% FPS drop)

## Success Criteria
- [ ] Rays visible in viewer when toggled on
- [ ] Rays correctly oriented in 120-degree forward arc
- [ ] Ray lengths match actual lidar sensor readings
- [ ] Minimal performance impact (<5% FPS reduction)
- [ ] Clean toggle on/off functionality
- [ ] Colors indicate hit type (wall/agent/no-hit)

## Notes
- The lidar system already calculates all ray directions and hit distances
- Reuse existing rendering infrastructure (similar to origin markers)
- Keep visualization optional to avoid performance impact during training
- Consider adding UI slider for ray transparency/visibility