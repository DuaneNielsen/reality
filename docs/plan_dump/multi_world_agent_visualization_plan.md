# Plan: Multi-World Agent Visualization for Exploration Analysis

## Understanding
The Madrona engine already collects all entity instances from all worlds into shared arrays:
- `InstanceData` array contains ALL entities from ALL worlds
- `instanceOffsets` array tracks where each world's entities start
- Each instance has a `worldIDX` field identifying its world
- The viewer currently only draws instances from the selected world

## Implementation Approach

### Phase 1: Modify Viewer Rendering to Show Multiple Worlds
1. **Modify the viewer's instance culling/drawing logic** (`external/madrona/src/viz/viewer_renderer.cpp`)
   - Currently filters instances by selected world ID
   - Change to include instances from worlds 0-127 (or fewer if less are running)
   - Apply visual differentiation (color/transparency) based on worldIDX

2. **Add shader modifications** (`external/madrona/data/shaders/`)
   - Modify vertex/fragment shaders to apply per-world coloring
   - Use worldIDX to generate unique colors (hue based on world index)
   - Apply transparency to make overlapping agents visible

3. **Add toggle control in viewer** (`src/viewer.cpp`)
   - Add keyboard shortcut (e.g., 'M' for multi-world mode)
   - Toggle between normal (single world) and multi-world visualization
   - Store toggle state in ViewerCore

### Phase 2: Optimize for Agent-Only Rendering
1. **Filter by entity type**
   - Only render Agent entities (filter by ObjectID == AGENT)
   - Skip walls, floors, and other static geometry
   - This reduces visual clutter when showing 128 worlds

2. **Performance considerations**
   - Limit to first 128 worlds maximum
   - Use instance culling to skip non-agent entities early
   - Consider LOD based on number of visible worlds

## Key Files to Modify
1. **Madrona engine files:**
   - `external/madrona/src/viz/viewer_renderer.cpp` - Main rendering logic
   - `external/madrona/src/render/batch_renderer.cpp` - Instance collection
   - `external/madrona/data/shaders/batch_draw_rgb.hlsl` - Shader for coloring

2. **Application files:**
   - `src/viewer.cpp` - Add toggle control
   - `src/viewer_core.hpp/cpp` - Track multi-world mode state

## Technical Details
- Instances are already collected in `RenderECSBridge::instances`
- Each instance has position, rotation, scale, objectID, and worldIDX
- The viewer loop already has access to all world data
- We just need to change which instances get drawn and how they're colored

This approach leverages the existing Madrona architecture where all world data is already available, we just need to visualize it differently.