### Asset Loading

#### Physics Assets (`loadPhysicsObjects()`):
- **Collision Mesh Loading**: Calls `AssetImporter::importFromDisk()` for OBJ files
- **Rigid Body Processing Pipeline**:
    1. Import raw meshes as convex hulls
    2. Process with `RigidBodyAssets::processRigidBodyAssets()`:
        - Optimizes convex hulls for collision detection
        - Computes bounding volumes and centroids
        - Calculates mass properties (center of mass, inertia tensor)
        - Builds collision primitives (hulls, planes)
        - Allocates contiguous memory block for cache efficiency
    3. Configure physics properties via `setupHull()`:
        - Movable objects: Small inverse mass values
        - Static objects: Zero inverse mass
        - Controlled entities: Unit mass with rotation constraints
        - Friction coefficients: μs=0.5, μd=0.5-0.75
    4. Load processed data via `PhysicsLoader::loadRigidBodies()`

#### Render Assets (`loadRenderObjects()`):
- **Meshes**: Calls `AssetImporter::importFromDisk()` for visual assets
- **Materials**: Configured with RGB values and texture indices
- **Textures**: Loaded via `ImageImporter::importImages()`
- **Lighting**: Set via `RenderManager::configureLighting()`
- **Final Load**: `RenderManager::loadObjects()` uploads to GPU

### Memory Layout
1. **World Data**: Array of `Sim` instances
2. **Exported Tensors**:
    - Actions: `[numWorlds × numAgents × 4]`
    - Rewards: `[numWorlds × numAgents × 1]`
    - Multiple observation tensors
3. **Physics Data**: Collision geometry, rigid body metadata
4. **Render Buffers**: GPU memory for meshes, textures, outputs
