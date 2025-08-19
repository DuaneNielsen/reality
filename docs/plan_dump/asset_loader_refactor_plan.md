# Asset Loader Refactor Plan

## Files to Read Before Implementation

### Core Files to Understand Current System
1. **src/sim.hpp** - Contains SimObject enum to be removed
2. **src/types.hpp** - Contains CompiledLevel struct with tile_types to change to object_ids
3. **src/level_gen.cpp** - NVRTC-compiled level generation, contains makeCube/makeWall to remove
4. **src/mgr.cpp** - Contains loadPhysicsObjects and loadRenderObjects functions
5. **src/asset_descriptors.hpp/cpp** - Current asset descriptor system with SimObject dependencies

### Files to Understand Asset Flow
6. **external/madrona/include/madrona/physics.hpp** - ObjectManager structure definition
7. **external/madrona/src/physics/physics.cpp** - How ObjectData singleton is set
8. **src/sim.inl** - makeRenderableEntity template implementation

### Files to Understand Constraints
9. **src/CMakeLists.txt** - Which files are NVRTC-compiled (SIMULATOR_SRCS)
10. **src/consts.hpp** - Game constants that may need updating

## Implementation Plan

### Phase 1: Create AssetRegistry with Capability Tracking

1. **Create new files: asset_registry.hpp/cpp and asset_ids.hpp**
   
   **asset_ids.hpp** (NVRTC-safe header):
   ```cpp
   // This file is included by both CPU and GPU code
   namespace AssetIDs {
       // Fixed IDs for built-in assets - explicit values
       constexpr uint32_t INVALID = 0;  // 0 means no asset/empty
       constexpr uint32_t CUBE = 1;
       constexpr uint32_t WALL = 2;
       constexpr uint32_t AGENT = 3;
       constexpr uint32_t PLANE = 4;
       constexpr uint32_t AXIS_X = 5;
       constexpr uint32_t AXIS_Y = 6;
       constexpr uint32_t AXIS_Z = 7;
       
       // Dynamic assets start here
       constexpr uint32_t DYNAMIC_START = 32;
       constexpr uint32_t MAX_ASSETS = 256;
   }
   ```

   **asset_registry.hpp** (CPU-only):
   ```cpp
   class AssetRegistry {
       struct AssetInfo {
           std::string name;
           uint32_t id;
           bool hasPhysics;
           bool hasRender;
       };
       
       std::unordered_map<std::string, AssetInfo> nameToAsset;
       std::vector<AssetInfo> idToAsset;
       uint32_t nextId = AssetIDs::DYNAMIC_START;
       
   public:
       AssetRegistry();  // Initialize built-in assets
       
       uint32_t registerAsset(const std::string& name, 
                             bool hasPhysics, bool hasRender);
       uint32_t registerAssetWithId(const std::string& name, uint32_t id,
                                   bool hasPhysics, bool hasRender);
       
       uint32_t getAssetId(const std::string& name) const;
       bool assetHasPhysics(uint32_t id) const;
       bool assetHasRender(uint32_t id) const;
   };
   ```

2. **Initialize built-in assets in AssetRegistry constructor**
   ```cpp
   AssetRegistry::AssetRegistry() {
       // Ensure vector is sized for max assets
       idToAsset.resize(AssetIDs::MAX_ASSETS);
       nextId = AssetIDs::DYNAMIC_START;  // Dynamic assets start at 32
       
       // Register built-in assets at their explicit IDs
       registerAssetWithId("cube", AssetIDs::CUBE, true, true);
       registerAssetWithId("wall", AssetIDs::WALL, true, true);
       registerAssetWithId("agent", AssetIDs::AGENT, true, true);
       registerAssetWithId("plane", AssetIDs::PLANE, true, true);
       
       // Render-only assets (no physics)
       registerAssetWithId("axis_x", AssetIDs::AXIS_X, false, true);
       registerAssetWithId("axis_y", AssetIDs::AXIS_Y, false, true);
       registerAssetWithId("axis_z", AssetIDs::AXIS_Z, false, true);
   }
   ```

3. **Create C++ Unit Tests for AssetRegistry**
   
   **tests/cpp/test_asset_registry.cpp**:
   ```cpp
   #include <gtest/gtest.h>
   #include "asset_registry.hpp"
   #include "asset_ids.hpp"
   
   class AssetRegistryTest : public ::testing::Test {
   protected:
       AssetRegistry registry;
   };
   
   TEST_F(AssetRegistryTest, BuiltInAssetsInitialized) {
       // Verify built-in assets are registered at correct IDs
       EXPECT_EQ(registry.getAssetId("cube"), AssetIDs::CUBE);
       EXPECT_EQ(registry.getAssetId("wall"), AssetIDs::WALL);
       EXPECT_EQ(registry.getAssetId("agent"), AssetIDs::AGENT);
       EXPECT_EQ(registry.getAssetId("plane"), AssetIDs::PLANE);
       EXPECT_EQ(registry.getAssetId("axis_x"), AssetIDs::AXIS_X);
       EXPECT_EQ(registry.getAssetId("axis_y"), AssetIDs::AXIS_Y);
       EXPECT_EQ(registry.getAssetId("axis_z"), AssetIDs::AXIS_Z);
   }
   
   TEST_F(AssetRegistryTest, BuiltInAssetCapabilities) {
       // Verify physics assets have correct capabilities
       EXPECT_TRUE(registry.assetHasPhysics(AssetIDs::CUBE));
       EXPECT_TRUE(registry.assetHasRender(AssetIDs::CUBE));
       
       EXPECT_TRUE(registry.assetHasPhysics(AssetIDs::WALL));
       EXPECT_TRUE(registry.assetHasRender(AssetIDs::WALL));
       
       EXPECT_TRUE(registry.assetHasPhysics(AssetIDs::PLANE));
       EXPECT_TRUE(registry.assetHasRender(AssetIDs::PLANE));
       
       // Verify render-only assets
       EXPECT_FALSE(registry.assetHasPhysics(AssetIDs::AXIS_X));
       EXPECT_TRUE(registry.assetHasRender(AssetIDs::AXIS_X));
       
       EXPECT_FALSE(registry.assetHasPhysics(AssetIDs::AXIS_Y));
       EXPECT_TRUE(registry.assetHasRender(AssetIDs::AXIS_Y));
       
       EXPECT_FALSE(registry.assetHasPhysics(AssetIDs::AXIS_Z));
       EXPECT_TRUE(registry.assetHasRender(AssetIDs::AXIS_Z));
   }
   
   TEST_F(AssetRegistryTest, RegisterDynamicAsset) {
       // Register a new dynamic asset
       uint32_t customId = registry.registerAsset("custom_box", true, true);
       
       // Should get ID >= DYNAMIC_START
       EXPECT_GE(customId, AssetIDs::DYNAMIC_START);
       
       // Should be retrievable by name
       EXPECT_EQ(registry.getAssetId("custom_box"), customId);
       
       // Should have correct capabilities
       EXPECT_TRUE(registry.assetHasPhysics(customId));
       EXPECT_TRUE(registry.assetHasRender(customId));
   }
   
   TEST_F(AssetRegistryTest, RegisterMultipleDynamicAssets) {
       uint32_t id1 = registry.registerAsset("asset1", true, false);
       uint32_t id2 = registry.registerAsset("asset2", false, true);
       uint32_t id3 = registry.registerAsset("asset3", true, true);
       
       // IDs should be unique and sequential
       EXPECT_EQ(id1, AssetIDs::DYNAMIC_START);
       EXPECT_EQ(id2, AssetIDs::DYNAMIC_START + 1);
       EXPECT_EQ(id3, AssetIDs::DYNAMIC_START + 2);
       
       // Verify capabilities
       EXPECT_TRUE(registry.assetHasPhysics(id1));
       EXPECT_FALSE(registry.assetHasRender(id1));
       
       EXPECT_FALSE(registry.assetHasPhysics(id2));
       EXPECT_TRUE(registry.assetHasRender(id2));
       
       EXPECT_TRUE(registry.assetHasPhysics(id3));
       EXPECT_TRUE(registry.assetHasRender(id3));
   }
   
   TEST_F(AssetRegistryTest, RegisterWithSpecificId) {
       // Register asset at specific ID
       uint32_t specificId = 100;
       uint32_t returnedId = registry.registerAssetWithId("specific", specificId, true, false);
       
       EXPECT_EQ(returnedId, specificId);
       EXPECT_EQ(registry.getAssetId("specific"), specificId);
       EXPECT_TRUE(registry.assetHasPhysics(specificId));
       EXPECT_FALSE(registry.assetHasRender(specificId));
   }
   
   TEST_F(AssetRegistryTest, DuplicateNameHandling) {
       // Attempting to register duplicate name should return existing ID
       uint32_t id1 = registry.registerAsset("duplicate", true, true);
       uint32_t id2 = registry.registerAsset("duplicate", false, false);
       
       // Should return same ID (not create new one)
       EXPECT_EQ(id1, id2);
       
       // Capabilities should remain from first registration
       EXPECT_TRUE(registry.assetHasPhysics(id1));
       EXPECT_TRUE(registry.assetHasRender(id1));
   }
   
   TEST_F(AssetRegistryTest, InvalidAssetLookup) {
       // Looking up non-existent asset should throw or return invalid ID
       EXPECT_THROW(registry.getAssetId("nonexistent"), std::runtime_error);
       // OR if using optional/error codes:
       // EXPECT_EQ(registry.getAssetId("nonexistent"), AssetRegistry::INVALID_ID);
   }
   
   TEST_F(AssetRegistryTest, BoundaryConditions) {
       // Test querying capabilities for invalid IDs
       EXPECT_FALSE(registry.assetHasPhysics(999));
       EXPECT_FALSE(registry.assetHasRender(999));
       
       // Test empty asset name
       EXPECT_THROW(registry.registerAsset("", true, true), std::invalid_argument);
   }
   ```
   
   **Add to tests/cpp/CMakeLists.txt**:
   ```cmake
   # Add asset registry tests
   add_executable(test_asset_registry test_asset_registry.cpp)
   target_link_libraries(test_asset_registry 
       madrona_escape_room_mgr
       GTest::gtest_main
   )
   add_test(NAME AssetRegistryTest COMMAND test_asset_registry)
   ```

4. **Update loadPhysicsObjects in mgr.cpp**
   ```cpp
   static void loadPhysicsObjects(PhysicsLoader &loader, AssetRegistry &registry) {
       // Pre-allocate for maximum assets
       HeapArray<SourceCollisionObject> src_objs(AssetIDs::MAX_ASSETS);
       
       // Load file-based assets (cube, wall, agent)
       // ... existing .obj loading code ...
       
       // Create programmatic plane asset
       static SourceCollisionPrimitive plane_prim {
           .type = CollisionPrimitive::Type::Plane,
           .plane = {},
       };
       
       src_objs[AssetIDs::PLANE] = {
           .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
           .invMass = 0.f,
           .friction = { .muS = 0.5f, .muD = 0.4f },
       };
       
       // Note: Axis indicators (AXIS_X, AXIS_Y, AXIS_Z) are render-only
       // They don't need physics data, so src_objs[4,5,6] remain empty
       
       // Future: Load dynamic assets
       // for (auto& file : scan_asset_directory()) {
       //     uint32_t id = registry.registerAsset(file.name, true, true);
       //     src_objs[id] = load_asset(file);
       // }
       
       loader.loadRigidBodies(rigid_body_assets);
   }
   ```

### Phase 2: Update GPU Code and Level Generation

1. **Update sim.hpp**
   - Remove SimObject enum entirely
   - Include asset_ids.hpp instead

2. **Update types.hpp**
   ```cpp
   struct CompiledLevel {
       // Change from tile_types to object_ids
       int32_t object_ids[MAX_TILES];  // Was tile_types
       float tile_x[MAX_TILES];
       float tile_y[MAX_TILES];
       // ... rest unchanged
   };
   ```

3. **Update level_gen.cpp**
   ```cpp
   #include "asset_ids.hpp"  // NVRTC-safe constants
   
   // Remove makeCube and makeWall functions
   
   // Add generic entity creator
   Entity createEntityFromId(Engine& ctx, uint32_t objectId, 
                           Vector3 pos, Quat rot, Diag3x3 scale) {
       // Check for render-only assets
       if (objectId == AssetIDs::AXIS_X || 
           objectId == AssetIDs::AXIS_Y || 
           objectId == AssetIDs::AXIS_Z) {
           Entity e = ctx.makeRenderableEntity<RenderOnlyEntity>();
           ctx.get<Position>(e) = pos;
           ctx.get<Rotation>(e) = rot;
           ctx.get<Scale>(e) = scale;
           ctx.get<ObjectID>(e) = ObjectID{objectId};
           return e;
       } else {
           // Physics entity
           Entity e = ctx.makeRenderableEntity<PhysicsEntity>();
           setupRigidBodyEntity(ctx, e, pos, rot, ObjectID{objectId},
                              EntityType::None, ResponseType::Static, scale);
           registerRigidBodyEntity(ctx, e, objectId);
           return e;
       }
   }
   
   // Specialized setup functions using compile-time constants
   void setupGroundPlane(Engine& ctx) {
       Entity plane = ctx.makeRenderableEntity<PhysicsEntity>();
       setupRigidBodyEntity(ctx, plane, Vector3{0,0,0}, 
                           Quat{1,0,0,0}, ObjectID{AssetIDs::PLANE},
                           EntityType::None, ResponseType::Static);
       registerRigidBodyEntity(ctx, plane, AssetIDs::PLANE);
       ctx.data().floorPlane = plane;
   }
   
   void setupAxisIndicators(Engine& ctx) {
       // X axis - render only!
       Entity axisX = ctx.makeRenderableEntity<RenderOnlyEntity>();
       ctx.get<Position>(axisX) = Vector3{2, 0, 0};
       ctx.get<Rotation>(axisX) = Quat{1, 0, 0, 0};
       ctx.get<Scale>(axisX) = Diag3x3{4, 0.2, 0.2};
       ctx.get<ObjectID>(axisX) = ObjectID{AssetIDs::AXIS_X};
       ctx.data().originMarkerBoxes[0] = axisX;
       
       // Similar for Y and Z axes
   }
   
   void setupAgent(Engine& ctx, Vector3 spawnPos, float facing, int agentIdx) {
       Entity agent = ctx.data().agents[agentIdx];
       setupRigidBodyEntity(ctx, agent, spawnPos,
                           Quat::angleAxis(facing, math::up),
                           ObjectID{AssetIDs::AGENT},
                           EntityType::Agent, ResponseType::Dynamic);
       registerRigidBodyEntity(ctx, agent, AssetIDs::AGENT);
       // Camera attachment, progress tracking, etc.
   }
   
   static void generateFromCompiled(Engine &ctx, CompiledLevel* level) {
       LevelState &level_state = ctx.singleton<LevelState>();
       CountT entity_count = 0;
       
       float tile_scale = level->scale;
       
       // Use object_ids instead of tile_types
       for (int32_t i = 0; i < level->num_tiles; i++) {
           uint32_t objectId = level->object_ids[i];
           float x = level->tile_x[i];
           float y = level->tile_y[i];
           
           Entity entity = Entity::none();
           
           // Create entity based on object ID
           if (objectId != 0) {  // 0 means empty
               // Determine scale based on object type
               Diag3x3 scale = {tile_scale, tile_scale, tile_scale};
               if (objectId == AssetIDs::WALL) {
                   scale.z = 2.0f;  // Walls are taller
               } else if (objectId == AssetIDs::CUBE) {
                   float s = tile_scale * 0.5f;
                   scale = {s, s, s};
               }
               
               entity = createEntityFromId(ctx, objectId, 
                                          Vector3{x, y, 0}, 
                                          Quat{1,0,0,0}, scale);
           }
           
           if (entity != Entity::none()) {
               level_state.rooms[0].entities[entity_count++] = entity;
           }
       }
       
       // Fill remaining with none
       for (CountT i = entity_count; i < CompiledLevel::MAX_TILES; i++) {
           level_state.rooms[0].entities[i] = Entity::none();
       }
   }
   
   void generateWorld(Engine &ctx) {
       // Create persistent entities first
       createPersistentEntities(ctx);  // Creates agent entities
       
       // Set up core world objects
       setupGroundPlane(ctx);
       setupAxisIndicators(ctx);
       
       // Reset agent positions from spawn data
       resetPersistentEntities(ctx);
       
       // Generate level tiles
       CompiledLevel& level = ctx.singleton<CompiledLevel>();
       generateFromCompiled(ctx, &level);
   }
   ```

4. **Update asset_descriptors.hpp/cpp**
   - Remove SimObject dependencies
   - Change objectId field to use AssetIDs constants
   - Add capability flags (hasPhysics, hasRender) to descriptors

5. **Update Python bindings**
   - Expose AssetRegistry to Python
   - Update level compiler to use object_ids instead of tile_types
   - Add registry.getAssetId() calls in Python level creation

### Key Benefits

- **Simpler Code**: No backward compatibility constraints
- **Unified System**: All assets go through same registration
- **No SimObject Enum**: Completely removed
- **Dynamic Loading Ready**: Can add new assets without code changes
- **Capability Aware**: Registry tracks physics vs render-only
- **NVRTC Compatible**: GPU code uses compile-time constants
- **Sequential IDs**: Clean, simple ID assignment starting from 1

### Testing Strategy

1. **Hardcode test CompiledLevel in mgr.cpp**:
   ```cpp
   CompiledLevel testLevel;
   testLevel.object_ids[0] = AssetIDs::WALL;  // 1
   testLevel.object_ids[1] = AssetIDs::CUBE;  // 0
   testLevel.object_ids[2] = AssetIDs::AXIS_X; // 4 (render-only)
   testLevel.tile_x[0] = -5; testLevel.tile_y[0] = 0;
   testLevel.tile_x[1] = 0;  testLevel.tile_y[1] = 0;
   testLevel.tile_x[2] = 5;  testLevel.tile_y[2] = 0;
   testLevel.num_tiles = 3;
   ```

2. **Verify**:
   - Wall and Cube create PhysicsEntity with collision
   - Axis_X creates RenderOnlyEntity without physics
   - Ground plane and axis indicators appear correctly
   - Agents spawn at correct positions

### Files to Modify Summary

1. **New Files**:
   - src/asset_registry.hpp
   - src/asset_registry.cpp
   - src/asset_ids.hpp

2. **Modified Files**:
   - src/sim.hpp (remove SimObject enum)
   - src/types.hpp (tile_types → object_ids)
   - src/mgr.cpp (add AssetRegistry, update loading)
   - src/level_gen.cpp (remove makeCube/makeWall, add generic creator)
   - src/asset_descriptors.hpp/cpp (remove SimObject deps)
   - src/madrona_escape_room_c_api.cpp (update for object_ids)
   - Python bindings (update level compiler)