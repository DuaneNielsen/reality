# Phase 2: Convert EntityType from Hardcoded Enum to Data-Driven Integer Values

## Required Reading

Before starting this implementation, read the following files in order to understand the current EntityType implementation:

### Core Architecture and Current Implementation
1. **`src/types.hpp`** - Study the `EntityType` enum definition (lines 114-120) and its usage in archetypes
2. **`src/level_gen.cpp`** - Understand current hardcoded EntityType assignment logic, especially:
   - `setupRigidBodyEntity()` function (lines 24-48) - Takes EntityType parameter
   - `setupEntityPhysics()` function (lines 80-94) - Contains hardcoded asset ID to EntityType mapping
   - `createFloorPlane()` function (lines 114-125) - Sets EntityType::None for floor
   - Agent initialization in `createPersistentEntities()` (line 193) - Sets EntityType::Agent
3. **`src/sim.cpp`** - Review EntityType component registration (line 48)
4. **`include/madrona_escape_room_c_api.h`** - Review current `MER_CompiledLevel` struct (lines 93-99)

### Test Infrastructure (from Phase 1)
5. **`tests/cpp/fixtures/test_levels.hpp`** - Understand test level creation patterns
6. **`tests/cpp/unit/test_persistence.cpp`** - Understand persistence testing patterns 
7. **`tests/cpp/fixtures/viewer_test_base.hpp`** - LevelComparer and test level creation utilities
8. **`tests/cpp/unit/test_c_api_cpu.cpp`** - C API testing patterns

### Understanding EntityType Usage Context
9. **`docs/architecture/COLLISION_SYSTEM.md`** - Understand how EntityType is used in collision detection and game logic
10. **Previous Phase 1 implementation** - Review render-only assets pattern for data-driven approach

## Implementation Plan

### Step 1: Update C++ Struct Definitions

#### 1.1 Update CompiledLevel in `src/types.hpp`
```cpp
struct CompiledLevel {
    // ... existing fields ...
    
    // Tile data arrays (packed for GPU efficiency)
    int32_t object_ids[MAX_TILES];    // Asset ID for each tile
    float tile_x[MAX_TILES];          // World X position
    float tile_y[MAX_TILES];          // World Y position  
    bool tile_persistent[MAX_TILES];  // Whether tile persists across episodes
    bool tile_render_only[MAX_TILES]; // Whether tile is render-only (no physics)
    int32_t tile_entity_type[MAX_TILES]; // NEW: EntityType value for each tile (0=None, 1=Cube, 2=Wall, 3=Agent)
};
```

**EntityType Value Mapping:**
- `0` = `EntityType::None` (default/unset)
- `1` = `EntityType::Cube` 
- `2` = `EntityType::Wall`
- `3` = `EntityType::Agent`
- `4` = (reserved for future types)

#### 1.2 Update C API struct in `include/madrona_escape_room_c_api.h`
```cpp
typedef struct {
    // ... existing fields ...
    
    // Tile data arrays
    int32_t object_ids[MER_MAX_TILES];
    float tile_x[MER_MAX_TILES];  
    float tile_y[MER_MAX_TILES];
    bool tile_persistent[MER_MAX_TILES];
    bool tile_render_only[MER_MAX_TILES];
    int32_t tile_entity_type[MER_MAX_TILES]; // NEW: EntityType integer values
} MER_CompiledLevel;
```

### Step 2: Update Entity Creation Logic

#### 2.1 Modify `src/level_gen.cpp`

**Replace hardcoded EntityType assignment in `setupEntityPhysics()`:**
```cpp
// OLD: Lines 82-89 in setupEntityPhysics()
static void setupEntityPhysics(Engine& ctx, Entity e, uint32_t objectId,
                              Vector3 pos, Quat rot, Diag3x3 scale) {
    EntityType entityType = EntityType::None;
    if (objectId == AssetIDs::WALL) {
        entityType = EntityType::Wall;
    } else if (objectId == AssetIDs::CUBE) {
        entityType = EntityType::Cube;
    } else if (objectId == AssetIDs::AGENT) {
        entityType = EntityType::Agent;
    }
    
    setupRigidBodyEntity(ctx, e, pos, rot, objectId,
                       entityType, ResponseType::Static, scale);
    registerRigidBodyEntity(ctx, e, objectId);
}

// NEW: Data-driven approach
static void setupEntityPhysics(Engine& ctx, Entity e, uint32_t objectId,
                              Vector3 pos, Quat rot, Diag3x3 scale, int32_t entityTypeValue) {
    EntityType entityType = static_cast<EntityType>(entityTypeValue);
    
    setupRigidBodyEntity(ctx, e, pos, rot, objectId,
                       entityType, ResponseType::Static, scale);
    registerRigidBodyEntity(ctx, e, objectId);
}
```

**Update calls to `setupEntityPhysics()` in `resetPersistentEntities()` - around line 276:**
```cpp
// OLD: setupEntityPhysics(ctx, e, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale);
// NEW: 
int32_t entityTypeValue = level.tile_entity_type[i];
setupEntityPhysics(ctx, e, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale, entityTypeValue);
```

**Update calls to `setupEntityPhysics()` in `generateFromCompiled()` - around line 396:**
```cpp
// OLD: setupEntityPhysics(ctx, entity, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale);
// NEW:
int32_t entityTypeValue = level->tile_entity_type[i];
setupEntityPhysics(ctx, entity, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale, entityTypeValue);
```

**Keep hardcoded EntityType for special cases that aren't level tiles:**
- Floor plane in `createFloorPlane()` - keep `EntityType::None`
- Agents in `createPersistentEntities()` - keep `EntityType::Agent` 
- Origin marker gizmos - these don't use EntityType (render-only)

#### 2.2 Update C API implementation in `src/madrona_escape_room_c_api.cpp`

Add field copying in `mer_create_manager()` around line 121:
```cpp
// After copying tile_render_only array
std::memcpy(cpp_level.tile_entity_type, c_level->tile_entity_type, 
           sizeof(int32_t) * array_size);
```

### Step 3: Update Test Infrastructure

#### 3.1 Fix Embedded Test Data in `tests/cpp/fixtures/test_levels.hpp`

**Critical**: Update `TestLevelHelper::GetEmbeddedTestLevel()`:
```cpp
static MER_CompiledLevel GetEmbeddedTestLevel() {
    MER_CompiledLevel level = {};  // Zero-initialize
    
    // ... existing initialization ...
    
    // Initialize entity type values for test tiles
    for (int i = 0; i < level.num_tiles; i++) {
        // Default entity type based on object_id pattern
        if (level.object_ids[i] == 1) {  // CUBE
            level.tile_entity_type[i] = 1;  // EntityType::Cube
        } else if (level.object_ids[i] == 2) {  // WALL  
            level.tile_entity_type[i] = 2;  // EntityType::Wall
        } else {
            level.tile_entity_type[i] = 0;  // EntityType::None
        }
    }
    
    return level;
}
```

#### 3.2 Update Persistence Tests in `tests/cpp/unit/test_persistence.cpp`

Add entity type initialization in `SetUp()` method:
```cpp
// After setting tile_render_only flags
level.tile_entity_type[0] = 2;  // Wall - EntityType::Wall
level.tile_entity_type[1] = 2;  // Wall - EntityType::Wall  
level.tile_entity_type[2] = 1;  // Cube - EntityType::Cube
level.tile_entity_type[3] = 1;  // Cube - EntityType::Cube
level.tile_entity_type[4] = 0;  // AXIS_X - EntityType::None (render-only)
level.tile_entity_type[5] = 0;  // AXIS_Y - EntityType::None (render-only)

// Initialize rest as None (default)
for (int i = 6; i < CompiledLevel::MAX_TILES; i++) {
    level.tile_entity_type[i] = 0;  // EntityType::None
}
```

#### 3.3 Add New Test Cases

Add to `tests/cpp/unit/test_persistence.cpp`:
```cpp
TEST_F(PersistenceTest, DataDrivenEntityTypesAssigned) {
    mgr->step();
    
    // Verify entity types are set from level data, not hardcoded logic
    // Test passes if simulation runs without entity type conflicts
    for (int step = 0; step < 50; step++) {
        mgr->step();
    }
}

TEST_F(PersistenceTest, MixedEntityTypesLevel) {
    // Test level with different entity types
    mgr->step();
    
    int wallCount = 0;
    int cubeCount = 0; 
    int noneCount = 0;
    for (int i = 0; i < testLevels[0]->num_tiles; i++) {
        int32_t entityType = testLevels[0]->tile_entity_type[i];
        if (entityType == 1) cubeCount++;       // Cube
        else if (entityType == 2) wallCount++;  // Wall
        else noneCount++;                        // None
    }
    EXPECT_EQ(wallCount, 2);   // 2 walls
    EXPECT_EQ(cubeCount, 2);   // 2 cubes  
    EXPECT_EQ(noneCount, 2);   // 2 axis markers (render-only)
}
```

### Step 4: Update Level Utilities Tests

#### 4.1 Update `tests/cpp/fixtures/viewer_test_base.hpp`

Update level comparison in `LevelComparer::compareLevels()`:
```cpp
// Add entity type array comparison
for (int32_t i = 0; i < level1.num_tiles; i++) {
    if (level1.object_ids[i] != level2.object_ids[i] ||
        level1.tile_x[i] != level2.tile_x[i] ||
        level1.tile_y[i] != level2.tile_y[i] ||
        level1.tile_persistent[i] != level2.tile_persistent[i] ||
        level1.tile_render_only[i] != level2.tile_render_only[i] ||
        level1.tile_entity_type[i] != level2.tile_entity_type[i]) {  // NEW
        return false;
    }
}
```

Update `createTestLevelFile()` method:
```cpp
// Fill with simple tile pattern
for (int32_t i = 0; i < level.num_tiles && i < 1024; i++) {
    level.object_ids[i] = (i % 2) + 1;  // Cycle between CUBE (1) and WALL (2)
    level.tile_x[i] = (i % width) * level.scale;
    level.tile_y[i] = (i / width) * level.scale;
    level.tile_persistent[i] = false;  // Default: non-persistent
    level.tile_render_only[i] = false;  // Default: physics entities
    level.tile_entity_type[i] = (i % 2) + 1;  // NEW: Match object_ids (1=Cube, 2=Wall)
}
```

Update `createTestRecordingFile()` method:
```cpp
// Initialize tile data
for (int32_t i = 0; i < level.num_tiles && i < 1024; i++) {
    level.object_ids[i] = 0;
    level.tile_x[i] = (i % 16) * level.scale;
    level.tile_y[i] = (i / 16) * level.scale;
    level.tile_persistent[i] = false;  // Default: non-persistent
    level.tile_render_only[i] = false;  // Default: physics entities
    level.tile_entity_type[i] = 0;  // NEW: Default EntityType::None
}
```

### Step 5: Update C API Tests

#### 5.1 Update `tests/cpp/unit/test_c_api_cpu.cpp`

Add entity type validation to existing tests:
```cpp
TEST_F(CApiCPUTest, DataDrivenEntityTypesInLevel) {
    // Create level with data-driven entity types
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    
    // Set specific entity types
    levels[0].tile_entity_type[0] = 2;  // Wall
    levels[0].tile_entity_type[1] = 1;  // Cube
    
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    // Should create manager without issues
    EXPECT_NE(handle, nullptr);
    
    // Run a few steps to verify no crashes with data-driven entity types
    for (int i = 0; i < 5; i++) {
        MER_Result result = mer_step(handle);
        EXPECT_EQ(result, MER_SUCCESS);
    }
}
```

## Validation Criteria

### Phase 2 Complete When:

1. **All C++ tests pass:**
   ```bash
   ./tests/run_cpp_tests.sh
   ```

2. **CompiledLevel struct tests pass specifically:**
   - `test_persistence.cpp` - All persistence tests including new entity type tests
   - `test_level_utilities.cpp` - Level loading and comparison with entity type arrays
   - `test_c_api_cpu.cpp` - C API integration with data-driven entity types

3. **Manager creation works with data-driven entity types:**
   - Can create managers with entity types set from level data
   - Simulation runs without crashes
   - Entity type assignment uses data-driven approach instead of hardcoded asset ID mapping

4. **Memory layout is correct:**
   - C++ and C API struct sizes match with new int32_t array
   - All integer arrays properly initialized
   - No memory corruption in tests

5. **Backward compatibility maintained:**
   - EntityType enum still exists and is used internally
   - Default entity type value `0` maps to `EntityType::None`
   - Special entities (floor, agents, origin markers) retain hardcoded types

## Implementation Notes

- **Critical**: Always initialize the `tile_entity_type` array to `0` (EntityType::None) by default
- **Keep EntityType enum**: The enum still exists for internal usage, we're just making the assignment data-driven
- **Special entities**: Floor plane, agents, and origin markers keep hardcoded EntityType assignment since they're not level tiles
- **Value mapping**: Use consistent integer-to-enum mapping across all code
- **Test with various combinations**: None, Wall, Cube, mixed types
- **Ensure backward compatibility**: Missing entity type values should default to `0` (None)

## Testing Strategy

1. Start with `test_persistence.cpp` - it has the most comprehensive level setup
2. Fix embedded test data in `test_levels.hpp` 
3. Update C API tests last - they depend on working struct definitions
4. Run tests frequently during implementation to catch issues early
5. Test data-driven behavior vs. previous hardcoded behavior

This phase establishes data-driven entity type assignment from level data while maintaining the existing EntityType enum for internal usage and collision detection logic.