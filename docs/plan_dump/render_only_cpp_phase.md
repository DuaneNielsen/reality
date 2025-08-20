# Phase 1: C++ Components and Tests for Render-Only Assets

## Required Reading

Before starting this implementation, read the following files in order:

### Core Architecture
1. **`src/types.hpp`** - Study the `CompiledLevel` struct definition and `RenderOnlyEntity` archetype
2. **`src/level_gen.cpp`** - Understand current entity creation logic, especially:
   - `createEntityShell()` function (lines 68-76)
   - `setupRenderOnlyEntity()` vs `setupEntityPhysics()` functions  
   - `generateFromCompiled()` and `resetPersistentEntities()` functions
3. **`include/madrona_escape_room_c_api.h`** - Review `MER_CompiledLevel` struct (lines 78-98)

### Test Infrastructure  
4. **`tests/cpp/fixtures/test_levels.hpp`** - Critical: embedded test level data format
5. **`tests/cpp/unit/test_persistence.cpp`** - Understand persistence testing patterns
6. **`tests/cpp/unit/test_c_api_cpu.cpp`** - C API testing patterns

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
    bool tile_render_only[MAX_TILES]; // NEW: Whether tile is render-only (no physics)
};
```

#### 1.2 Update C API struct in `include/madrona_escape_room_c_api.h`
```cpp
typedef struct {
    // ... existing fields ...
    
    // Tile data arrays
    int32_t object_ids[MER_MAX_TILES];
    float tile_x[MER_MAX_TILES];  
    float tile_y[MER_MAX_TILES];
    bool tile_persistent[MER_MAX_TILES];
    bool tile_render_only[MER_MAX_TILES]; // NEW: Render-only flag array
} MER_CompiledLevel;
```

### Step 2: Update Entity Creation Logic

#### 2.1 Modify `src/level_gen.cpp`

**Replace hardcoded asset ID checks:**
```cpp
// OLD: In createEntityShell() - lines 68-76
static Entity createEntityShell(Engine& ctx, uint32_t objectId) {
    if (objectId == AssetIDs::AXIS_X || 
        objectId == AssetIDs::AXIS_Y || 
        objectId == AssetIDs::AXIS_Z) {
        return ctx.makeRenderableEntity<RenderOnlyEntity>();
    } else {
        return ctx.makeRenderableEntity<PhysicsEntity>();
    }
}

// NEW: Data-driven approach
static Entity createEntityShell(Engine& ctx, uint32_t objectId, bool isRenderOnly) {
    if (isRenderOnly) {
        return ctx.makeRenderableEntity<RenderOnlyEntity>();
    } else {
        return ctx.makeRenderableEntity<PhysicsEntity>();
    }
}
```

**Update generateFromCompiled() - around line 390:**
```cpp
// OLD: entity = createEntityShell(ctx, objectId);
// NEW: 
bool isRenderOnly = level->tile_render_only[i];
entity = createEntityShell(ctx, objectId, isRenderOnly);

// Update setup calls:
if (isRenderOnly) {
    setupRenderOnlyEntity(ctx, entity, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale);
} else {
    setupEntityPhysics(ctx, entity, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale);
}
```

**Update resetPersistentEntities() - around line 270:**
```cpp
bool isRenderOnly = level.tile_render_only[i];
if (isRenderOnly) {
    setupRenderOnlyEntity(ctx, e, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale);
} else {
    setupEntityPhysics(ctx, e, objectId, Vector3{x, y, z}, Quat{1,0,0,0}, scale);
}
```

#### 2.2 Update C API implementation in `src/madrona_escape_room_c_api.cpp`

Add field copying in `mer_create_manager()` around line 100:
```cpp
// After copying tile_persistent array
std::memcpy(cpp_level.tile_render_only, c_level->tile_render_only, 
           sizeof(cpp_level.tile_render_only));
```

### Step 3: Update Test Infrastructure

#### 3.1 Fix Embedded Test Data in `tests/cpp/fixtures/test_levels.hpp`

**Critical**: Update `TestLevelHelper::GetEmbeddedTestLevel()`:
```cpp
static MER_CompiledLevel GetEmbeddedTestLevel() {
    MER_CompiledLevel level = {};  // Zero-initialize
    
    // ... existing initialization ...
    
    // Initialize render-only flags for test tiles
    for (int i = 0; i < level.num_tiles; i++) {
        level.tile_render_only[i] = false;  // Default: all physics entities
    }
    
    return level;
}
```

#### 3.2 Update Persistence Tests in `tests/cpp/unit/test_persistence.cpp`

Add render-only flag initialization in `SetUp()` method:
```cpp
// After setting tile_persistent flags
level.tile_render_only[0] = false;  // Wall - physics
level.tile_render_only[1] = false;  // Wall - physics  
level.tile_render_only[2] = false;  // Cube - physics
level.tile_render_only[3] = false;  // Cube - physics
level.tile_render_only[4] = true;   // AXIS_X - render-only
level.tile_render_only[5] = true;   // AXIS_Y - render-only

// Initialize rest as false (physics entities)
for (int i = 6; i < CompiledLevel::MAX_TILES; i++) {
    level.tile_render_only[i] = false;
}
```

#### 3.3 Add New Test Cases

Add to `tests/cpp/unit/test_persistence.cpp`:
```cpp
TEST_F(PersistenceTest, RenderOnlyEntitiesCreated) {
    mgr->step();
    
    // Verify render-only entities don't participate in physics
    // Test passes if simulation runs without physics conflicts
    for (int step = 0; step < 50; step++) {
        mgr->step();
    }
}

TEST_F(PersistenceTest, MixedPhysicsRenderOnlyLevel) {
    // Test level with both physics and render-only entities
    mgr->step();
    
    int renderOnlyCount = 0;
    int physicsCount = 0;
    for (int i = 0; i < testLevels[0]->num_tiles; i++) {
        if (testLevels[0]->tile_render_only[i]) {
            renderOnlyCount++;
        } else {
            physicsCount++;
        }
    }
    EXPECT_EQ(renderOnlyCount, 2);  // 2 axis markers
    EXPECT_EQ(physicsCount, 4);     // 2 walls + 2 cubes
}
```

### Step 4: Update Level Utilities Tests

#### 4.1 Update `tests/cpp/unit/test_level_utilities.cpp`

Update struct size validation:
```cpp
TEST_F(LevelUtilitiesTest, LevelFileSizeMatchesStructSize) {
    // File size should include new render-only boolean array
    size_t expectedSize = sizeof(MER_CompiledLevel);
    EXPECT_EQ(file_size, expectedSize);
}
```

Update level comparison in `LevelComparer::compareLevels()`:
```cpp
// Add render-only array comparison
for (int i = 0; i < level1.num_tiles; i++) {
    if (level1.tile_render_only[i] != level2.tile_render_only[i]) {
        return false;
    }
}
```

### Step 5: Update C API Tests

#### 5.1 Update `tests/cpp/unit/test_c_api_cpu.cpp`

Add render-only validation to tensor tests:
```cpp
TEST_F(CApiCPUTest, RenderOnlyEntitiesInLevel) {
    // Create level with render-only entities
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    
    // Mark some entities as render-only
    levels[0].tile_render_only[0] = true;  // First tile render-only
    
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    // Should create manager without issues
    EXPECT_NE(handle, nullptr);
}
```

## Validation Criteria

### Phase 1 Complete When:

1. **All C++ tests pass:**
   ```bash
   ./tests/run_cpp_tests.sh
   ```

2. **CompiledLevel struct tests pass specifically:**
   - `test_persistence.cpp` - All persistence tests
   - `test_level_utilities.cpp` - Level loading and comparison
   - `test_c_api_cpu.cpp` - C API integration

3. **Manager creation works with render-only entities:**
   - Can create managers with mixed physics/render-only levels
   - Simulation runs without crashes
   - Entity creation logic uses data-driven approach instead of hardcoded asset IDs

4. **Memory layout is correct:**
   - C++ and C API struct sizes match
   - All boolean arrays properly initialized
   - No memory corruption in tests

## Implementation Notes

- **Critical**: Always initialize the `tile_render_only` array to `false` by default
- Update embedded test data first - this affects all C++ tests
- Test with various combinations: all physics, all render-only, mixed
- Ensure backward compatibility: missing render-only flags should default to `false`

## Testing Strategy

1. Start with `test_persistence.cpp` - it has the most comprehensive level setup
2. Fix embedded test data in `test_levels.hpp` 
3. Update C API tests last - they depend on working struct definitions
4. Run tests frequently during implementation to catch issues early

This phase establishes the C++ foundation for render-only assets without touching the Python level compiler.