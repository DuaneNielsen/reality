# Plan: Add Persistence Flag to Compiled Level System

## Overview
Add a persistence flag to the compiled level system to differentiate between entities that persist across episodes (performance optimization) vs entities that are recreated each episode.

## Reading List
Files to understand before implementing:
- **src/types.hpp** - SimContext and CompiledLevel structures
- **src/level_gen.cpp** - Entity creation and reset logic
- **src/sim.cpp** - Initialization flow and episode reset
- **include/madrona_escape_room_c_api.h** - C API CompiledLevel structure
- **src/madrona_escape_room_c_api.cpp** - C to C++ struct conversion

## Phase 1: Update Data Structures

### 1.1 Modify CompiledLevel struct (types.hpp)
- Add `bool tile_persistent[MAX_TILES]` array
- This marks which tiles should persist across episodes

### 1.2 Update MER_CompiledLevel (C API)
- Add matching `bool tile_persistent[MER_MAX_TILES]` to C API structure
- Ensure proper conversion in madrona_escape_room_c_api.cpp

### 1.3 Update SimContext (types.hpp)
- Add `Entity persistentLevelEntities[MAX_PERSISTENT_TILES]`
- Add `CountT numPersistentLevelEntities`
- These store references to persistent tiles (like agents/floor)

## Phase 2: Modify Level Generation

### 2.1 Update createPersistentEntities()
After creating floor/agents/gizmo:
```cpp
// Scan compiled level for persistent tiles
CompiledLevel& level = ctx.singleton<CompiledLevel>();
CountT persistentCount = 0;

for (int32_t i = 0; i < level.num_tiles; i++) {
    if (level.tile_persistent[i]) {
        // Create persistent entity
        Entity e = createEntityFromId(ctx, 
            level.object_ids[i],
            Vector3{level.tile_x[i], level.tile_y[i], 0},
            Quat{1,0,0,0},
            Diag3x3{1,1,1}  // Or appropriate scale
        );
        
        // Store reference
        ctx.data().persistentLevelEntities[persistentCount++] = e;
    }
}
ctx.data().numPersistentLevelEntities = persistentCount;
```

### 2.2 Update resetPersistentEntities()
Add after agent registration:
```cpp
// Re-register persistent level entities
for (CountT i = 0; i < ctx.data().numPersistentLevelEntities; i++) {
    Entity e = ctx.data().persistentLevelEntities[i];
    uint32_t objectId = ctx.get<ObjectID>(e).idx;
    registerRigidBodyEntity(ctx, e, objectId);
}
```

### 2.3 Update generateFromCompiled()
Modify tile generation loop:
```cpp
for (int32_t i = 0; i < level->num_tiles && entity_count < CompiledLevel::MAX_TILES; i++) {
    // Skip persistent tiles - they already exist
    if (level->tile_persistent[i]) {
        continue;
    }
    
    // Create non-persistent entity as usual
    uint32_t objectId = level->object_ids[i];
    // ... rest of entity creation code
}
```

## Benefits

### Performance
- Persistent walls/boundaries created once at startup
- No recreation overhead each episode
- Faster episode resets

### Memory
- Less allocation/deallocation per episode
- Reduced memory fragmentation
- More predictable memory usage

### Flexibility
- Levels can mix static boundaries with dynamic obstacles
- Persistent decorative elements (that don't change)
- Dynamic puzzle elements (that reset each episode)

### Compatibility
- Non-breaking change (persistence defaults to false)
- Existing levels work without modification
- Opt-in performance optimization

## Example Use Cases

1. **Maze with moving obstacles**:
   - Outer walls: persistent (never change)
   - Inner obstacles: non-persistent (randomized each episode)

2. **Training room**:
   - Room boundaries: persistent
   - Training targets: non-persistent (reset positions)

3. **Puzzle room**:
   - Walls and floor: persistent
   - Puzzle pieces: non-persistent (reset to start state)

## Testing Strategy

### C++ Unit Tests (tests/cpp/unit/test_persistence.cpp)

Create a new test fixture `PersistenceTest` that extends `MadronaCppTestBase`:

```cpp
class PersistenceTest : public MadronaCppTestBase {
protected:
    void SetUp() override {
        // Create test level with mix of persistent and non-persistent tiles
        CompiledLevel level {};
        level.width = 16;
        level.height = 16;
        level.scale = 1.0f;
        level.num_tiles = 6;
        level.max_entities = level.num_tiles + 6 + 30;
        
        // Persistent walls (boundaries)
        level.object_ids[0] = AssetIDs::WALL;
        level.tile_x[0] = -8.0f;
        level.tile_y[0] = 0.0f;
        level.tile_persistent[0] = true;  // PERSISTENT
        
        level.object_ids[1] = AssetIDs::WALL;
        level.tile_x[1] = 8.0f;
        level.tile_y[1] = 0.0f;
        level.tile_persistent[1] = true;  // PERSISTENT
        
        // Non-persistent obstacles
        level.object_ids[2] = AssetIDs::CUBE;
        level.tile_x[2] = -2.0f;
        level.tile_y[2] = 0.0f;
        level.tile_persistent[2] = false;  // NON-PERSISTENT
        
        level.object_ids[3] = AssetIDs::CUBE;
        level.tile_x[3] = 2.0f;
        level.tile_y[3] = 0.0f;
        level.tile_persistent[3] = false;  // NON-PERSISTENT
        
        // Persistent decorative elements
        level.object_ids[4] = AssetIDs::AXIS_X;
        level.tile_x[4] = 0.0f;
        level.tile_y[4] = 5.0f;
        level.tile_persistent[4] = true;  // PERSISTENT (visual marker)
        
        level.object_ids[5] = AssetIDs::AXIS_Y;
        level.tile_x[5] = 0.0f;
        level.tile_y[5] = -5.0f;
        level.tile_persistent[5] = true;  // PERSISTENT (visual marker)
        
        // Initialize rest as empty
        for (int i = 6; i < CompiledLevel::MAX_TILES; i++) {
            level.object_ids[i] = 0;
            level.tile_persistent[i] = false;
        }
    }
};
```

#### Test Cases:

1. **TEST_F(PersistenceTest, PersistentEntitiesCreatedOnce)**
   - Create Manager
   - Store entity IDs from first frame
   - Step simulation multiple episodes
   - Verify persistent entity IDs remain the same
   - Verify non-persistent entity IDs change

2. **TEST_F(PersistenceTest, PersistentEntitiesReregistered)**
   - Create Manager
   - Step to trigger episode reset
   - Verify persistent entities are in broadphase
   - Verify physics registration is valid

3. **TEST_F(PersistenceTest, NonPersistentEntitiesRecreated)**
   - Create Manager
   - Store non-persistent entity positions
   - Trigger episode reset
   - Verify non-persistent entities are at initial positions
   - Verify entity count matches expected

4. **TEST_F(PersistenceTest, MixedPersistenceLevel)**
   - Create Manager with mixed level
   - Verify total entity count
   - Verify persistent entity count in SimContext
   - Verify dynamic entity count in LevelState

5. **TEST_F(PersistenceTest, AllPersistentLevel)**
   - Create level with all tiles marked persistent
   - Verify all entities created in createPersistentEntities
   - Verify generateFromCompiled creates no entities

6. **TEST_F(PersistenceTest, NoPersistentLevel)**
   - Create level with no persistent tiles (current behavior)
   - Verify backward compatibility
   - Verify all entities created each episode

7. **TEST_F(PersistenceTest, PersistencePerformance)**
   - Measure time for episode reset with persistent entities
   - Measure time for episode reset without persistent entities
   - Verify persistent approach is faster

### Integration with Existing Tests

Update existing tests that may be affected:
- **AssetRefactorTest**: Add persistence flags (default false for compatibility)
- **LevelUtilitiesTest**: Update level creation to include persistence array
- **test_level_utilities.cpp**: Verify persistence flags serialize/deserialize correctly

### Running Tests

Add to build and test workflow:
```bash
# Build the project
./scripts/build.sh  # or use project-builder agent

# Run all C++ tests
./build/mad_escape_tests

# Run only persistence tests
./build/mad_escape_tests --gtest_filter="PersistenceTest.*"

# Run with verbose output
./build/mad_escape_tests --gtest_filter="PersistenceTest.*" --gtest_print_time=1
```

### Success Criteria
1. All new persistence tests pass
2. All existing tests still pass (backward compatibility)
3. No memory leaks (run with valgrind if needed)
4. Performance improvement measurable (>10% faster episode reset)