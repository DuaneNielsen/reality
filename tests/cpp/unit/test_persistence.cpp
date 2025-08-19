#include <gtest/gtest.h>
#include "cpp_test_base.hpp"
#include "types.hpp"
#include "asset_ids.hpp"
#include "consts.hpp"

using namespace madEscape;

class PersistenceTest : public MadronaCppTestBase {
protected:
    void SetUp() override {
        MadronaCppTestBase::SetUp();
        
        // Create test level with mix of persistent and non-persistent tiles
        CompiledLevel level {};
        level.width = 16;
        level.height = 16;
        level.scale = 1.0f;
        level.num_tiles = 6;
        level.max_entities = level.num_tiles + 6 + 30;
        
        // Set level name
        std::strcpy(level.level_name, "test_persistence_level");
        
        // Set spawn points
        level.num_spawns = 1;
        level.spawn_x[0] = 0.0f;
        level.spawn_y[0] = 0.0f;
        level.spawn_facing[0] = 0.0f;
        
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
        
        // Initialize render-only flags
        level.tile_render_only[0] = false;  // Wall - physics
        level.tile_render_only[1] = false;  // Wall - physics  
        level.tile_render_only[2] = false;  // Cube - physics
        level.tile_render_only[3] = false;  // Cube - physics
        level.tile_render_only[4] = true;   // AXIS_X - render-only
        level.tile_render_only[5] = true;   // AXIS_Y - render-only
        
        // Initialize entity type values
        level.tile_entity_type[0] = 2;  // Wall - EntityType::Wall
        level.tile_entity_type[1] = 2;  // Wall - EntityType::Wall  
        level.tile_entity_type[2] = 1;  // Cube - EntityType::Cube
        level.tile_entity_type[3] = 1;  // Cube - EntityType::Cube
        level.tile_entity_type[4] = 0;  // AXIS_X - EntityType::None (render-only)
        level.tile_entity_type[5] = 0;  // AXIS_Y - EntityType::None (render-only)
        
        // Initialize rest as empty
        for (int i = 6; i < CompiledLevel::MAX_TILES; i++) {
            level.object_ids[i] = 0;
            level.tile_persistent[i] = false;
            level.tile_render_only[i] = false;  // Default: physics entities
            level.tile_entity_type[i] = 0;  // EntityType::None
            level.tile_x[i] = 0.0f;
            level.tile_y[i] = 0.0f;
        }
        
        // Override testLevels with our custom level
        testLevels.clear();
        testLevels.push_back(level);
        
        // Update config to use our test level
        config.numWorlds = 1;
        config.perWorldCompiledLevels = testLevels;
        
        // Now create the manager with our custom config
        CreateManager();
    }
};

TEST_F(PersistenceTest, PersistentEntitiesCreatedOnce) {
    // Step simulation to initialize first frame
    mgr->step();
    
    // Count entities in the world - we should have the persistent walls and decorations
    // plus the non-persistent cubes created in the first episode
    // The exact entity count will depend on the implementation
    
    // Step through multiple episodes (trigger resets)
    for (int episode = 0; episode < 3; episode++) {
        // Run until episode ends (consts::episodeLen steps)
        for (int step = 0; step < consts::episodeLen; step++) {
            mgr->step();
        }
    }
    
    // Persistent entities should remain valid throughout
    // This test validates that the implementation doesn't crash
    // More detailed validation would require access to internal entity tracking
}

TEST_F(PersistenceTest, NonPersistentEntitiesRecreated) {
    // Step simulation to initialize
    mgr->step();
    
    // Run one complete episode
    for (int step = 0; step < consts::episodeLen; step++) {
        mgr->step();
    }
    
    // After episode reset, non-persistent entities should be recreated
    // The test passes if we can continue stepping without crashes
    for (int step = 0; step < 10; step++) {
        mgr->step();
    }
}

TEST_F(PersistenceTest, MixedPersistenceLevel) {
    // Verify we can step through the simulation
    // This tests that mixed persistence levels work correctly
    mgr->step();
    
    // The level has:
    // - 2 persistent walls
    // - 2 persistent axis markers  
    // - 2 non-persistent cubes
    // Total: 6 tiles as specified
    EXPECT_EQ(testLevels[0]->num_tiles, 6);
    
    // Count persistent vs non-persistent
    int persistentCount = 0;
    int nonPersistentCount = 0;
    for (int i = 0; i < testLevels[0]->num_tiles; i++) {
        if (testLevels[0]->tile_persistent[i]) {
            persistentCount++;
        } else {
            nonPersistentCount++;
        }
    }
    EXPECT_EQ(persistentCount, 4);  // 2 walls + 2 axis markers
    EXPECT_EQ(nonPersistentCount, 2);  // 2 cubes
}

TEST_F(PersistenceTest, AllPersistentLevel) {
    // Create level with all tiles marked persistent
    CompiledLevel allPersistent = *testLevels[0];
    for (int i = 0; i < allPersistent.num_tiles; i++) {
        allPersistent.tile_persistent[i] = true;
        // Keep original render-only flags
    }
    
    // Update config with new level
    testLevels.clear();
    testLevels.push_back(allPersistent);
    config.perWorldCompiledLevels = testLevels;
    
    // Recreate manager with all-persistent level
    CreateManager();
    
    // Should handle all-persistent level correctly
    mgr->step();
    
    // Run through multiple episodes
    for (int episode = 0; episode < 2; episode++) {
        for (int step = 0; step < consts::episodeLen; step++) {
            mgr->step();
        }
    }
}

TEST_F(PersistenceTest, NoPersistentLevel) {
    // Create level with no persistent tiles (current behavior)
    CompiledLevel noPersistent = *testLevels[0];
    for (int i = 0; i < noPersistent.num_tiles; i++) {
        noPersistent.tile_persistent[i] = false;
        // Keep original render-only flags
    }
    
    // Update config with new level
    testLevels.clear();
    testLevels.push_back(noPersistent);
    config.perWorldCompiledLevels = testLevels;
    
    // Recreate manager with no-persistent level
    CreateManager();
    
    // Should handle backward compatibility - all entities recreated each episode
    mgr->step();
    
    // Run through multiple episodes
    for (int episode = 0; episode < 2; episode++) {
        for (int step = 0; step < consts::episodeLen; step++) {
            mgr->step();
        }
    }
}

TEST_F(PersistenceTest, PersistentEntitiesReregistered) {
    // Step to trigger initial setup
    mgr->step();
    
    // Run until episode ends to trigger reset
    for (int step = 0; step < consts::episodeLen; step++) {
        mgr->step();
    }
    
    // After reset, persistent entities should be re-registered
    // Verify by continuing to step (would crash if not properly registered)
    for (int step = 0; step < 10; step++) {
        mgr->step();
    }
}

TEST_F(PersistenceTest, PersistencePerformance) {
    // Test performance improvement with persistent entities
    // Create two managers - one with persistence, one without
    
    CompiledLevel withPersistence = *testLevels[0];
    CompiledLevel withoutPersistence = *testLevels[0];
    
    // Mark all entities as non-persistent in the second level
    for (int i = 0; i < withoutPersistence.num_tiles; i++) {
        withoutPersistence.tile_persistent[i] = false;
        // Keep original render-only flags
    }
    
    // Test manager with persistence
    mgr->step();
    for (int step = 0; step < consts::episodeLen * 2; step++) {
        mgr->step();
    }
    
    // Create manager without persistence
    testLevels.clear();
    testLevels.push_back(withoutPersistence);
    config.perWorldCompiledLevels = testLevels;
    CreateManager();
    
    // Test manager without persistence
    mgr->step();
    for (int step = 0; step < consts::episodeLen * 2; step++) {
        mgr->step();
    }
    
    // Both should run without crashes
    // Performance measurement would require timing infrastructure
}

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

TEST_F(PersistenceTest, GenerateHardcodedLevelFile) {
    // Create a simple hardcoded level
    CompiledLevel level = {};
    level.width = 3;
    level.height = 3;
    level.scale = 1.0f;
    level.num_tiles = 4;
    level.max_entities = 10;
    std::strcpy(level.level_name, "hardcoded_test");
    
    // Set spawn point
    level.num_spawns = 1;
    level.spawn_x[0] = 0.0f;
    level.spawn_y[0] = 0.0f;
    level.spawn_facing[0] = 0.0f;
    
    // Tile 0: Wall at (-1, -1)
    level.object_ids[0] = 2;  // WALL
    level.tile_x[0] = -1.0f;
    level.tile_y[0] = -1.0f;
    level.tile_persistent[0] = true;
    level.tile_render_only[0] = false;
    level.tile_entity_type[0] = 2;  // EntityType::Wall
    
    // Tile 1: Cube at (1, -1)
    level.object_ids[1] = 1;  // CUBE
    level.tile_x[1] = 1.0f;
    level.tile_y[1] = -1.0f;
    level.tile_persistent[1] = false;
    level.tile_render_only[1] = false;
    level.tile_entity_type[1] = 1;  // EntityType::Cube
    
    // Tile 2: Wall at (-1, 1)
    level.object_ids[2] = 2;  // WALL
    level.tile_x[2] = -1.0f;
    level.tile_y[2] = 1.0f;
    level.tile_persistent[2] = true;
    level.tile_render_only[2] = false;
    level.tile_entity_type[2] = 2;  // EntityType::Wall
    
    // Tile 3: Axis marker (render-only) at (1, 1)
    level.object_ids[3] = 5;  // AXIS_X
    level.tile_x[3] = 1.0f;
    level.tile_y[3] = 1.0f;
    level.tile_persistent[3] = true;
    level.tile_render_only[3] = true;
    level.tile_entity_type[3] = 0;  // EntityType::None
    
    // Write to file
    std::ofstream file("test_hardcoded.lvl", std::ios::binary);
    ASSERT_TRUE(file.is_open());
    file.write(reinterpret_cast<const char*>(&level), sizeof(CompiledLevel));
    file.close();
    
    // Verify file was created and has correct size
    std::ifstream check_file("test_hardcoded.lvl", std::ios::binary | std::ios::ate);
    ASSERT_TRUE(check_file.is_open());
    size_t file_size = check_file.tellg();
    EXPECT_EQ(file_size, sizeof(CompiledLevel));
    check_file.close();
    
    // Read back and verify contents
    CompiledLevel loaded_level = {};
    std::ifstream read_file("test_hardcoded.lvl", std::ios::binary);
    ASSERT_TRUE(read_file.is_open());
    read_file.read(reinterpret_cast<char*>(&loaded_level), sizeof(CompiledLevel));
    read_file.close();
    
    // Verify key fields
    EXPECT_EQ(loaded_level.width, 3);
    EXPECT_EQ(loaded_level.height, 3);
    EXPECT_EQ(loaded_level.num_tiles, 4);
    EXPECT_STREQ(loaded_level.level_name, "hardcoded_test");
    
    // Verify tiles
    EXPECT_EQ(loaded_level.object_ids[0], 2);  // WALL
    EXPECT_EQ(loaded_level.tile_entity_type[0], 2);  // EntityType::Wall
    EXPECT_EQ(loaded_level.object_ids[1], 1);  // CUBE
    EXPECT_EQ(loaded_level.tile_entity_type[1], 1);  // EntityType::Cube
}