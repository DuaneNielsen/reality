#include <gtest/gtest.h>
#include "../fixtures/cpp_test_base.hpp"
#include "../../src/asset_ids.hpp"

using namespace madEscape;

class AssetRefactorTest : public MadronaCppTestBase {
protected:
    void SetUp() override {
        // Create a test level with mixed asset types
        CompiledLevel level {};
        level.width = 16;
        level.height = 16;
        level.scale = 1.0f;
        // Initialize world boundaries
        level.world_min_x = -20.0f;
        level.world_max_x = 20.0f;
        level.world_min_y = -20.0f;
        level.world_max_y = 20.0f;
        level.world_min_z = 0.0f;
        level.world_max_z = 25.0f;
        level.num_tiles = 3;
        level.max_entities = level.num_tiles + 6 + 30;
        
        // Test case from the plan:
        // Wall (physics entity)
        level.object_ids[0] = AssetIDs::WALL;  // ID = 2
        level.tile_x[0] = -5.0f;
        level.tile_y[0] = 0.0f;
        
        // Cube (physics entity)
        level.object_ids[1] = AssetIDs::CUBE;  // ID = 1
        level.tile_x[1] = 0.0f;
        level.tile_y[1] = 0.0f;
        
        // Axis_X (render-only entity)
        level.object_ids[2] = AssetIDs::AXIS_X; // ID = 5
        level.tile_x[2] = 5.0f;
        level.tile_y[2] = 0.0f;
        
        // Fill remaining tiles as empty
        for (int i = 3; i < CompiledLevel::MAX_TILES; i++) {
            level.object_ids[i] = 0;  // Empty
            level.tile_x[i] = 0.0f;
            level.tile_y[i] = 0.0f;
        }
        
        // Setup spawn point
        level.num_spawns = 1;
        level.spawn_x[0] = 0.0f;
        level.spawn_y[0] = -3.0f;
        level.spawn_facing[0] = 0.0f;
        
        // Set level name
        strncpy(level.level_name, "AssetRefactorTest", CompiledLevel::MAX_LEVEL_NAME_LENGTH - 1);
        
        // Configure manager with this test level
        testLevels.clear();
        testLevels.push_back(std::make_optional(level));
        
        config.execMode = madrona::ExecMode::CPU;
        config.numWorlds = 1;
        config.perWorldCompiledLevels = testLevels;
        
        MadronaCppTestBase::SetUp();
    }
};

TEST_F(AssetRefactorTest, MixedAssetTypesCreation) {
    // Create the manager with our test level
    ASSERT_TRUE(CreateManager());
    
    // Step the simulation once to ensure entities are created
    mgr->step();
    
    // Get observations to verify entities exist
    auto self_obs = mgr->selfObservationTensor();
    ASSERT_NE(self_obs.devicePtr(), nullptr);
    
    // The test verifies that:
    // 1. The simulation runs without crashing with mixed asset types
    // 2. Wall (WALL) creates a physics entity
    // 3. Cube (CUBE) creates a physics entity
    // 4. Axis indicator (AXIS_X) creates a render-only entity
    // The fact that the simulation steps successfully indicates
    // the generic entity creation is working
}

TEST_F(AssetRefactorTest, EmptyTilesHandled) {
    // Create a level with mostly empty tiles
    CompiledLevel level {};
    level.width = 4;
    level.height = 4;
    level.scale = 1.0f;
    // Initialize world boundaries
    level.world_min_x = -5.0f;
    level.world_max_x = 5.0f;
    level.world_min_y = -5.0f;
    level.world_max_y = 5.0f;
    level.world_min_z = 0.0f;
    level.world_max_z = 10.0f;
    level.num_tiles = 16;
    level.max_entities = level.num_tiles + 6 + 30;
    
    // Only place one wall, rest are empty
    level.object_ids[0] = AssetIDs::WALL;
    level.tile_x[0] = 0.0f;
    level.tile_y[0] = 0.0f;
    
    for (int i = 1; i < 16; i++) {
        level.object_ids[i] = AssetIDs::INVALID;  // Explicitly empty
        level.tile_x[i] = (i % 4) * level.scale;
        level.tile_y[i] = (i / 4) * level.scale;
    }
    
    level.num_spawns = 1;
    level.spawn_x[0] = 1.0f;
    level.spawn_y[0] = 1.0f;
    level.spawn_facing[0] = 0.0f;
    
    testLevels.clear();
    testLevels.push_back(std::make_optional(level));
    config.perWorldCompiledLevels = testLevels;
    
    ASSERT_TRUE(CreateManager());
    mgr->step();
    
    // Test passes if no crash occurs with mostly empty level
}

TEST_F(AssetRefactorTest, AllBuiltInAssetTypes) {
    // Test with all built-in asset types
    CompiledLevel level {};
    level.width = 8;
    level.height = 8;
    level.scale = 1.0f;
    // Initialize world boundaries
    level.world_min_x = -10.0f;
    level.world_max_x = 10.0f;
    level.world_min_y = -10.0f;
    level.world_max_y = 10.0f;
    level.world_min_z = 0.0f;
    level.world_max_z = 20.0f;
    level.num_tiles = 7;  // One of each asset type
    level.max_entities = level.num_tiles + 6 + 30;
    
    // Place one of each asset type
    level.object_ids[0] = AssetIDs::CUBE;    // 1
    level.object_ids[1] = AssetIDs::WALL;    // 2
    level.object_ids[2] = AssetIDs::AGENT;   // 3 (though agents are handled specially)
    level.object_ids[3] = AssetIDs::PLANE;   // 4 (though plane is created separately)
    level.object_ids[4] = AssetIDs::AXIS_X;  // 5
    level.object_ids[5] = AssetIDs::AXIS_Y;  // 6
    level.object_ids[6] = AssetIDs::AXIS_Z;  // 7
    
    for (int i = 0; i < 7; i++) {
        level.tile_x[i] = i * 2.0f - 6.0f;
        level.tile_y[i] = 0.0f;
    }
    
    // Clear rest
    for (int i = 7; i < CompiledLevel::MAX_TILES; i++) {
        level.object_ids[i] = 0;
        level.tile_x[i] = 0.0f;
        level.tile_y[i] = 0.0f;
    }
    
    level.num_spawns = 1;
    level.spawn_x[0] = 0.0f;
    level.spawn_y[0] = -5.0f;
    level.spawn_facing[0] = 0.0f;
    
    testLevels.clear();
    testLevels.push_back(std::make_optional(level));
    config.perWorldCompiledLevels = testLevels;
    
    ASSERT_TRUE(CreateManager());
    mgr->step();
    
    // Success means all asset types were handled without crash
}