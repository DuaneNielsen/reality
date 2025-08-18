#include <gtest/gtest.h>
#include "../../src/asset_registry.hpp"
#include "../../src/asset_ids.hpp"

using namespace madEscape;

class AssetRegistryTest : public ::testing::Test {
protected:
    AssetRegistry registry;
};

TEST_F(AssetRegistryTest, BuiltInAssetsInitialized) {
    EXPECT_EQ(registry.getAssetId("cube"), AssetIDs::CUBE);
    EXPECT_EQ(registry.getAssetId("wall"), AssetIDs::WALL);
    EXPECT_EQ(registry.getAssetId("agent"), AssetIDs::AGENT);
    EXPECT_EQ(registry.getAssetId("plane"), AssetIDs::PLANE);
    EXPECT_EQ(registry.getAssetId("axis_x"), AssetIDs::AXIS_X);
    EXPECT_EQ(registry.getAssetId("axis_y"), AssetIDs::AXIS_Y);
    EXPECT_EQ(registry.getAssetId("axis_z"), AssetIDs::AXIS_Z);
}

TEST_F(AssetRegistryTest, BuiltInAssetCapabilities) {
    EXPECT_TRUE(registry.assetHasPhysics(AssetIDs::CUBE));
    EXPECT_TRUE(registry.assetHasRender(AssetIDs::CUBE));
    
    EXPECT_TRUE(registry.assetHasPhysics(AssetIDs::WALL));
    EXPECT_TRUE(registry.assetHasRender(AssetIDs::WALL));
    
    EXPECT_TRUE(registry.assetHasPhysics(AssetIDs::PLANE));
    EXPECT_TRUE(registry.assetHasRender(AssetIDs::PLANE));
    
    EXPECT_FALSE(registry.assetHasPhysics(AssetIDs::AXIS_X));
    EXPECT_TRUE(registry.assetHasRender(AssetIDs::AXIS_X));
    
    EXPECT_FALSE(registry.assetHasPhysics(AssetIDs::AXIS_Y));
    EXPECT_TRUE(registry.assetHasRender(AssetIDs::AXIS_Y));
    
    EXPECT_FALSE(registry.assetHasPhysics(AssetIDs::AXIS_Z));
    EXPECT_TRUE(registry.assetHasRender(AssetIDs::AXIS_Z));
}

TEST_F(AssetRegistryTest, RegisterDynamicAsset) {
    uint32_t customId = registry.registerAsset("custom_box", true, true);
    
    EXPECT_GE(customId, AssetIDs::DYNAMIC_START);
    
    EXPECT_EQ(registry.getAssetId("custom_box"), customId);
    
    EXPECT_TRUE(registry.assetHasPhysics(customId));
    EXPECT_TRUE(registry.assetHasRender(customId));
}

TEST_F(AssetRegistryTest, RegisterMultipleDynamicAssets) {
    uint32_t id1 = registry.registerAsset("asset1", true, false);
    uint32_t id2 = registry.registerAsset("asset2", false, true);
    uint32_t id3 = registry.registerAsset("asset3", true, true);
    
    EXPECT_EQ(id1, AssetIDs::DYNAMIC_START);
    EXPECT_EQ(id2, AssetIDs::DYNAMIC_START + 1);
    EXPECT_EQ(id3, AssetIDs::DYNAMIC_START + 2);
    
    EXPECT_TRUE(registry.assetHasPhysics(id1));
    EXPECT_FALSE(registry.assetHasRender(id1));
    
    EXPECT_FALSE(registry.assetHasPhysics(id2));
    EXPECT_TRUE(registry.assetHasRender(id2));
    
    EXPECT_TRUE(registry.assetHasPhysics(id3));
    EXPECT_TRUE(registry.assetHasRender(id3));
}

TEST_F(AssetRegistryTest, RegisterWithSpecificId) {
    uint32_t specificId = 100;
    uint32_t returnedId = registry.registerAssetWithId("specific", specificId, true, false);
    
    EXPECT_EQ(returnedId, specificId);
    EXPECT_EQ(registry.getAssetId("specific"), specificId);
    EXPECT_TRUE(registry.assetHasPhysics(specificId));
    EXPECT_FALSE(registry.assetHasRender(specificId));
}

TEST_F(AssetRegistryTest, DuplicateNameHandling) {
    uint32_t id1 = registry.registerAsset("duplicate", true, true);
    uint32_t id2 = registry.registerAsset("duplicate", false, false);
    
    EXPECT_EQ(id1, id2);
    
    EXPECT_TRUE(registry.assetHasPhysics(id1));
    EXPECT_TRUE(registry.assetHasRender(id1));
}

TEST_F(AssetRegistryTest, InvalidAssetLookup) {
    // Using tryGetAssetId for non-existent assets
    uint32_t id;
    EXPECT_FALSE(registry.tryGetAssetId("nonexistent", id));
}

TEST_F(AssetRegistryTest, BoundaryConditions) {
    EXPECT_FALSE(registry.assetHasPhysics(999));
    EXPECT_FALSE(registry.assetHasRender(999));
    
    // Empty asset name returns INVALID
    EXPECT_EQ(registry.registerAsset("", true, true), AssetIDs::INVALID);
}

TEST_F(AssetRegistryTest, GetAssetName) {
    EXPECT_EQ(registry.getAssetName(AssetIDs::CUBE), "cube");
    EXPECT_EQ(registry.getAssetName(AssetIDs::WALL), "wall");
    EXPECT_EQ(registry.getAssetName(999), "");
    
    uint32_t customId = registry.registerAsset("custom_name", true, false);
    EXPECT_EQ(registry.getAssetName(customId), "custom_name");
}