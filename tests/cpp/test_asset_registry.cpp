#include <gtest/gtest.h>
#include "../../src/asset_registry.hpp"
#include "../../src/asset_ids.hpp"

using namespace madEscape;

class AssetRegistryTest : public ::testing::Test {
protected:
    // No longer need an instance - using static functions
};

TEST_F(AssetRegistryTest, BuiltInAssetsInitialized) {
    EXPECT_EQ(Assets::getAssetId("cube"), AssetIDs::CUBE);
    EXPECT_EQ(Assets::getAssetId("wall"), AssetIDs::WALL);
    EXPECT_EQ(Assets::getAssetId("agent"), AssetIDs::AGENT);
    EXPECT_EQ(Assets::getAssetId("plane"), AssetIDs::PLANE);
    EXPECT_EQ(Assets::getAssetId("axis_x"), AssetIDs::AXIS_X);
    EXPECT_EQ(Assets::getAssetId("axis_y"), AssetIDs::AXIS_Y);
    EXPECT_EQ(Assets::getAssetId("axis_z"), AssetIDs::AXIS_Z);
    EXPECT_EQ(Assets::getAssetId("cylinder"), AssetIDs::CYLINDER);
}

TEST_F(AssetRegistryTest, BuiltInAssetCapabilities) {
    EXPECT_TRUE(Assets::assetHasPhysics(AssetIDs::CUBE));
    EXPECT_TRUE(Assets::assetHasRender(AssetIDs::CUBE));
    
    EXPECT_TRUE(Assets::assetHasPhysics(AssetIDs::WALL));
    EXPECT_TRUE(Assets::assetHasRender(AssetIDs::WALL));
    
    EXPECT_TRUE(Assets::assetHasPhysics(AssetIDs::PLANE));
    EXPECT_TRUE(Assets::assetHasRender(AssetIDs::PLANE));
    
    EXPECT_FALSE(Assets::assetHasPhysics(AssetIDs::AXIS_X));
    EXPECT_TRUE(Assets::assetHasRender(AssetIDs::AXIS_X));
    
    EXPECT_FALSE(Assets::assetHasPhysics(AssetIDs::AXIS_Y));
    EXPECT_TRUE(Assets::assetHasRender(AssetIDs::AXIS_Y));
    
    EXPECT_FALSE(Assets::assetHasPhysics(AssetIDs::AXIS_Z));
    EXPECT_TRUE(Assets::assetHasRender(AssetIDs::AXIS_Z));
    
    EXPECT_TRUE(Assets::assetHasPhysics(AssetIDs::CYLINDER));
    EXPECT_TRUE(Assets::assetHasRender(AssetIDs::CYLINDER));
}

TEST_F(AssetRegistryTest, InvalidAssetLookup) {
    // Non-existent assets return INVALID
    EXPECT_EQ(Assets::getAssetId("nonexistent"), AssetIDs::INVALID);
}

TEST_F(AssetRegistryTest, BoundaryConditions) {
    // Out of bounds IDs return false
    EXPECT_FALSE(Assets::assetHasPhysics(999));
    EXPECT_FALSE(Assets::assetHasRender(999));
    EXPECT_FALSE(Assets::assetHasPhysics(AssetIDs::INVALID));
    EXPECT_FALSE(Assets::assetHasRender(AssetIDs::INVALID));
    
    // Check valid range
    EXPECT_FALSE(Assets::assetHasPhysics(AssetIDs::MAX_ASSETS));
    EXPECT_FALSE(Assets::assetHasRender(AssetIDs::MAX_ASSETS));
}

TEST_F(AssetRegistryTest, GetAssetName) {
    EXPECT_STREQ(Assets::getAssetName(AssetIDs::CUBE), "cube");
    EXPECT_STREQ(Assets::getAssetName(AssetIDs::WALL), "wall");
    EXPECT_STREQ(Assets::getAssetName(AssetIDs::AGENT), "agent");
    EXPECT_STREQ(Assets::getAssetName(AssetIDs::PLANE), "plane");
    EXPECT_STREQ(Assets::getAssetName(AssetIDs::CYLINDER), "cylinder");
    EXPECT_STREQ(Assets::getAssetName(999), "");
    EXPECT_STREQ(Assets::getAssetName(AssetIDs::INVALID), "");
}

TEST_F(AssetRegistryTest, GetAssetInfo) {
    // Test getting full asset info
    const auto* cubeInfo = Assets::getAssetInfo(AssetIDs::CUBE);
    ASSERT_NE(cubeInfo, nullptr);
    EXPECT_STREQ(cubeInfo->name, "cube");
    EXPECT_EQ(cubeInfo->id, AssetIDs::CUBE);
    EXPECT_TRUE(cubeInfo->hasPhysics);
    EXPECT_TRUE(cubeInfo->hasRender);
    EXPECT_EQ(cubeInfo->assetType, AssetInfo::FILE_MESH);
    
    const auto* planeInfo = Assets::getAssetInfo(AssetIDs::PLANE);
    ASSERT_NE(planeInfo, nullptr);
    EXPECT_STREQ(planeInfo->name, "plane");
    EXPECT_EQ(planeInfo->id, AssetIDs::PLANE);
    EXPECT_TRUE(planeInfo->hasPhysics);
    EXPECT_TRUE(planeInfo->hasRender);
    EXPECT_EQ(planeInfo->assetType, AssetInfo::BUILTIN_PLANE);
    
    // Test invalid ID
    const auto* invalidInfo = Assets::getAssetInfo(999);
    EXPECT_EQ(invalidInfo, nullptr);
}

TEST_F(AssetRegistryTest, AssetCounts) {
    // Count physics assets (cube, wall, agent, plane, cylinder)
    EXPECT_EQ(Assets::getPhysicsAssetCount(), 5);
    
    // Count render assets (cube, wall, agent, plane, axis_x, axis_y, axis_z, cylinder, lidar_ray)
    EXPECT_EQ(Assets::getRenderAssetCount(), 9);
}

TEST_F(AssetRegistryTest, AssetProperties) {
    // Test agent has rotation constraint
    const auto* agentInfo = Assets::getAssetInfo(AssetIDs::AGENT);
    ASSERT_NE(agentInfo, nullptr);
    EXPECT_TRUE(agentInfo->constrainRotationXY);
    
    // Test other assets don't have rotation constraint
    const auto* cubeInfo = Assets::getAssetInfo(AssetIDs::CUBE);
    ASSERT_NE(cubeInfo, nullptr);
    EXPECT_FALSE(cubeInfo->constrainRotationXY);
    
    // Test wall has zero inverse mass (static)
    const auto* wallInfo = Assets::getAssetInfo(AssetIDs::WALL);
    ASSERT_NE(wallInfo, nullptr);
    EXPECT_EQ(wallInfo->inverseMass, 0.0f);
    
    // Test cube has non-zero inverse mass (dynamic)
    EXPECT_GT(cubeInfo->inverseMass, 0.0f);
}