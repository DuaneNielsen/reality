#include <gtest/gtest.h>
#include "test_base.hpp"
#include "test_levels.hpp"

// Test fixture for CPU-specific tests
class CApiCPUTest : public MadronaTestBase {
protected:
    void SetUp() override {
        MadronaTestBase::SetUp();
        config.exec_mode = MER_EXEC_MODE_CPU;
    }
};

// Basic manager creation test
TEST_F(CApiCPUTest, ManagerCreation) {
    // Always provide test levels since manager requires them
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
}

// Test manager creation with loaded levels
TEST_F(CApiCPUTest, ManagerCreationWithLoadedLevels) {
    config.num_worlds = 4;
    MER_CompiledLevel level = TestLevelHelper::LoadLevelFromFile("tests/cpp/test_levels/quick_test.lvl");
    
    // Use loaded level if valid, otherwise use generated test level
    if (level.width > 0 && level.height > 0) {
        std::vector<MER_CompiledLevel> levels(config.num_worlds, level);
        ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    } else {
        auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
        ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    }
    EXPECT_NE(handle, nullptr);
}

// Test manager creation with custom levels
TEST_F(CApiCPUTest, ManagerCreationWithCustomLevels) {
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
}

// Test loading level from file
TEST_F(CApiCPUTest, LevelLoading) {
    MER_CompiledLevel level = TestLevelHelper::LoadLevelFromFile("tests/cpp/test_levels/quick_test.lvl");
    
    // Check if level was loaded (width/height should be non-zero for valid level)
    if (level.width > 0 && level.height > 0) {
        std::vector<MER_CompiledLevel> levels(config.num_worlds, level);
        ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
        EXPECT_NE(handle, nullptr);
    } else {
        // If file doesn't exist, use embedded level
        auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
        ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
        EXPECT_NE(handle, nullptr);
    }
}

// Test tensor access
TEST_F(CApiCPUTest, TensorAccess) {
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    MER_Tensor action_tensor;
    ASSERT_TRUE(GetTensor(action_tensor, mer_get_action_tensor));
    
    // Validate tensor properties
    EXPECT_EQ(action_tensor.gpu_id, -1);  // CPU tensor
    EXPECT_GT(action_tensor.num_dimensions, 0);
    EXPECT_NE(action_tensor.data, nullptr);
    
    // Expected shape: [num_worlds * 1 agent, 3 actions (moveAmount, moveAngle, rotate)]
    std::vector<int64_t> expected_shape = {config.num_worlds, 3};
    EXPECT_TRUE(ValidateTensorShape(action_tensor, expected_shape));
}

// Test all tensor getters
TEST_F(CApiCPUTest, AllTensorGetters) {
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    // Test action tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_action_tensor));
        EXPECT_EQ(tensor.gpu_id, -1);
        EXPECT_NE(tensor.data, nullptr);
    }
    
    // Test observation tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_self_observation_tensor));
        EXPECT_EQ(tensor.gpu_id, -1);
        EXPECT_NE(tensor.data, nullptr);
    }
    
    // Test reward tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_reward_tensor));
        EXPECT_EQ(tensor.gpu_id, -1);
        EXPECT_NE(tensor.data, nullptr);
    }
    
    // Test done tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_done_tensor));
        EXPECT_EQ(tensor.gpu_id, -1);
        EXPECT_NE(tensor.data, nullptr);
    }
}

// Test simulation step
TEST_F(CApiCPUTest, SimulationStep) {
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    // Run a single step
    MER_Result result = mer_step(handle);
    EXPECT_EQ(result, MER_SUCCESS);
}

// Test multiple steps
TEST_F(CApiCPUTest, MultipleSteps) {
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    const int num_steps = 10;
    for (int i = 0; i < num_steps; i++) {
        MER_Result result = mer_step(handle);
        EXPECT_EQ(result, MER_SUCCESS) << "Failed at step " << i;
    }
}

// Parameterized test for different world counts
class CApiCPUWorldCountTest : public MadronaWorldCountTest {};

TEST_P(CApiCPUWorldCountTest, VariousWorldCounts) {
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
    
    // Verify action tensor has correct shape
    MER_Tensor action_tensor;
    ASSERT_TRUE(GetTensor(action_tensor, mer_get_action_tensor));
    
    std::vector<int64_t> expected_shape = {GetParam(), 3};  // num_worlds * 1 agent, 3 actions
    EXPECT_TRUE(ValidateTensorShape(action_tensor, expected_shape));
}

INSTANTIATE_TEST_SUITE_P(
    WorldCounts,
    CApiCPUWorldCountTest,
    ::testing::Values(1, 2, 4, 8, 16, 32)
);

// Test error handling
TEST_F(CApiCPUTest, NullHandleError) {
    MER_Tensor tensor;
    MER_Result result = mer_get_action_tensor(nullptr, &tensor);
    EXPECT_NE(result, MER_SUCCESS);
}

TEST_F(CApiCPUTest, NullTensorError) {
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    MER_Result result = mer_get_action_tensor(handle, nullptr);
    EXPECT_NE(result, MER_SUCCESS);
}

// Test manager destruction and recreation
TEST_F(CApiCPUTest, ManagerRecreation) {
    // Create first manager
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
    
    // Destroy it
    mer_destroy_manager(handle);
    handle = nullptr;
    
    // Create another one
    auto levels2 = TestLevelHelper::CreateTestLevels(config.num_worlds);
    ASSERT_TRUE(CreateManager(levels2.data(), levels2.size()));
    EXPECT_NE(handle, nullptr);
}

TEST_F(CApiCPUTest, RenderOnlyEntitiesInLevel) {
    // Create level with render-only entities
    auto levels = TestLevelHelper::CreateTestLevels(config.num_worlds);
    
    // Mark some entities as render-only
    levels[0].tile_render_only[0] = true;  // First tile render-only
    
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    // Should create manager without issues
    EXPECT_NE(handle, nullptr);
    
    // Run a few steps to verify no crashes with render-only entities
    for (int i = 0; i < 5; i++) {
        MER_Result result = mer_step(handle);
        EXPECT_EQ(result, MER_SUCCESS);
    }
}