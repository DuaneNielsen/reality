#include <gtest/gtest.h>
#include "test_base.hpp"
#include "test_level_helper.hpp"

// Test fixture for GPU-specific tests
class CApiGPUTest : public MadronaGPUTest {};

// Basic GPU manager creation test
TEST_F(CApiGPUTest, ManagerCreation) {
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
}

// Test GPU manager creation with embedded levels
TEST_F(CApiGPUTest, ManagerCreationWithEmbeddedLevels) {
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
}

// Test GPU tensor properties
TEST_F(CApiGPUTest, GPUTensorProperties) {
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    MER_Tensor action_tensor;
    ASSERT_TRUE(GetTensor(action_tensor, mer_get_action_tensor));
    
    // GPU tensors should have gpu_id >= 0
    EXPECT_GE(action_tensor.gpu_id, 0);
    EXPECT_NE(action_tensor.data, nullptr);
    
    // Expected shape: [num_worlds * 1 agent, 3 actions (moveAmount, moveAngle, rotate)]
    std::vector<int64_t> expected_shape = {config.num_worlds, 3};
    EXPECT_TRUE(ValidateTensorShape(action_tensor, expected_shape));
}

// Test all GPU tensor getters
TEST_F(CApiGPUTest, AllGPUTensorGetters) {
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    // Test action tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_action_tensor));
        EXPECT_GE(tensor.gpu_id, 0);
        EXPECT_NE(tensor.data, nullptr);
    }
    
    // Test observation tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_self_observation_tensor));
        EXPECT_GE(tensor.gpu_id, 0);
        EXPECT_NE(tensor.data, nullptr);
    }
    
    // Test reward tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_reward_tensor));
        EXPECT_GE(tensor.gpu_id, 0);
        EXPECT_NE(tensor.data, nullptr);
    }
    
    // Test done tensor
    {
        MER_Tensor tensor;
        ASSERT_TRUE(GetTensor(tensor, mer_get_done_tensor));
        EXPECT_GE(tensor.gpu_id, 0);
        EXPECT_NE(tensor.data, nullptr);
    }
}

// Test GPU simulation step
TEST_F(CApiGPUTest, SimulationStep) {
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    // Run a single step
    MER_Result result = mer_step(handle);
    EXPECT_EQ(result, MER_SUCCESS);
}

// Test multiple GPU steps
TEST_F(CApiGPUTest, MultipleSteps) {
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    
    const int num_steps = 10;
    for (int i = 0; i < num_steps; i++) {
        MER_Result result = mer_step(handle);
        EXPECT_EQ(result, MER_SUCCESS) << "Failed at step " << i;
    }
}

// Test large world count on GPU
TEST_F(CApiGPUTest, LargeWorldCount) {
    config.num_worlds = 1024;  // Large world count for GPU
    
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
    
    // Verify tensor shape
    MER_Tensor action_tensor;
    ASSERT_TRUE(GetTensor(action_tensor, mer_get_action_tensor));
    
    std::vector<int64_t> expected_shape = {1024, 3};
    EXPECT_TRUE(ValidateTensorShape(action_tensor, expected_shape));
    
    // Run a few steps
    for (int i = 0; i < 5; i++) {
        MER_Result result = mer_step(handle);
        EXPECT_EQ(result, MER_SUCCESS);
    }
}

// Parameterized test for different GPU world counts
class CApiGPUWorldCountTest : public MadronaGPUTest,
                              public ::testing::WithParamInterface<int32_t> {
protected:
    void SetUp() override {
        MadronaGPUTest::SetUp();
        config.num_worlds = GetParam();
    }
};

TEST_P(CApiGPUWorldCountTest, VariousGPUWorldCounts) {
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    ASSERT_TRUE(CreateManager(levels.data(), levels.size()));
    EXPECT_NE(handle, nullptr);
    
    // Verify action tensor has correct shape
    MER_Tensor action_tensor;
    ASSERT_TRUE(GetTensor(action_tensor, mer_get_action_tensor));
    
    std::vector<int64_t> expected_shape = {GetParam(), 3};  // num_worlds * 1 agent, 3 actions
    EXPECT_TRUE(ValidateTensorShape(action_tensor, expected_shape));
}

// Test with power-of-2 world counts (optimal for GPU)
INSTANTIATE_TEST_SUITE_P(
    GPUWorldCounts,
    CApiGPUWorldCountTest,
    ::testing::Values(32, 64, 128, 256, 512)
);

// Test GPU-specific error conditions
TEST_F(CApiGPUTest, InvalidGPUDevice) {
    config.gpu_id = 999;  // Invalid GPU ID
    
    auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
    MER_Result result = mer_create_manager(&handle, &config, levels.data(), levels.size());
    
    // Should fail with invalid GPU ID
    EXPECT_NE(result, MER_SUCCESS);
    EXPECT_EQ(handle, nullptr);
}

// Test GPU memory stress (create and destroy multiple times)
TEST_F(CApiGPUTest, GPUMemoryStress) {
    const int iterations = 5;
    
    for (int i = 0; i < iterations; i++) {
        config.num_worlds = 256;
        auto levels = std::vector<MER_CompiledLevel>(config.num_worlds, DefaultLevelProvider::GetDefaultLevelC());
        
        ASSERT_TRUE(CreateManager(levels.data(), levels.size())) 
            << "Failed at iteration " << i;
        EXPECT_NE(handle, nullptr);
        
        // Run a few steps
        for (int j = 0; j < 3; j++) {
            MER_Result result = mer_step(handle);
            EXPECT_EQ(result, MER_SUCCESS);
        }
        
        // Clean up for next iteration
        mer_destroy_manager(handle);
        handle = nullptr;
    }
}