#include <gtest/gtest.h>
#include "../fixtures/reusable_gpu_test.hpp"

using namespace madEscape;

// ============================================================================
// Tests that can share a manager (use ReusableGPUManagerTest)
// ============================================================================

// Basic GPU manager functionality
TEST_F(ReusableGPUManagerTest, BasicFunctionality) {
    ASSERT_NE(mgr, nullptr);
    
    // Test basic step operation
    mgr->step();
}

// Test GPU tensor properties with shared manager
TEST_F(ReusableGPUManagerTest, GPUTensorProperties) {
    auto action_tensor = GetActionTensor();
    
    // GPU tensors should have gpu_id >= 0
    EXPECT_GE(action_tensor.gpuID(), 0);
    EXPECT_NE(action_tensor.devicePtr(), nullptr);
    
    // Expected shape: [num_worlds, 3 actions (moveAmount, moveAngle, rotate)]
    std::vector<int64_t> expected_shape = {static_cast<int64_t>(config.numWorlds), 3};
    EXPECT_TRUE(ValidateTensorShape(action_tensor, expected_shape));
}

// Test all GPU tensor getters with shared manager
TEST_F(ReusableGPUManagerTest, AllGPUTensorGetters) {
    // Test action tensor
    {
        auto tensor = GetActionTensor();
        EXPECT_GE(tensor.gpuID(), 0);
        EXPECT_NE(tensor.devicePtr(), nullptr);
    }
    
    // Test observation tensor
    {
        auto tensor = GetObservationTensor();
        EXPECT_GE(tensor.gpuID(), 0);
        EXPECT_NE(tensor.devicePtr(), nullptr);
    }
    
    // Test reward tensor
    {
        auto tensor = GetRewardTensor();
        EXPECT_GE(tensor.gpuID(), 0);
        EXPECT_NE(tensor.devicePtr(), nullptr);
    }
    
    // Test done tensor
    {
        auto tensor = GetDoneTensor();
        EXPECT_GE(tensor.gpuID(), 0);
        EXPECT_NE(tensor.devicePtr(), nullptr);
    }
}

// Test GPU simulation step with shared manager
TEST_F(ReusableGPUManagerTest, SimulationStep) {
    // Run a single step
    Step();
    
    // Should complete without errors
    SUCCEED();
}

// Test multiple GPU steps with shared manager
TEST_F(ReusableGPUManagerTest, MultipleSteps) {
    const int num_steps = 10;
    for (int i = 0; i < num_steps; i++) {
        Step();
        // Should complete without errors
    }
    SUCCEED();
}

// Test action setting and stepping
TEST_F(ReusableGPUManagerTest, ActionSetting) {
    // Set different actions for different worlds
    for (uint32_t i = 0; i < config.numWorlds; i++) {
        SetWorldAction(i, 1, 2, 1);  // move forward-right, rotate left
    }
    
    // Step to process actions
    Step();
    
    // Reset for next test
    ResetAllWorlds();
    
    SUCCEED();
}

// Test world reset functionality
TEST_F(ReusableGPUManagerTest, WorldReset) {
    // Step a few times
    Step();
    Step();
    Step();
    
    // Reset world 0
    mgr->triggerReset(0);
    Step();  // Apply reset
    
    // Should work without errors
    SUCCEED();
}

// Test tensor consistency after reset
TEST_F(ReusableGPUManagerTest, TensorConsistencyAfterReset) {
    // Get initial tensor
    auto initial_tensor = GetActionTensor();
    std::vector<int64_t> initial_shape = {static_cast<int64_t>(config.numWorlds), 3};
    EXPECT_TRUE(ValidateTensorShape(initial_tensor, initial_shape));
    
    // Reset all worlds
    ResetAllWorlds();
    
    // Get tensor after reset
    auto reset_tensor = GetActionTensor();
    EXPECT_TRUE(ValidateTensorShape(reset_tensor, initial_shape));
    
    // Tensors should have same properties
    EXPECT_EQ(initial_tensor.gpuID(), reset_tensor.gpuID());
}

// ============================================================================
// Tests that need custom managers (use CustomGPUManagerTest)
// ============================================================================

// Test large world count on GPU
TEST_F(CustomGPUManagerTest, LargeWorldCount) {
    config.numWorlds = 1024;  // Large world count for GPU
    
    ASSERT_TRUE(CreateManager());
    EXPECT_NE(custom_manager, nullptr);
    
    // Verify tensor shape
    auto action_tensor = custom_manager->actionTensor();
    
    std::vector<int64_t> expected_shape = {1024, 3};
    
    // Manual tensor shape validation since this is in CustomGPUManagerTest
    const int64_t* dims = action_tensor.dims();
    int64_t num_dims = action_tensor.numDims();
    EXPECT_EQ(num_dims, 2);
    EXPECT_EQ(dims[0], 1024);
    EXPECT_EQ(dims[1], 3);
    
    // Run a few steps
    for (int i = 0; i < 5; i++) {
        custom_manager->step();
    }
}

// Parameterized test for different GPU world counts
class GPUWorldCountTest : public CustomGPUManagerTest,
                         public ::testing::WithParamInterface<int32_t> {
protected:
    void SetUp() override {
        CustomGPUManagerTest::SetUp();
        config.numWorlds = GetParam();
    }
    
    // Helper to validate tensor shape
    ::testing::AssertionResult ValidateTensorShape(const madrona::py::Tensor& tensor,
                                                  const std::vector<int64_t>& expected_shape) {
        const int64_t* dims = tensor.dims();
        int64_t num_dims = tensor.numDims();
        
        if (num_dims != static_cast<int64_t>(expected_shape.size())) {
            return ::testing::AssertionFailure()
                << "Dimension count mismatch. Expected: " << expected_shape.size()
                << ", Got: " << num_dims;
        }
        
        for (size_t i = 0; i < expected_shape.size(); i++) {
            if (dims[i] != expected_shape[i]) {
                return ::testing::AssertionFailure()
                    << "Dimension " << i << " mismatch. Expected: " << expected_shape[i]
                    << ", Got: " << dims[i];
            }
        }
        
        return ::testing::AssertionSuccess();
    }
};

TEST_P(GPUWorldCountTest, VariousGPUWorldCounts) {
    ASSERT_TRUE(CreateManager());
    EXPECT_NE(custom_manager, nullptr);
    
    // Verify action tensor has correct shape
    auto action_tensor = custom_manager->actionTensor();
    
    std::vector<int64_t> expected_shape = {GetParam(), 3};  // num_worlds, 3 actions
    EXPECT_TRUE(ValidateTensorShape(action_tensor, expected_shape));
}

// Test with power-of-2 world counts (optimal for GPU)
INSTANTIATE_TEST_SUITE_P(
    GPUWorldCounts,
    GPUWorldCountTest,
    ::testing::Values(32, 64, 128, 256, 512)
);

// Test GPU-specific error conditions
// NOTE: This test verifies that Manager provides clear error messages for invalid GPU IDs
// The test expects the process to abort with a descriptive error message
TEST_F(CustomGPUManagerTest, DISABLED_InvalidGPUDevice) {
    config.gpuID = 999;  // Invalid GPU ID
    
    // NOTE: This test would cause the program to abort with a clear error message
    // "ERROR: Invalid GPU ID 999. Available devices: 0-X"
    // We keep it disabled in regular test runs to avoid test suite interruption
    // but it can be manually enabled to verify error handling works correctly
    EXPECT_FALSE(CreateManager());
    EXPECT_EQ(custom_manager, nullptr);
}

// Test GPU memory stress (create and destroy multiple times)
TEST_F(CustomGPUManagerTest, GPUMemoryStress) {
    const int iterations = 5;
    
    for (int i = 0; i < iterations; i++) {
        config.numWorlds = 256;
        
        ASSERT_TRUE(CreateManager()) << "Failed at iteration " << i;
        EXPECT_NE(custom_manager, nullptr);
        
        // Run a few steps
        for (int j = 0; j < 3; j++) {
            custom_manager->step();
        }
        
        // Clean up for next iteration (happens automatically in TearDown)
        custom_manager.reset();
    }
}