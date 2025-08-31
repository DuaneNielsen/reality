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
// Note: Expensive stress tests moved to test_manager_gpu_stress.cpp
// This file contains only fast tests using shared managers
// ============================================================================

// Test GPU-specific error conditions
// NOTE: Disabled because invalid GPU ID causes process abort in CUDA runtime
TEST_F(CustomGPUManagerTest, DISABLED_InvalidGPUDevice) {
    config.gpuID = 999;  // Invalid GPU ID
    
    // This test is disabled because CUDA runtime aborts the process on invalid GPU ID
    // TODO: Implement proper error handling with std::expected factory pattern
    EXPECT_FALSE(CreateManager());
    EXPECT_EQ(custom_manager, nullptr);
}