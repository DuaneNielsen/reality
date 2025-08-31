#include <gtest/gtest.h>
#include "../fixtures/reusable_gpu_test.hpp"

using namespace madEscape;

// ============================================================================
// Stress tests that require custom managers (expensive NVRTC compilations)
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

