#pragma once

#include <gtest/gtest.h>
#include "compiled_level_compat.hpp"
#include "madrona_escape_room_c_api.h"
#include "types.hpp"  // For madEscape types
#include <memory>
#include <cstring>
#include <mutex>
#include <cstdlib>
#include <string>

class MadronaTestBase : public ::testing::Test {
protected:
    MER_ManagerHandle handle = nullptr;
    madEscape::ManagerConfig config = {};
    
    void SetUp() override {
        // Default config - can be overridden in derived classes
        config.exec_mode = madrona::ExecMode::CPU;
        config.gpu_id = 0;
        config.num_worlds = 4;
        config.rand_seed = 42;
        config.auto_reset = true;
        config.enable_batch_renderer = false;
    }
    
    void TearDown() override {
        if (handle) {
            mer_destroy_manager(handle);
            handle = nullptr;
        }
    }
    
    // Helper to create manager with custom config
    ::testing::AssertionResult CreateManager(const madEscape::CompiledLevel* levels = nullptr,
                                            int32_t num_levels = 0,
                                            const madEscape::SensorConfig* sensor_cfg = nullptr) {
        MER_Result result = mer_create_manager(&handle, &config, sensor_cfg, levels, num_levels);
        if (result != static_cast<MER_Result>(madEscape::Result::Success)) {
            return ::testing::AssertionFailure() 
                << "Failed to create manager: " << mer_result_to_string(result);
        }
        return ::testing::AssertionSuccess();
    }
    
    // Helper to validate tensor dimensions
    ::testing::AssertionResult ValidateTensorShape(const MER_Tensor& tensor,
                                                  const std::vector<int64_t>& expected_shape) {
        if (static_cast<size_t>(tensor.num_dimensions) != expected_shape.size()) {
            return ::testing::AssertionFailure()
                << "Dimension count mismatch. Expected: " << expected_shape.size()
                << ", Got: " << tensor.num_dimensions;
        }
        
        for (size_t i = 0; i < expected_shape.size(); i++) {
            if (tensor.dimensions[i] != expected_shape[i]) {
                return ::testing::AssertionFailure()
                    << "Dimension " << i << " mismatch. Expected: " << expected_shape[i]
                    << ", Got: " << tensor.dimensions[i];
            }
        }
        
        return ::testing::AssertionSuccess();
    }
    
    // Helper to get tensor safely
    ::testing::AssertionResult GetTensor(MER_Tensor& tensor,
                                        MER_Result (*getter)(MER_ManagerHandle, MER_Tensor*)) {
        MER_Result result = getter(handle, &tensor);
        if (result != MER_SUCCESS) {
            return ::testing::AssertionFailure()
                << "Failed to get tensor: " << mer_result_to_string(result);
        }
        return ::testing::AssertionSuccess();
    }
};

// Parameterized test base for testing with different world counts
class MadronaWorldCountTest : public MadronaTestBase,
                             public ::testing::WithParamInterface<int32_t> {
protected:
    void SetUp() override {
        MadronaTestBase::SetUp();
        config.num_worlds = GetParam();
    }
};

// Test fixture for GPU tests - IMPORTANT: Only one GPU manager can exist at a time!
class MadronaGPUTest : public MadronaTestBase {
protected:
    static bool cuda_available_checked;
    static bool cuda_available;
    static std::mutex gpu_test_mutex;  // Ensures only one GPU test runs at a time
    
    void SetUp() override {
        MadronaTestBase::SetUp();
        config.exec_mode = MER_EXEC_MODE_CUDA;
        
        // Check if CUDA is available (only check once for all tests)
        if (!cuda_available_checked) {
            cuda_available_checked = true;
            cuda_available = CheckCudaAvailable();
        }
        
        if (!cuda_available) {
            GTEST_SKIP() << "CUDA not available on this system";
        }
        
        // Lock the mutex for this test - will be released in TearDown
        gpu_test_mutex.lock();
    }
    
    void TearDown() override {
        // First clean up the manager
        MadronaTestBase::TearDown();
        
        // Then release the mutex so the next GPU test can run
        gpu_test_mutex.unlock();
    }
    
private:
    bool CheckCudaAvailable() {
        // Check if CUDA is available by checking for CUDA device
        // We can't actually create a manager here because only one GPU manager
        // can exist at a time, and we need to create managers in the actual tests
        
        // For now, just assume CUDA is available and let tests fail if it's not
        // A proper check would use CUDA runtime API to check for devices
        // without creating a Madrona manager
        
        // TODO: Use cudaGetDeviceCount or similar to check without creating manager
        return true;  // Assume CUDA is available
    }
};

