#pragma once

#include <gtest/gtest.h>
#include "mgr.hpp"
#include "viewer_core.hpp"
#include "types.hpp"
#include "test_level_helper.hpp"
#include <memory>
#include <vector>
#include <string>
#include <mutex>
#include <cstdlib>
#include <optional>
#include <cmath>

namespace madEscape {

// Base class for C++ tests that directly use Manager
class MadronaCppTestBase : public ::testing::Test {
protected:
    std::unique_ptr<Manager> mgr;
    Manager::Config config;
    std::vector<std::optional<CompiledLevel>> testLevels;
    
    void SetUp() override {
        // Create simple test levels
        CreateTestLevels();
        
        // Default config - can be overridden in derived classes
        config.execMode = madrona::ExecMode::CPU;
        config.gpuID = 0;
        config.numWorlds = 4;
        config.randSeed = 42;
        config.autoReset = true;
        config.enableBatchRenderer = false;
        config.batchRenderViewWidth = 64;
        config.batchRenderViewHeight = 64;
        config.extRenderAPI = nullptr;
        config.extRenderDev = nullptr;
        config.enableTrajectoryTracking = false;
        config.perWorldCompiledLevels = testLevels;
    }
    
    void TearDown() override {
        mgr.reset();
        testLevels.clear();
    }
    
    // Helper to create simple test levels
    void CreateTestLevels() {
        // Use the embedded default level
        CompiledLevel default_level = DefaultLevelProvider::GetDefaultLevel();
        
        // Create levels for all worlds (default 4)
        testLevels.clear();
        for (int i = 0; i < 4; i++) {
            testLevels.push_back(std::make_optional(default_level));
        }
    }
    
    // Helper to create manager
    ::testing::AssertionResult CreateManager() {
        // Ensure we have the right number of levels for the configured worlds
        if (testLevels.size() != static_cast<size_t>(config.numWorlds)) {
            CompiledLevel default_level = DefaultLevelProvider::GetDefaultLevel();
            
            testLevels.clear();
            for (int i = 0; i < config.numWorlds; i++) {
                testLevels.push_back(std::make_optional(default_level));
            }
            config.perWorldCompiledLevels = testLevels;
        }
        
        // Note: exceptions are disabled in this project, so we can't use try/catch
        mgr = std::make_unique<Manager>(config);
        if (!mgr) {
            return ::testing::AssertionFailure() 
                << "Failed to create manager: allocation failed";
        }
        return ::testing::AssertionSuccess();
    }
    
    // Helper to validate tensor dimensions
    ::testing::AssertionResult ValidateTensorShape(
        const madrona::py::Tensor& tensor,
        const std::vector<int64_t>& expected_shape) {
        
        int64_t num_dims = tensor.numDims();
        const int64_t* shape = tensor.dims();
        
        if (static_cast<size_t>(num_dims) != expected_shape.size()) {
            return ::testing::AssertionFailure()
                << "Dimension count mismatch. Expected: " << expected_shape.size()
                << ", Got: " << num_dims;
        }
        
        for (size_t i = 0; i < expected_shape.size(); i++) {
            if (shape[i] != expected_shape[i]) {
                return ::testing::AssertionFailure()
                    << "Dimension " << i << " mismatch. Expected: " << expected_shape[i]
                    << ", Got: " << shape[i];
            }
        }
        
        return ::testing::AssertionSuccess();
    }
};

// Base class for ViewerCore tests
class ViewerCoreTestBase : public MadronaCppTestBase {
protected:
    std::unique_ptr<ViewerCore> viewer;
    ViewerCore::Config viewerConfig;
    
    void SetUp() override {
        MadronaCppTestBase::SetUp();
        
        // Default viewer config
        viewerConfig.num_worlds = 4;
        viewerConfig.rand_seed = 42;
        viewerConfig.auto_reset = true;
        viewerConfig.load_path = "";
        viewerConfig.record_path = "";
        viewerConfig.replay_path = "";
    }
    
    void TearDown() override {
        viewer.reset();
        MadronaCppTestBase::TearDown();
    }
    
    ::testing::AssertionResult CreateViewer() {
        if (!mgr) {
            auto result = CreateManager();
            if (!result) return result;
        }
        
        viewer = std::make_unique<ViewerCore>(viewerConfig, mgr.get());
        if (!viewer) {
            return ::testing::AssertionFailure()
                << "Failed to create viewer: allocation failed";
        }
        return ::testing::AssertionSuccess();
    }
};

// Parameterized test base for testing with different world counts
class MadronaCppWorldCountTest : public MadronaCppTestBase,
                                 public ::testing::WithParamInterface<int32_t> {
protected:
    void SetUp() override {
        MadronaCppTestBase::SetUp();
        config.numWorlds = GetParam();
    }
};

// Test fixture for GPU tests
class MadronaCppGPUTest : public MadronaCppTestBase {
protected:
    static std::mutex gpu_test_mutex;  // Ensures only one GPU test runs at a time
    
    void SetUp() override {
        MadronaCppTestBase::SetUp();
        config.execMode = madrona::ExecMode::CUDA;
        config.gpuID = 0;
        
        // Check for environment variable to allow GPU tests
        const char* allow_gpu = std::getenv("ALLOW_GPU_TESTS_IN_SUITE");
        if (!allow_gpu || std::string(allow_gpu) != "1") {
            GTEST_SKIP() << "GPU tests disabled in main suite due to one-GPU-manager-per-process limitation.\n"
                        << "Run with ALLOW_GPU_TESTS_IN_SUITE=1 or use ./tests/run_gpu_tests_isolated.sh";
        }
        
        // Lock the mutex for this test - will be released in TearDown
        gpu_test_mutex.lock();
    }
    
    void TearDown() override {
        // First clean up the manager
        MadronaCppTestBase::TearDown();
        
        // Then release the mutex so the next GPU test can run
        gpu_test_mutex.unlock();
    }
};

// Static member definition moved to test_base.cpp to avoid multiple definitions

} // namespace madEscape