#pragma once

#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <vector>
#include "test_level_helper.hpp"
#include "../../../src/mgr.hpp"
#include "../../../src/types.hpp"

namespace madEscape {

// Test fixture that creates a single GPU Manager and reuses it across tests
// Uses triggerReset() to reset state between tests instead of recreating managers
class ReusableGPUManagerTest : public ::testing::Test {
protected:
    static std::unique_ptr<Manager> shared_manager;
    static std::mutex gpu_test_mutex;
    static bool cuda_available_checked;
    static bool cuda_available;
    static uint32_t shared_num_worlds;
    
    Manager::Config config;
    Manager* mgr = nullptr;  // Pointer to shared manager or local manager

    void SetUp() override {
        // Check CUDA availability once
        if (!cuda_available_checked) {
            cuda_available_checked = true;
            cuda_available = CheckCudaAvailable();
        }
        
        if (!cuda_available) {
            GTEST_SKIP() << "CUDA not available on this system";
        }
        
        // Lock mutex for thread safety
        gpu_test_mutex.lock();
        
        // Set up default config
        config.execMode = madrona::ExecMode::CUDA;
        config.gpuID = 0;
        config.numWorlds = 4;
        config.randSeed = 42;
        config.autoReset = true;
        config.enableBatchRenderer = false;
        
        // Create shared manager if it doesn't exist or config changed
        if (!shared_manager || shared_num_worlds != config.numWorlds) {
            CreateSharedManager();
        }
        
        mgr = shared_manager.get();
        
        // Reset all worlds to clean state for this test
        ResetAllWorlds();
    }
    
    void TearDown() override {
        // Release mutex
        gpu_test_mutex.unlock();
    }
    
    // Create a new shared manager with current config
    void CreateSharedManager() {
        shared_manager.reset();  // Clean up existing manager
        
        // Create compiled levels for all worlds
        std::vector<std::optional<CompiledLevel>> levels;
        for (uint32_t i = 0; i < config.numWorlds; i++) {
            levels.push_back(DefaultLevelProvider::GetDefaultLevel());
        }
        config.perWorldCompiledLevels = std::move(levels);
        
        shared_manager = std::make_unique<Manager>(config);
        shared_num_worlds = config.numWorlds;
    }
    
    // Reset all worlds to clean state
    void ResetAllWorlds() {
        if (!mgr) return;
        
        for (uint32_t i = 0; i < config.numWorlds; i++) {
            mgr->triggerReset(i);
        }
        
        // Step once to apply resets
        mgr->step();
    }
    
    // Helper to validate tensor dimensions
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
    
    // Helper to set actions for a specific world
    void SetWorldAction(int32_t world_idx, int32_t move_amount, int32_t move_angle, int32_t rotate) {
        mgr->setAction(world_idx, move_amount, move_angle, rotate);
    }
    
    // Helper to step the simulation
    void Step() {
        mgr->step();
    }
    
    // Helper to get action tensor
    madrona::py::Tensor GetActionTensor() {
        return mgr->actionTensor();
    }
    
    // Helper to get observation tensor
    madrona::py::Tensor GetObservationTensor() {
        return mgr->selfObservationTensor();
    }
    
    // Helper to get reward tensor
    madrona::py::Tensor GetRewardTensor() {
        return mgr->rewardTensor();
    }
    
    // Helper to get done tensor
    madrona::py::Tensor GetDoneTensor() {
        return mgr->doneTensor();
    }

private:
    bool CheckCudaAvailable() {
        // For now, assume CUDA is available and let tests fail if it's not
        // TODO: Use CUDA runtime API to check for devices without creating manager
        return true;
    }
};

// Test fixture for tests that need a different configuration
// These tests will create their own managers
class CustomGPUManagerTest : public ::testing::Test {
protected:
    std::unique_ptr<Manager> custom_manager;
    static std::mutex gpu_test_mutex;
    static bool cuda_available_checked;
    static bool cuda_available;
    
    Manager::Config config;

    void SetUp() override {
        // Check CUDA availability once
        if (!cuda_available_checked) {
            cuda_available_checked = true;
            cuda_available = CheckCudaAvailable();
        }
        
        if (!cuda_available) {
            GTEST_SKIP() << "CUDA not available on this system";
        }
        
        // Lock mutex for thread safety
        gpu_test_mutex.lock();
        
        // Set up default config - derived classes can override
        config.execMode = madrona::ExecMode::CUDA;
        config.gpuID = 0;
        config.numWorlds = 4;
        config.randSeed = 42;
        config.autoReset = true;
        config.enableBatchRenderer = false;
    }
    
    void TearDown() override {
        custom_manager.reset();
        gpu_test_mutex.unlock();
    }
    
    // Create manager with current config
    ::testing::AssertionResult CreateManager() {
        // Create compiled levels for all worlds
        std::vector<std::optional<CompiledLevel>> levels;
        for (uint32_t i = 0; i < config.numWorlds; i++) {
            levels.push_back(DefaultLevelProvider::GetDefaultLevel());
        }
        config.perWorldCompiledLevels = std::move(levels);
        
        // Note: Manager constructor can fail, but since exceptions are disabled,
        // we rely on the constructor to handle errors gracefully or abort
        custom_manager = std::make_unique<Manager>(config);
        
        if (!custom_manager) {
            return ::testing::AssertionFailure() << "Failed to create manager";
        }
        
        return ::testing::AssertionSuccess();
    }

private:
    bool CheckCudaAvailable() {
        return true;  // Assume CUDA is available
    }
};

// Static member definitions
std::unique_ptr<Manager> ReusableGPUManagerTest::shared_manager = nullptr;
std::mutex ReusableGPUManagerTest::gpu_test_mutex;
bool ReusableGPUManagerTest::cuda_available_checked = false;
bool ReusableGPUManagerTest::cuda_available = false;
uint32_t ReusableGPUManagerTest::shared_num_worlds = 0;

std::mutex CustomGPUManagerTest::gpu_test_mutex;
bool CustomGPUManagerTest::cuda_available_checked = false;
bool CustomGPUManagerTest::cuda_available = false;

}