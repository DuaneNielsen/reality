// Simple test without GoogleTest to verify we can use C++ classes directly
#include "mgr.hpp"
#include "viewer_core.hpp"
#include "types.hpp"
#include <iostream>
#include <cassert>

using namespace madEscape;

int main() {
    std::cout << "Testing direct C++ usage without GoogleTest...\n";
    
    // Test 1: Create Manager directly
    std::cout << "Test 1: Creating Manager...";
    
    // Create a simple level for testing
    CompiledLevel level {};
    level.width = 16;
    level.height = 16;
    level.scale = 1.0f;
    level.num_tiles = 256;
    level.max_entities = level.num_tiles + 6 + 30; // tiles + persistent + buffer
    
    // Fill with empty tiles
    for (int i = 0; i < level.num_tiles; i++) {
        level.tile_types[i] = 0;  // Empty
        level.tile_x[i] = (i % 16) * level.scale;
        level.tile_y[i] = (i / 16) * level.scale;
    }
    
    Manager::Config config {
        .execMode = madrona::ExecMode::CPU,
        .gpuID = 0,
        .numWorlds = 4,
        .randSeed = 42,
        .autoReset = true,
        .enableBatchRenderer = false,
        .batchRenderViewWidth = 64,
        .batchRenderViewHeight = 64,
        .extRenderAPI = nullptr,
        .extRenderDev = nullptr,
        .enableTrajectoryTracking = false,
        .perWorldCompiledLevels = {level, level, level, level}  // 4 worlds
    };
    
    Manager mgr(config);
    std::cout << " PASSED\n";
    
    // Test 2: Access tensors
    std::cout << "Test 2: Accessing tensors...";
    auto actionTensor = mgr.actionTensor();
    auto selfObsTensor = mgr.selfObservationTensor();
    auto rewardTensor = mgr.rewardTensor();
    auto doneTensor = mgr.doneTensor();
    
    std::cout << " Action dims: " << actionTensor.numDims();
    std::cout << ", SelfObs dims: " << selfObsTensor.numDims();
    std::cout << ", Reward dims: " << rewardTensor.numDims();
    std::cout << ", Done dims: " << doneTensor.numDims() << std::endl;
    
    // Check actual shapes (all tensors are 3D: [numWorlds, agentsPerWorld, features])
    assert(actionTensor.numDims() == 2 || actionTensor.numDims() == 3);
    assert(selfObsTensor.numDims() == 3);  // [worlds, agents, features]
    assert(rewardTensor.numDims() == 3);
    assert(doneTensor.numDims() == 3);
    std::cout << " PASSED\n";
    
    // Test 3: Step simulation
    std::cout << "Test 3: Stepping simulation...";
    mgr.step();
    std::cout << " PASSED\n";
    
    // Test 4: Create ViewerCore
    std::cout << "Test 4: Creating ViewerCore...";
    ViewerCore::Config viewerConfig {
        .num_worlds = 4,
        .rand_seed = 42,
        .auto_reset = true,
        .load_path = "",
        .record_path = "",
        .replay_path = ""
    };
    
    ViewerCore viewer(viewerConfig, &mgr);
    std::cout << " PASSED\n";
    
    // Test 5: ViewerCore operations
    std::cout << "Test 5: ViewerCore operations...";
    viewer.toggleTrajectoryTracking(0);
    assert(viewer.isTrackingTrajectory(0));
    viewer.toggleTrajectoryTracking(0);
    assert(!viewer.isTrackingTrajectory(0));
    std::cout << " PASSED\n";
    
    std::cout << "\nAll tests passed!\n";
    return 0;
}