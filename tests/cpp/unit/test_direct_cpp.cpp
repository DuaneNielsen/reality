#include "cpp_test_base.hpp"
#include "sim.hpp"

using namespace madEscape;

class DirectCppTest : public MadronaCppTestBase {};

TEST_F(DirectCppTest, CreateManagerDirectly) {
    // Test creating manager directly without C API
    ASSERT_TRUE(CreateManager());
    ASSERT_NE(mgr, nullptr);
    
    // Access tensors directly
    auto actionTensor = mgr->actionTensor();
    auto selfObsTensor = mgr->selfObservationTensor();
    auto rewardTensor = mgr->rewardTensor();
    auto doneTensor = mgr->doneTensor();
    
    // Validate shapes
    EXPECT_TRUE(ValidateTensorShape(actionTensor, {4, 4}));  // 4 worlds, 4 action dims
    EXPECT_TRUE(ValidateTensorShape(selfObsTensor, {4, 3 + 1 + 1}));  // pos(3) + maxY(1) + theta(1)
    EXPECT_TRUE(ValidateTensorShape(rewardTensor, {4}));
    EXPECT_TRUE(ValidateTensorShape(doneTensor, {4}));
}

TEST_F(DirectCppTest, StepSimulation) {
    ASSERT_TRUE(CreateManager());
    
    // Get action tensor and set some actions
    auto actionTensor = mgr->actionTensor();
    auto* actionData = static_cast<float*>(actionTensor.devicePtr());
    
    // Set action for first agent: move forward
    actionData[0] = 1.0f;  // moveAmount
    actionData[1] = 0.0f;  // moveAngle  
    actionData[2] = 0.0f;  // rotate
    actionData[3] = 0.0f;  // grab
    
    // Step the simulation
    mgr->step();
    
    // Check that observation has been updated
    auto selfObsTensor = mgr->selfObservationTensor();
    auto* obsData = static_cast<float*>(selfObsTensor.devicePtr());
    
    // Agent should have moved (position should not be at origin)
    float x = obsData[0];
    float y = obsData[1];
    float z = obsData[2];
    
    // At least one coordinate should be non-zero after movement
    EXPECT_TRUE(x != 0.0f || y != 0.0f || z != 0.0f);
}

class DirectViewerCoreTest : public ViewerCoreTestBase {};

TEST_F(DirectViewerCoreTest, CreateViewerCore) {
    ASSERT_TRUE(CreateManager());
    ASSERT_TRUE(CreateViewer());
    ASSERT_NE(viewer, nullptr);
}

TEST_F(DirectViewerCoreTest, TrajectoryTracking) {
    ASSERT_TRUE(CreateManager());
    ASSERT_TRUE(CreateViewer());
    
    // Enable trajectory tracking for world 0
    viewer->toggleTrajectoryTracking(0);
    EXPECT_TRUE(viewer->isTrackingTrajectory(0));
    
    // Simulate some steps
    for (int i = 0; i < 10; i++) {
        viewer->stepSimulation();
    }
    
    // Disable trajectory tracking
    viewer->toggleTrajectoryTracking(0);
    EXPECT_FALSE(viewer->isTrackingTrajectory(0));
}

// Parameterized test for different world counts
class DirectCppWorldCountTest : public MadronaCppWorldCountTest {};

TEST_P(DirectCppWorldCountTest, CreateWithDifferentWorldCounts) {
    ASSERT_TRUE(CreateManager());
    
    auto actionTensor = mgr->actionTensor();
    int32_t numWorlds = GetParam();
    
    // First dimension should match world count
    EXPECT_TRUE(ValidateTensorShape(actionTensor, {numWorlds, 4}));
}

INSTANTIATE_TEST_SUITE_P(
    WorldCounts,
    DirectCppWorldCountTest,
    ::testing::Values(1, 2, 4, 8, 16)
);