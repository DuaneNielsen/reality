#include "cpp_test_base.hpp"
#include "sim.hpp"

// For capturing stdout/stderr output in tests
using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

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
    
    // Based on consts.hpp: numAgents = 1
    // Action tensor is flattened: [numWorlds * numAgents, actionDims]
    EXPECT_EQ(actionTensor.numDims(), 2);
    EXPECT_EQ(selfObsTensor.numDims(), 3);
    EXPECT_EQ(rewardTensor.numDims(), 3);
    EXPECT_EQ(doneTensor.numDims(), 3);
    
    // Validate shapes with 1 agent per world
    // Action tensor is flattened: [numWorlds * numAgents, actionDims]
    // Actions: moveAmount, moveAngle, rotate (3 dimensions)
    EXPECT_TRUE(ValidateTensorShape(actionTensor, {4, 3}));  // 4 worlds * 1 agent, 3 action dims
    
    // Other tensors: [numWorlds, numAgents, features]
    EXPECT_TRUE(ValidateTensorShape(selfObsTensor, {4, 1, 5}));  // pos(3) + maxY(1) + theta(1)
    EXPECT_TRUE(ValidateTensorShape(rewardTensor, {4, 1, 1}));   // 1 reward per agent
    EXPECT_TRUE(ValidateTensorShape(doneTensor, {4, 1, 1}));      // 1 done flag per agent
}

TEST_F(DirectCppTest, StepSimulation) {
    ASSERT_TRUE(CreateManager());
    
    // Get action tensor and set some actions
    auto actionTensor = mgr->actionTensor();
    auto* actionData = static_cast<float*>(actionTensor.devicePtr());
    
    // Action tensor is flattened: [numWorlds * numAgents, actionDims]
    // Set action for first agent in first world (index 0)
    actionData[0] = 1.0f;  // moveAmount (1 = slow movement)
    actionData[1] = 0.0f;  // moveAngle (0 = forward)
    actionData[2] = 2.0f;  // rotate (2 = no rotation)
    
    // Step the simulation
    mgr->step();
    
    // Check that observation has been updated
    auto selfObsTensor = mgr->selfObservationTensor();
    auto* obsData = static_cast<float*>(selfObsTensor.devicePtr());
    
    // SelfObs tensor: [numWorlds, numAgents, features]
    // First agent in first world: offset = 0
    float x = obsData[0];
    float y = obsData[1];
    float z = obsData[2];
    
    // Note: Agent may start at origin and movement depends on level geometry
    // So we just verify the tensor is accessible and contains valid data
    EXPECT_FALSE(std::isnan(x));
    EXPECT_FALSE(std::isnan(y));
    EXPECT_FALSE(std::isnan(z));
}

class DirectViewerCoreTest : public ViewerCoreTestBase {};

TEST_F(DirectViewerCoreTest, CreateViewerCore) {
    ASSERT_TRUE(CreateManager());
    ASSERT_TRUE(CreateViewer());
    ASSERT_NE(viewer, nullptr);
}

TEST_F(DirectViewerCoreTest, TrajectoryTracking) {
    // Capture stdout to suppress trajectory logging output
    CaptureStdout();
    
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
    
    // Get captured output and verify trajectory logging occurred
    std::string captured_output = GetCapturedStdout();
    EXPECT_TRUE(captured_output.find("Trajectory logging enabled") != std::string::npos);
    EXPECT_TRUE(captured_output.find("Trajectory logging disabled") != std::string::npos);
}

// Parameterized test for different world counts
class DirectCppWorldCountTest : public MadronaCppWorldCountTest {};

TEST_P(DirectCppWorldCountTest, CreateWithDifferentWorldCounts) {
    ASSERT_TRUE(CreateManager());
    
    auto actionTensor = mgr->actionTensor();
    int32_t numWorlds = GetParam();
    
    // Action tensor is flattened: [numWorlds * numAgents, actionDims]
    // With 1 agent per world and 3 action dimensions
    EXPECT_TRUE(ValidateTensorShape(actionTensor, {numWorlds, 3}));
}

INSTANTIATE_TEST_SUITE_P(
    WorldCounts,
    DirectCppWorldCountTest,
    ::testing::Values(1, 2, 4, 8, 16)
);