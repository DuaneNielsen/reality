#include <gtest/gtest.h>
#include "../../src/viewer_core.hpp"

using namespace madEscape;

// Tests for RecordReplayStateMachine (standalone class)
class RecordReplayStateMachineTest : public ::testing::Test {
protected:
    RecordReplayStateMachine sm;
};

TEST_F(RecordReplayStateMachineTest, InitialState) {
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Idle);
    EXPECT_FALSE(sm.isPaused());
    EXPECT_FALSE(sm.isRecording());
    EXPECT_FALSE(sm.isReplaying());
}

TEST_F(RecordReplayStateMachineTest, RecordingStartsPaused) {
    sm.startRecording();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    EXPECT_TRUE(sm.isPaused());
    EXPECT_TRUE(sm.isRecording());
    EXPECT_FALSE(sm.shouldRecordFrame());
}

TEST_F(RecordReplayStateMachineTest, PauseToggleRecording) {
    sm.startRecording();
    
    // Start paused
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    EXPECT_FALSE(sm.shouldRecordFrame());
    
    // Toggle to recording
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Recording);
    EXPECT_TRUE(sm.shouldRecordFrame());
    EXPECT_FALSE(sm.isPaused());
    
    // Toggle back to paused
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    EXPECT_FALSE(sm.shouldRecordFrame());
    EXPECT_TRUE(sm.isPaused());
}

TEST_F(RecordReplayStateMachineTest, ReplayStartsImmediately) {
    sm.startReplay();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Replaying);
    EXPECT_FALSE(sm.isPaused());
    EXPECT_TRUE(sm.isReplaying());
    EXPECT_TRUE(sm.shouldAdvanceReplay());
}

TEST_F(RecordReplayStateMachineTest, PauseToggleReplay) {
    sm.startReplay();
    
    // Start replaying
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Replaying);
    
    // Toggle to paused
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::ReplayingPaused);
    EXPECT_FALSE(sm.shouldAdvanceReplay());
    
    // Toggle back to replaying
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Replaying);
    EXPECT_TRUE(sm.shouldAdvanceReplay());
}

TEST_F(RecordReplayStateMachineTest, ReplayFinish) {
    sm.startReplay();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Replaying);
    
    sm.finishReplay();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::ReplayFinished);
}

TEST_F(RecordReplayStateMachineTest, StopReturnsToIdle) {
    sm.startRecording();
    EXPECT_NE(sm.getState(), RecordReplayStateMachine::Idle);
    
    sm.stop();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Idle);
}

// Tests for FrameActionManager (standalone class)
class FrameActionManagerTest : public ::testing::Test {
protected:
    static constexpr uint32_t NUM_WORLDS = 4;
    std::unique_ptr<FrameActionManager> manager;
    
    void SetUp() override {
        manager = std::make_unique<FrameActionManager>(NUM_WORLDS);
    }
};

TEST_F(FrameActionManagerTest, InitialDefaults) {
    const auto& actions = manager->getFrameActions();
    EXPECT_EQ(actions.size(), NUM_WORLDS * 3);
    
    for (uint32_t i = 0; i < NUM_WORLDS; i++) {
        EXPECT_EQ(actions[i * 3], 0);      // move_amount
        EXPECT_EQ(actions[i * 3 + 1], 0);  // move_angle
        EXPECT_EQ(actions[i * 3 + 2], 2);  // rotate (no rotation)
    }
    
    EXPECT_FALSE(manager->hasChanges());
}

TEST_F(FrameActionManagerTest, SetAction) {
    manager->setAction(1, 1, 3, 1);  // World 1: move forward-right, rotate left
    
    const auto& actions = manager->getFrameActions();
    EXPECT_EQ(actions[3], 1);  // move_amount
    EXPECT_EQ(actions[4], 3);  // move_angle
    EXPECT_EQ(actions[5], 1);  // rotate
    
    EXPECT_TRUE(manager->hasChanges());
}

TEST_F(FrameActionManagerTest, ResetToDefaults) {
    manager->setAction(0, 2, 4, 3);
    manager->setAction(2, 1, 1, 1);
    EXPECT_TRUE(manager->hasChanges());
    
    manager->resetToDefaults();
    
    const auto& actions = manager->getFrameActions();
    for (uint32_t i = 0; i < NUM_WORLDS; i++) {
        EXPECT_EQ(actions[i * 3], 0);
        EXPECT_EQ(actions[i * 3 + 1], 0);
        EXPECT_EQ(actions[i * 3 + 2], 2);
    }
    
    EXPECT_FALSE(manager->hasChanges());
}

TEST_F(FrameActionManagerTest, OutOfBoundsWorld) {
    // Should not crash or modify actions for out-of-bounds world
    manager->setAction(NUM_WORLDS + 1, 1, 1, 1);
    
    const auto& actions = manager->getFrameActions();
    // All actions should remain at defaults
    for (uint32_t i = 0; i < NUM_WORLDS; i++) {
        EXPECT_EQ(actions[i * 3], 0);
        EXPECT_EQ(actions[i * 3 + 1], 0);
        EXPECT_EQ(actions[i * 3 + 2], 2);
    }
}

// Test state transitions
TEST(RecordReplayStateMachineTransitions, RecordingSequence) {
    RecordReplayStateMachine sm;
    
    // Idle -> RecordingPaused
    sm.startRecording();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    
    // RecordingPaused -> Recording
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Recording);
    
    // Recording -> RecordingPaused
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    
    // RecordingPaused -> Idle
    sm.stop();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Idle);
}

TEST(RecordReplayStateMachineTransitions, ReplaySequence) {
    RecordReplayStateMachine sm;
    
    // Idle -> Replaying
    sm.startReplay();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Replaying);
    
    // Replaying -> ReplayingPaused
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::ReplayingPaused);
    
    // ReplayingPaused -> Replaying
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Replaying);
    
    // Replaying -> ReplayFinished
    sm.finishReplay();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::ReplayFinished);
}

// Test edge cases
TEST(RecordReplayStateMachineEdgeCases, MultipleStarts) {
    RecordReplayStateMachine sm;
    
    sm.startRecording();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
    
    // Starting recording again while recording should have no effect
    sm.startRecording();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::RecordingPaused);
}

TEST(RecordReplayStateMachineEdgeCases, TogglePauseInIdle) {
    RecordReplayStateMachine sm;
    
    // Toggling pause in idle state should have no effect
    sm.togglePause();
    EXPECT_EQ(sm.getState(), RecordReplayStateMachine::Idle);
}

// Main test runner
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}