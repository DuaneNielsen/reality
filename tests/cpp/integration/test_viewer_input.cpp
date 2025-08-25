#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "mock_components.hpp"
#include "../../../src/consts.hpp"

using namespace madEscape::consts::action;

// These tests simulate viewer input processing logic
// They test the keyboard-to-action mapping that mirrors viewer.cpp behavior
class ViewerInputMappingTest : public ViewerTestBase {
protected:
    void SetUp() override {
        ViewerTestBase::SetUp();
        
        // Use the default level
        auto level = LevelComparer::getDefaultLevel();
        
        config.num_worlds = 1;
        ASSERT_TRUE(CreateManager(&level, 1));
        
        mgr_ = std::make_unique<TestManagerWrapper>(handle);
    }
    
    void TearDown() override {
        mgr_.reset();
        ViewerTestBase::TearDown();
        // No test file to remove anymore
    }
    
    // Helper to process input and return resulting action
    ActionRecorder::RecordedAction processInput(InputSimulator& input) {
        MockViewer viewer(1);
        viewer.setFrameLimit(1);
        
        // Copy input state to the viewer's input simulator
        InputSimulator& viewer_input = viewer.getInputSimulator();
        viewer_input.releaseAll();
        for (int i = 0; i < InputSimulator::Key_Count; i++) {
            if (input.getInput().keyPressed(static_cast<InputSimulator::Key>(i))) {
                viewer_input.pressKey(static_cast<InputSimulator::Key>(i));
            }
        }
        
        ActionRecorder::RecordedAction result = {};
        
        viewer.loop(
            [](int32_t, const MockViewer::UserInput&) {},
            [&](int32_t world_idx, int32_t, const MockViewer::UserInput& user_input) {
                // This mimics the viewer's input processing logic
                int32_t x = 0, y = 0, r = 2;
                bool shift_pressed = user_input.keyPressed(MockViewer::KeyboardKey::Shift);
                
                if (user_input.keyPressed(MockViewer::KeyboardKey::W)) y += 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::S)) y -= 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::D)) x += 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::A)) x -= 1;
                if (user_input.keyPressed(MockViewer::KeyboardKey::Q)) r -= shift_pressed ? 2 : 1;  // Q left: r=2-1=1 (SLOW_LEFT)
                if (user_input.keyPressed(MockViewer::KeyboardKey::E)) r += shift_pressed ? 2 : 1;  // E right: r=2+1=3 (SLOW_RIGHT)
                
                // Calculate move_amount
                int32_t move_amount;
                if (x == 0 && y == 0) {
                    move_amount = 0;
                } else if (shift_pressed) {
                    move_amount = 3;  // move_amount::FAST
                } else {
                    move_amount = 1;  // move_amount::SLOW
                }
                
                // Calculate move_angle
                int32_t move_angle;
                if (x == 0 && y == 1) {
                    move_angle = 0;  // Forward
                } else if (x == 1 && y == 1) {
                    move_angle = 1;  // Forward-right
                } else if (x == 1 && y == 0) {
                    move_angle = 2;  // Right
                } else if (x == 1 && y == -1) {
                    move_angle = 3;  // Backward-right
                } else if (x == 0 && y == -1) {
                    move_angle = 4;  // Backward
                } else if (x == -1 && y == -1) {
                    move_angle = 5;  // Backward-left
                } else if (x == -1 && y == 0) {
                    move_angle = 6;  // Left
                } else if (x == -1 && y == 1) {
                    move_angle = 7;  // Forward-left
                } else {
                    move_angle = 0;
                }
                
                mgr_->setAction(world_idx, move_amount, move_angle, r);
                result = {0, world_idx, move_amount, move_angle, r};
            },
            [&]() { mgr_->step(); },
            []() {}
        );
        
        // Input state is already updated in the viewer
        
        return result;
    }
    
    std::unique_ptr<TestManagerWrapper> mgr_;
};

// Test WASD movement mapping
TEST_F(ViewerInputMappingTest, WASDToMovementActions) {
    InputSimulator input;
    
    // Test forward (W)
    input.simulateMovement(0, 1);
    auto action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::FORWARD);
    
    // Test backward (S)
    input.simulateMovement(0, -1);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::BACKWARD);
    
    // Test right (D)
    input.simulateMovement(1, 0);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::RIGHT);
    
    // Test left (A)
    input.simulateMovement(-1, 0);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::LEFT);
}

// Test diagonal movement
TEST_F(ViewerInputMappingTest, DiagonalInputMapping) {
    InputSimulator input;
    
    // Test forward-right (W+D)
    input.simulateMovement(1, 1);
    auto action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::FORWARD_RIGHT);
    
    // Test forward-left (W+A)
    input.simulateMovement(-1, 1);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::FORWARD_LEFT);
    
    // Test backward-right (S+D)
    input.simulateMovement(1, -1);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::BACKWARD_RIGHT);
    
    // Test backward-left (S+A)
    input.simulateMovement(-1, -1);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::BACKWARD_LEFT);
}

// Test rotation mapping
TEST_F(ViewerInputMappingTest, QERotationMapping) {
    InputSimulator input;
    
    // Test Q key rotation (left, counter-clockwise, following right-hand rule)
    input.releaseAll();
    input.pressKey(InputSimulator::Key::Q);
    auto action = processInput(input);
    EXPECT_EQ(action.rotate, rotate::SLOW_LEFT);  // Q rotates left (r -= 1 = 1)
    
    // Test E key rotation (right, clockwise, following right-hand rule)
    input.releaseAll();
    input.pressKey(InputSimulator::Key::E);
    action = processInput(input);
    EXPECT_EQ(action.rotate, rotate::SLOW_RIGHT);   // E rotates right (r += 1 = 3)
    
    // Test no rotation (default)
    input.releaseAll();
    action = processInput(input);
    EXPECT_EQ(action.rotate, rotate::NONE);
}

// Test shift speed modifier
TEST_F(ViewerInputMappingTest, ShiftSpeedModifier) {
    InputSimulator input;
    
    // Test slow movement without shift
    input.simulateMovement(0, 1, false);
    auto action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    
    // Test fast movement with shift
    input.simulateMovement(0, 1, true);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::FAST);
    
    // Test stop when no movement keys pressed
    input.releaseAll();
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::STOP);
}

// Test shift rotation modifier
TEST_F(ViewerInputMappingTest, ShiftRotationModifier) {
    InputSimulator input;
    
    // Test slow rotate left without shift
    input.releaseAll();
    input.pressKey(InputSimulator::Key::Q);
    auto action = processInput(input);
    EXPECT_EQ(action.rotate, rotate::SLOW_LEFT);
    
    // Test fast rotate left with shift
    input.releaseAll();
    input.pressKey(InputSimulator::Key::Q);
    input.pressKey(InputSimulator::Key::Shift);
    action = processInput(input);
    EXPECT_EQ(action.rotate, rotate::FAST_LEFT);
    
    // Test slow rotate right without shift
    input.releaseAll();
    input.pressKey(InputSimulator::Key::E);
    action = processInput(input);
    EXPECT_EQ(action.rotate, rotate::SLOW_RIGHT);
    
    // Test fast rotate right with shift
    input.releaseAll();
    input.pressKey(InputSimulator::Key::E);
    input.pressKey(InputSimulator::Key::Shift);
    action = processInput(input);
    EXPECT_EQ(action.rotate, rotate::FAST_RIGHT);
}

// Test reset key functionality
TEST_F(ViewerInputMappingTest, RKeyTriggersReset) {
    MockViewer viewer(1);
    InputSimulator& input = viewer.getInputSimulator();
    
    bool reset_triggered = false;
    int32_t reset_world = -1;
    
    viewer.setFrameLimit(1);
    input.hitKey(MockViewer::KeyboardKey::R);
    
    viewer.loop(
        [&](int32_t world_idx, const MockViewer::UserInput& user_input) {
            if (user_input.keyHit(MockViewer::KeyboardKey::R)) {
                mgr_->triggerReset(world_idx);
                reset_triggered = true;
                reset_world = world_idx;
            }
        },
        [&](int32_t, int32_t, const MockViewer::UserInput&) {},
        [&]() { mgr_->step(); },
        []() {}
    );
    
    EXPECT_TRUE(reset_triggered);
    EXPECT_EQ(reset_world, 0);
    
    auto& resets = mgr_->getResets();
    EXPECT_EQ(resets.size(), 1);
    EXPECT_EQ(resets[0].second, 0);
}

// Test complex input combinations
TEST_F(ViewerInputMappingTest, ComplexInputCombinations) {
    InputSimulator input;
    
    // Test movement + rotation (W + Q)
    input.releaseAll();
    input.pressKey(InputSimulator::Key::W);
    input.pressKey(InputSimulator::Key::Q);
    auto action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::SLOW);
    EXPECT_EQ(action.move_angle, move_angle::FORWARD);
    EXPECT_EQ(action.rotate, rotate::SLOW_LEFT);
    
    // Test diagonal + rotation + shift (W + D + E + Shift)
    input.releaseAll();
    input.pressKey(InputSimulator::Key::W);
    input.pressKey(InputSimulator::Key::D);
    input.pressKey(InputSimulator::Key::E);
    input.pressKey(InputSimulator::Key::Shift);
    action = processInput(input);
    EXPECT_EQ(action.move_amount, move_amount::FAST);
    EXPECT_EQ(action.move_angle, move_angle::FORWARD_RIGHT);
    EXPECT_EQ(action.rotate, rotate::FAST_RIGHT);
}

// Test input sequence over multiple frames
TEST_F(ViewerInputMappingTest, MultiFrameInputSequence) {
    MockViewer viewer(1);
    InputSimulator& input = viewer.getInputSimulator();
    std::vector<ActionRecorder::RecordedAction> actions;
    
    viewer.setFrameLimit(5);
    
    // Simulate a sequence: forward, turn right, forward-right, stop
    viewer.loop(
        [](int32_t, const MockViewer::UserInput&) {},
        [&](int32_t world_idx, int32_t, const MockViewer::UserInput& user_input) {
            static int frame = 0;
            
            // Different input per frame
            if (frame == 0) {
                // Frame 0: Move forward
                input.releaseAll();
                input.pressKey(InputSimulator::Key::W);
            } else if (frame == 1) {
                // Frame 1: Turn right while moving
                input.pressKey(InputSimulator::Key::E);
            } else if (frame == 2) {
                // Frame 2: Move forward-right
                input.releaseAll();
                input.pressKey(InputSimulator::Key::W);
                input.pressKey(InputSimulator::Key::D);
            } else if (frame == 3) {
                // Frame 3: Stop moving, keep turning
                input.releaseAll();
                input.pressKey(InputSimulator::Key::E);
            } else {
                // Frame 4: Full stop
                input.releaseAll();
            }
            
            // Process current input state
            int32_t x = 0, y = 0, r = 2;
            
            if (user_input.keyPressed(MockViewer::KeyboardKey::W)) y += 1;
            if (user_input.keyPressed(MockViewer::KeyboardKey::S)) y -= 1;
            if (user_input.keyPressed(MockViewer::KeyboardKey::D)) x += 1;
            if (user_input.keyPressed(MockViewer::KeyboardKey::A)) x -= 1;
            if (user_input.keyPressed(MockViewer::KeyboardKey::Q)) r -= 1;  // Q rotates left: r=2-1=1 (SLOW_LEFT)
            if (user_input.keyPressed(MockViewer::KeyboardKey::E)) r += 1;  // E rotates right: r=2+1=3 (SLOW_RIGHT)
            
            int32_t move_amount = (x == 0 && y == 0) ? 0 : 1;
            int32_t move_angle = 0;
            
            if (x == 0 && y == 1) move_angle = 0;
            else if (x == 1 && y == 1) move_angle = 1;
            else if (x == 1 && y == 0) move_angle = 2;
            
            mgr_->setAction(world_idx, move_amount, move_angle, r);
            actions.push_back({static_cast<uint32_t>(frame), world_idx, move_amount, move_angle, r});
            
            frame++;
        },
        [&]() { mgr_->step(); },
        []() {}
    );
    
    // Verify the sequence
    ASSERT_EQ(actions.size(), 5);
    
    // Frame 0: Forward only
    EXPECT_EQ(actions[0].move_amount, move_amount::SLOW);
    EXPECT_EQ(actions[0].move_angle, move_angle::FORWARD);
    EXPECT_EQ(actions[0].rotate, rotate::NONE);
    
    // Frame 1: Forward + turn right
    EXPECT_EQ(actions[1].move_amount, move_amount::SLOW);
    EXPECT_EQ(actions[1].move_angle, move_angle::FORWARD);
    EXPECT_EQ(actions[1].rotate, rotate::SLOW_RIGHT);
    
    // Frame 2: Forward-right
    EXPECT_EQ(actions[2].move_amount, move_amount::SLOW);
    EXPECT_EQ(actions[2].move_angle, move_angle::FORWARD_RIGHT);
    EXPECT_EQ(actions[2].rotate, rotate::NONE);
    
    // Frame 3: Just turning
    EXPECT_EQ(actions[3].move_amount, move_amount::STOP);
    EXPECT_EQ(actions[3].rotate, rotate::SLOW_RIGHT);
    
    // Frame 4: Full stop
    EXPECT_EQ(actions[4].move_amount, move_amount::STOP);
    EXPECT_EQ(actions[4].rotate, rotate::NONE);
}