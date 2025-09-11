#include "viewer_core.hpp"
#include "mgr.hpp"
#include "consts.hpp"
#include <cassert>
#include <chrono>
#include <iostream>

namespace madEscape {

// RecordReplayStateMachine implementation
void RecordReplayStateMachine::startRecording() {
    if (state_ == Idle) {
        state_ = RecordingPaused;
        // Start paused as per viewer behavior
    }
}

void RecordReplayStateMachine::startReplay() {
    if (state_ == Idle) {
        state_ = Replaying;
        // Replay starts immediately (not paused)
    }
}

void RecordReplayStateMachine::startInitialPause() {
    if (state_ == Idle) {
        state_ = InitialPaused;
    }
}

void RecordReplayStateMachine::togglePause() {
    switch (state_) {
        case Idle:
            // Allow pausing during normal operation
            state_ = InitialPaused;
            break;
        case InitialPaused:
            state_ = Idle;
            break;
        case RecordingPaused:
            state_ = Recording;
            break;
        case Recording:
            state_ = RecordingPaused;
            break;
        case ReplayingPaused:
            state_ = Replaying;
            break;
        case Replaying:
            state_ = ReplayingPaused;
            break;
        default:
            // No-op for other states
            break;
    }
}

void RecordReplayStateMachine::finishReplay() {
    if (state_ == Replaying || state_ == ReplayingPaused) {
        state_ = ReplayFinished;
    }
}

void RecordReplayStateMachine::stop() {
    state_ = Idle;
}

bool RecordReplayStateMachine::shouldRecordFrame() const {
    return state_ == Recording;
}

bool RecordReplayStateMachine::shouldAdvanceReplay() const {
    return state_ == Replaying;
}

bool RecordReplayStateMachine::isPaused() const {
    return state_ == InitialPaused || state_ == RecordingPaused || state_ == ReplayingPaused;
}

bool RecordReplayStateMachine::isRecording() const {
    return state_ == Recording || state_ == RecordingPaused;
}

bool RecordReplayStateMachine::isReplaying() const {
    return state_ == Replaying || state_ == ReplayingPaused;
}

// FrameActionManager implementation
FrameActionManager::FrameActionManager(uint32_t num_worlds)
    : num_worlds_(num_worlds),
      frame_actions_(num_worlds * consts::viewer::actionComponentsPerAgent) {
    resetToDefaults();
}

void FrameActionManager::setAction(uint32_t world, int32_t move_amount, 
                                   int32_t move_angle, int32_t rotate) {
    if (world >= num_worlds_) return;
    
    uint32_t base = world * consts::viewer::actionComponentsPerAgent;
    frame_actions_[base] = move_amount;
    frame_actions_[base + 1] = move_angle;
    frame_actions_[base + 2] = rotate;
    has_changes_ = true;
}

void FrameActionManager::resetToDefaults() {
    for (uint32_t i = 0; i < num_worlds_; i++) {
        frame_actions_[i * consts::viewer::actionComponentsPerAgent] = consts::action::move_amount::STOP;      // move_amount = 0 (stop)
        frame_actions_[i * consts::viewer::actionComponentsPerAgent + 1] = consts::action::move_angle::FORWARD;  // move_angle = 0 (forward)
        frame_actions_[i * consts::viewer::actionComponentsPerAgent + 2] = consts::action::rotate::NONE;  // rotate = 2 (no rotation)
    }
    has_changes_ = false;
}

const std::vector<int32_t>& FrameActionManager::getFrameActions() const {
    return frame_actions_;
}

// ViewerCore implementation
ViewerCore::ViewerCore(const Config& cfg, Manager* mgr)
    : mgr_(mgr),
      action_manager_(cfg.num_worlds),
      config_(cfg) {
    
    // Initialize input state
    for (int i = 0; i < MAX_WORLDS; i++) {
        resetInputState(i);
    }
    
    // Setup initial state based on config
    if (!config_.replay_path.empty()) {
        state_machine_.startReplay();
        if (config_.start_paused) {
            state_machine_.togglePause();  // Transition from Replaying to ReplayingPaused
            initial_pause_active_ = true;
            pause_start_time_ = std::chrono::steady_clock::now();
        }
    } else if (!config_.record_path.empty()) {
        state_machine_.startRecording();
        printf("Recording mode: Starting PAUSED (press SPACE to start recording)\n");
    } else if (config_.start_paused) {
        state_machine_.startInitialPause();
        initial_pause_active_ = true;
        pause_start_time_ = std::chrono::steady_clock::now();
    }
}

ViewerCore::~ViewerCore() {
    if (state_machine_.isRecording()) {
        stopRecording();
    }
}

void ViewerCore::handleInput(int world_idx, const InputEvent& event) {
    if (world_idx >= MAX_WORLDS) return;
    
    auto& input = input_state_[world_idx];
    
    // Handle special keys first
    if (event.key == InputEvent::Space && event.type == InputEvent::KeyHit) {
        // If we're in initial pause, clear the flag when manually resuming
        if (state_machine_.getState() == RecordReplayStateMachine::InitialPaused) {
            initial_pause_active_ = false;
        }
        state_machine_.togglePause();
        printf("Simulation %s\n", state_machine_.isPaused() ? "PAUSED" : "RESUMED");
        return;
    }
    
    if (event.key == InputEvent::R && event.type == InputEvent::KeyHit) {
        mgr_->triggerReset(world_idx);
        return;
    }
    
    if (event.key == InputEvent::T && event.type == InputEvent::KeyHit) {
        toggleTrajectoryTracking(world_idx);
        return;
    }
    
    if (event.key == InputEvent::M && event.type == InputEvent::KeyHit) {
        config_.multi_world_grid = !config_.multi_world_grid;
        printf("Multi-world grid: %s\n", config_.multi_world_grid ? "ON" : "OFF");
        return;
    }
    
    // Update input state for movement keys
    bool pressed = (event.type == InputEvent::KeyPress);
    
    switch (event.key) {
        case InputEvent::W:
            input.keys_pressed[InputEvent::W] = pressed;
            break;
        case InputEvent::A:
            input.keys_pressed[InputEvent::A] = pressed;
            break;
        case InputEvent::S:
            input.keys_pressed[InputEvent::S] = pressed;
            break;
        case InputEvent::D:
            input.keys_pressed[InputEvent::D] = pressed;
            break;
        case InputEvent::Q:
            input.keys_pressed[InputEvent::Q] = pressed;
            break;
        case InputEvent::E:
            input.keys_pressed[InputEvent::E] = pressed;
            break;
        case InputEvent::Shift:
            input.shift_pressed = pressed;
            break;
        default:
            break;
    }
    
    // For key hit events
    if (event.type == InputEvent::KeyHit) {
        input.keys_hit[event.key] = true;
    }
}

void ViewerCore::computeActionsFromInput(int world_idx) {
    if (world_idx >= MAX_WORLDS) return;
    
    const auto& input = input_state_[world_idx];
    
    // Compute movement direction
    int32_t x = 0;
    int32_t y = 0;
    int32_t r = 2;  // Default no rotation
    
    if (input.keys_pressed[InputEvent::W]) y += 1;
    if (input.keys_pressed[InputEvent::S]) y -= 1;
    if (input.keys_pressed[InputEvent::D]) x += 1;
    if (input.keys_pressed[InputEvent::A]) x -= 1;
    
    // NOTE: Non-standard rotation encoding - lower values = left turn
    if (input.keys_pressed[InputEvent::Q]) {
        r -= input.shift_pressed ? 2 : 1;  // Q left: decrease value (non-standard)
    }
    if (input.keys_pressed[InputEvent::E]) {
        r += input.shift_pressed ? 2 : 1;  // E right: increase value (non-standard)
    }
    
    // Compute move amount
    int32_t move_amount;
    if (x == 0 && y == 0) {
        move_amount = 0;
    } else if (input.shift_pressed) {
        move_amount = consts::numMoveAmountBuckets - 1;
    } else {
        move_amount = 1;
    }
    
    // Compute move angle
    int32_t move_angle;
    if (x == 0 && y == 1) {
        move_angle = consts::action::move_angle::FORWARD;
    } else if (x == 1 && y == 1) {
        move_angle = consts::action::move_angle::FORWARD_RIGHT;
    } else if (x == 1 && y == 0) {
        move_angle = consts::action::move_angle::RIGHT;
    } else if (x == 1 && y == -1) {
        move_angle = consts::action::move_angle::BACKWARD_RIGHT;
    } else if (x == 0 && y == -1) {
        move_angle = consts::action::move_angle::BACKWARD;
    } else if (x == -1 && y == -1) {
        move_angle = consts::action::move_angle::BACKWARD_LEFT;
    } else if (x == -1 && y == 0) {
        move_angle = consts::action::move_angle::LEFT;
    } else if (x == -1 && y == 1) {
        move_angle = consts::action::move_angle::FORWARD_LEFT;
    } else {
        move_angle = consts::action::move_angle::FORWARD;
    }
    
    // Store computed actions
    input_state_[world_idx].move_x = x;
    input_state_[world_idx].move_y = y;
    input_state_[world_idx].rotate = r;
    
    // Set action in manager
    mgr_->setAction(world_idx, move_amount, move_angle, r);
    
    // Always update frame actions (needed for testing and monitoring)
    action_manager_.setAction(world_idx, move_amount, move_angle, r);
}

void ViewerCore::updateFrameActions(int world_idx, int /* agent_idx */) {
    // Compute actions from current input state
    computeActionsFromInput(world_idx);
}

void ViewerCore::updateFrame() {
    // Handle auto-resume timer for initial pause
    if (initial_pause_active_ && state_machine_.getState() == RecordReplayStateMachine::InitialPaused) {
        if (config_.pause_delay_seconds > 0.0f) {
            // Calculate actual elapsed time
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<float>>(now - pause_start_time_).count();
            
            // Debug: Print timer progress every 0.5 seconds
            static auto last_print = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() > 500) {
                printf("Timer: %.2f / %.2f seconds elapsed\n", elapsed, config_.pause_delay_seconds);
                last_print = now;
            }
            
            if (elapsed >= config_.pause_delay_seconds) {
                printf("Auto-resuming after %.1f seconds\n", elapsed);
                state_machine_.togglePause();
                initial_pause_active_ = false;
            }
        }
    }
    
    // Only step the simulation if not paused
    if (!state_machine_.isPaused()) {
        stepSimulation();
    }
}

void ViewerCore::stepSimulation() {
    // Load actions from replay if replaying
    bool replay_finished = false;
    if (state_machine_.isReplaying()) {
        replay_finished = loadReplayActions();
    }
    
    // Don't call recordActions here - Manager::step() handles recording internally
    // when recording is active
    
    // Always step the simulation
    mgr_->step();
    
    // After stepping, check if replay finished
    if (replay_finished) {
        state_machine_.finishReplay();
        should_exit_ = true;
    }
    
    // Reset frame actions for next frame
    if (state_machine_.isRecording()) {
        action_manager_.resetToDefaults();
    }
}

ViewerCore::FrameState ViewerCore::getFrameState() const {
    return {
        .is_paused = state_machine_.isPaused(),
        .is_recording = state_machine_.isRecording(),
        .has_replay = state_machine_.isReplaying(),
        .should_exit = should_exit_,
        .frame_actions = action_manager_.getFrameActions()
    };
}

void ViewerCore::startRecording(const std::string& path) {
    Result result = mgr_->startRecording(path);
    if (result != Result::Success) {
        // Handle recording failure - could log error or notify user
        std::cerr << "Failed to start recording to: " << path << std::endl;
        return;
    }
    state_machine_.startRecording();
}

void ViewerCore::stopRecording() {
    mgr_->stopRecording();
    state_machine_.stop();
}

void ViewerCore::loadReplay(const std::string& path) {
    mgr_->loadReplay(path);
    state_machine_.startReplay();
}

bool ViewerCore::loadReplayActions() {
    if (!mgr_->hasReplay()) {
        return true;  // Finished
    }
    
    // Load actions from replay data and increment replay counter
    // Does NOT step the simulation - that happens separately in stepSimulation()
    return mgr_->replayStep();
}

void ViewerCore::toggleTrajectoryTracking(int world_idx) {
    if (track_trajectory_ && track_world_idx_ == world_idx) {
        mgr_->disableTrajectoryLogging();
        track_trajectory_ = false;
        printf("Trajectory logging disabled\n");
    } else {
        mgr_->enableTrajectoryLogging(world_idx, 0, std::nullopt);
        track_trajectory_ = true;
        track_world_idx_ = world_idx;
        track_agent_idx_ = 0;
        printf("Trajectory logging enabled for World %d, Agent 0\n", world_idx);
    }
}

bool ViewerCore::isTrackingTrajectory(int world_idx) const {
    return track_trajectory_ && track_world_idx_ == world_idx;
}

void ViewerCore::printCompassTensor(int world_idx) const {
    // Validate world index
    if (world_idx < 0 || world_idx >= (int)config_.num_worlds) {
        printf("ERROR: Invalid world_idx: %d. Must be between 0 and %u\n", 
               world_idx, config_.num_worlds - 1);
        return;
    }
    
    // Get compass tensor from manager
    auto compass = mgr_->compassTensor();
    
    // Calculate agent index (assuming agent 0)
    int32_t idx = world_idx * madEscape::consts::numAgents + 0;
    
    // Get compass data pointer
    const float* compass_data = (const float*)compass.devicePtr();
    if (!compass_data) {
        printf("ERROR: Could not access compass tensor data\n");
        return;
    }
    
    // Find the index of the one-hot value
    int active_index = -1;
    for (int i = 0; i < 128; i++) {
        float value = compass_data[idx * 128 + i];
        if (value > 0.5f) {  // Should be 1.0 in one-hot encoding
            active_index = i;
            break;
        }
    }
    
    if (active_index >= 0) {
        printf("Compass [world %d]: index %d\n", world_idx, active_index);
    } else {
        printf("Compass [world %d]: no active index found\n", world_idx);
    }
}

void ViewerCore::resetInputState(int world_idx) {
    if (world_idx >= MAX_WORLDS) return;
    
    auto& input = input_state_[world_idx];
    for (int i = 0; i < 11; i++) {
        input.keys_pressed[i] = false;
        input.keys_hit[i] = false;
    }
    input.move_x = 0;
    input.move_y = 0;
    input.rotate = 2;
    input.shift_pressed = false;
}

void ViewerCore::applyFrameActions() {
    const auto& actions = action_manager_.getFrameActions();
    for (uint32_t i = 0; i < config_.num_worlds; i++) {
        uint32_t base = i * consts::viewer::actionComponentsPerAgent;
        mgr_->setAction(i, actions[base], actions[base + 1], actions[base + 2]);
    }
}

} // namespace madEscape