#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <chrono>

namespace madEscape {

// Forward declaration
class Manager;

// Standalone state machine for recording/replay logic
class RecordReplayStateMachine {
public:
    enum State {
        Idle,
        InitialPaused,
        RecordingPaused,
        Recording,
        ReplayingPaused,
        Replaying,
        ReplayFinished
    };
    
    State getState() const { return state_; }
    
    // State transitions - all testable
    void startRecording();
    void startReplay();
    void startInitialPause();
    void togglePause();
    void finishReplay();
    void stop();
    
    bool shouldRecordFrame() const;
    bool shouldAdvanceReplay() const;
    bool isPaused() const;
    bool isRecording() const;
    bool isReplaying() const;
    
private:
    State state_ = Idle;
};

// Manages frame-by-frame action batching
class FrameActionManager {
public:
    FrameActionManager(uint32_t num_worlds);
    
    void setAction(uint32_t world, int32_t move_amount, 
                  int32_t move_angle, int32_t rotate);
    void resetToDefaults();
    const std::vector<int32_t>& getFrameActions() const;
    bool hasChanges() const { return has_changes_; }
    
private:
    uint32_t num_worlds_;
    std::vector<int32_t> frame_actions_;
    bool has_changes_ = false;
};

// Core viewer logic separated from I/O
class ViewerCore {
public:
    struct Config {
        uint32_t num_worlds;
        uint32_t rand_seed;
        bool auto_reset;
        std::string load_path;
        std::string record_path;
        std::string replay_path;
        bool start_paused = false;
        float pause_delay_seconds = 0.0f;
    };
    
    struct InputEvent {
        enum Type { KeyPress, KeyRelease, KeyHit };
        enum Key { W, A, S, D, Q, E, R, T, Space, Shift };
        Type type;
        Key key;
    };
    
    struct FrameState {
        bool is_paused;
        bool is_recording;
        bool has_replay;
        bool should_exit;
        std::vector<int32_t> frame_actions;
    };

    ViewerCore(const Config& cfg, Manager* mgr);
    ~ViewerCore();
    
    // Pure logic functions - fully testable
    void handleInput(int world_idx, const InputEvent& event);
    void updateFrameActions(int world_idx, int agent_idx);
    void updateFrame();  // Handles timing and pause logic
    void stepSimulation();  // Actually steps the simulation
    FrameState getFrameState() const;
    
    // Recording/replay control
    void startRecording(const std::string& path);
    void stopRecording();
    void loadReplay(const std::string& path);
    bool loadReplayActions();
    
    // Trajectory tracking control
    void toggleTrajectoryTracking(int world_idx);
    bool isTrackingTrajectory(int world_idx) const;
    
    // Compass tensor display
    void printCompassTensor(int world_idx) const;
    
    // For testing - access internal state
    const RecordReplayStateMachine& getStateMachine() const { return state_machine_; }
    const FrameActionManager& getActionManager() const { return action_manager_; }
    
private:
    // Core components
    Manager* mgr_;
    RecordReplayStateMachine state_machine_;
    FrameActionManager action_manager_;
    Config config_;
    
    // Tracking state
    bool track_trajectory_ = false;
    int32_t track_world_idx_ = 0;
    int32_t track_agent_idx_ = 0;
    
    // Input state per world
    static constexpr int MAX_WORLDS = 256;
    struct InputState {
        bool keys_pressed[10] = {}; // Maps to Key enum
        bool keys_hit[10] = {};
        int32_t move_x = 0;
        int32_t move_y = 0;
        int32_t rotate = 2;  // Default no rotation
        bool shift_pressed = false;
    };
    InputState input_state_[MAX_WORLDS];
    
    // Convert input state to actions
    void computeActionsFromInput(int world_idx);
    void resetInputState(int world_idx);
    void applyFrameActions();
    bool should_exit_ = false;
    
    // Pause timer for auto-resume
    std::chrono::steady_clock::time_point pause_start_time_;
    bool initial_pause_active_ = false;
};

} // namespace madEscape