#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <fstream>
#include <cstring>
#include "madrona_escape_room_c_api.h"

// Enhanced mock window manager for headless testing
class EnhancedMockWindowManager {
public:
    struct WindowHandle {
        void* ptr;
        void* get() const { return ptr; }
    };
    
    struct GPUHandle {
        void* device() const { return device_ptr; }
        void* device_ptr;
    };
    
    struct GPUAPIManager {
        enum Backend { Vulkan, OpenGL };
        Backend backend() const { return Backend::Vulkan; }
    };
    
    EnhancedMockWindowManager() : api_manager_(std::make_unique<GPUAPIManager>()) {}
    
    WindowHandle makeWindow(const std::string& title, int width, int height) {
        window_created_ = true;
        window_title_ = title;
        window_width_ = width;
        window_height_ = height;
        return {reinterpret_cast<void*>(0x1234)};
    }
    
    GPUHandle initGPU(uint32_t gpu_id, std::vector<void*> windows) {
        gpu_initialized_ = true;
        gpu_id_ = gpu_id;
        num_windows_ = windows.size();
        return {reinterpret_cast<void*>(0x5678)};
    }
    
    GPUAPIManager& gpuAPIManager() { return *api_manager_; }
    
    // Test helpers
    bool wasWindowCreated() const { return window_created_; }
    bool wasGPUInitialized() const { return gpu_initialized_; }
    const std::string& getWindowTitle() const { return window_title_; }
    int getWindowWidth() const { return window_width_; }
    int getWindowHeight() const { return window_height_; }
    uint32_t getGPUID() const { return gpu_id_; }
    
private:
    bool window_created_ = false;
    bool gpu_initialized_ = false;
    std::string window_title_;
    int window_width_ = 0;
    int window_height_ = 0;
    uint32_t gpu_id_ = 0;
    size_t num_windows_ = 0;
    std::unique_ptr<GPUAPIManager> api_manager_;
};

// Action recorder to capture Manager actions
class ActionRecorder {
public:
    struct RecordedAction {
        uint32_t step;
        int32_t world_idx;
        int32_t move_amount;
        int32_t move_angle;
        int32_t rotate;
        
        bool operator==(const RecordedAction& other) const {
            return step == other.step &&
                   world_idx == other.world_idx &&
                   move_amount == other.move_amount &&
                   move_angle == other.move_angle &&
                   rotate == other.rotate;
        }
    };
    
    void recordAction(uint32_t step, int32_t world_idx, 
                     int32_t move_amount, int32_t move_angle, int32_t rotate) {
        actions_.push_back({step, world_idx, move_amount, move_angle, rotate});
    }
    
    void clear() { actions_.clear(); current_step_ = 0; }
    void nextStep() { current_step_++; }
    
    const std::vector<RecordedAction>& getActions() const { return actions_; }
    size_t getActionCount() const { return actions_.size(); }
    uint32_t getCurrentStep() const { return current_step_; }
    
    RecordedAction getLastAction() const {
        return actions_.empty() ? RecordedAction{} : actions_.back();
    }
    
    bool hasAction(int32_t world_idx, uint32_t step) const {
        for (const auto& action : actions_) {
            if (action.world_idx == world_idx && action.step == step) {
                return true;
            }
        }
        return false;
    }
    
private:
    std::vector<RecordedAction> actions_;
    uint32_t current_step_ = 0;
};

// Input simulator for generating keyboard events
class InputSimulator {
public:
    enum Key {
        W = 0, A, S, D, Q, E, R, T, Space, Shift,
        Key_Count
    };
    
    struct UserInput {
        bool keyPressed(Key key) const {
            return pressed_keys_[key];
        }
        
        bool keyHit(Key key) const {
            return hit_keys_[key];
        }
        
        bool pressed_keys_[Key_Count] = {};
        bool hit_keys_[Key_Count] = {};
    };
    
    // Set pressed state (held down)
    void pressKey(Key key) {
        input_.pressed_keys_[key] = true;
    }
    
    void releaseKey(Key key) {
        input_.pressed_keys_[key] = false;
    }
    
    // Set hit state (just pressed this frame)
    void hitKey(Key key) {
        input_.hit_keys_[key] = true;
        frame_hits_.push_back(key);
    }
    
    // Clear frame-specific state
    void nextFrame() {
        // Clear all hit states
        for (int i = 0; i < Key_Count; i++) {
            input_.hit_keys_[i] = false;
        }
        frame_hits_.clear();
    }
    
    const UserInput& getInput() const { return input_; }
    
    // Test helpers
    void simulateMovement(int x, int y, bool shift = false) {
        releaseAll();
        if (x > 0) pressKey(D);
        if (x < 0) pressKey(A);
        if (y > 0) pressKey(W);
        if (y < 0) pressKey(S);
        if (shift) pressKey(Shift);
    }
    
    void simulateRotation(int direction) {
        releaseAll();
        if (direction > 0) pressKey(E);
        if (direction < 0) pressKey(Q);
    }
    
    void releaseAll() {
        for (int i = 0; i < Key_Count; i++) {
            input_.pressed_keys_[i] = false;
            input_.hit_keys_[i] = false;
        }
    }
    
private:
    UserInput input_;
    std::vector<Key> frame_hits_;
};

// Mock viewer for testing
class MockViewer {
public:
    using KeyboardKey = InputSimulator::Key;
    using UserInput = InputSimulator::UserInput;
    
    MockViewer([[maybe_unused]] uint32_t num_worlds) 
        : current_world_(0),
          is_running_(true) {}
    
    // Simulate the viewer loop with callbacks
    template<typename WorldCallback, typename AgentCallback, typename StepCallback>
    void loop(WorldCallback world_cb, AgentCallback agent_cb, 
              StepCallback step_cb, std::function<void()> cleanup_cb) {
        
        // Run for a fixed number of frames in test mode
        const int max_frames = test_frame_limit_;
        
        for (int frame = 0; frame < max_frames && is_running_; frame++) {
            // Process world-level input (R, T, Space)
            world_cb(current_world_, input_sim_.getInput());
            
            // Process agent control input (WASD, Q/E) 
            agent_cb(current_world_, 0, input_sim_.getInput());
            
            // Step simulation
            step_cb();
            
            // Move to next frame
            input_sim_.nextFrame();
            frames_processed_++;
        }
        
        // Cleanup
        cleanup_cb();
    }
    
    void stopLoop() { is_running_ = false; }
    
    // Test control methods
    void setCurrentWorld(uint32_t world) { current_world_ = world; }
    void setFrameLimit(int limit) { test_frame_limit_ = limit; }
    InputSimulator& getInputSimulator() { return input_sim_; }
    int getFramesProcessed() const { return frames_processed_; }
    
private:
    uint32_t current_world_;
    bool is_running_;
    InputSimulator input_sim_;
    int test_frame_limit_ = 10;  // Default test limit
    int frames_processed_ = 0;
};

// Test manager wrapper with recording capabilities
class TestManagerWrapper {
public:
    TestManagerWrapper(MER_ManagerHandle handle) : handle_(handle) {}
    
    void setAction(int32_t world_idx, int32_t move_amount, 
                  int32_t move_angle, int32_t rotate) {
        mer_set_action(handle_, world_idx, move_amount, move_angle, rotate);
        recorder_.recordAction(recorder_.getCurrentStep(), world_idx, 
                             move_amount, move_angle, rotate);
    }
    
    void step() {
        mer_step(handle_);
        recorder_.nextStep();
    }
    
    void triggerReset(int32_t world_idx) {
        mer_trigger_reset(handle_, world_idx);
        resets_.push_back({recorder_.getCurrentStep(), world_idx});
    }
    
    void enableTrajectoryLogging(int32_t world, int32_t agent, const char* file) {
        mer_enable_trajectory_logging(handle_, world, agent, file);
        trajectory_enabled_ = true;
        trajectory_world_ = world;
        trajectory_agent_ = agent;
    }
    
    void disableTrajectoryLogging() {
        mer_disable_trajectory_logging(handle_);
        trajectory_enabled_ = false;
    }
    
    bool startRecording(const char* filepath) {
        MER_Result result = mer_start_recording(handle_, filepath);
        if (result == MER_SUCCESS) {
            is_recording_ = true;
            recording_file_ = filepath;
            return true;
        }
        return false;
    }
    
    void stopRecording() {
        mer_stop_recording(handle_);
        is_recording_ = false;
    }
    
    // NOTE: loadReplay() removed - use Manager::fromReplay() to create managers with replay data
    // void loadReplay(const char* filepath) { // REMOVED - API no longer available
    //     mer_load_replay(handle_, filepath);
    //     has_replay_ = true;
    //     replay_file_ = filepath;
    // }
    
    bool replayStep() {
        bool success;
        mer_replay_step(handle_, &success);
        return success;
    }

    bool hasChecksumFailed() {
        bool failed;
        mer_has_checksum_failed(handle_, &failed);
        return failed;
    }

    bool hasReplay() const { return has_replay_; }
    bool isRecording() const { return is_recording_; }

    // Test helpers
    const ActionRecorder& getRecorder() const { return recorder_; }
    const std::vector<std::pair<uint32_t, int32_t>>& getResets() const { return resets_; }
    bool isTrajectoryEnabled() const { return trajectory_enabled_; }
    int32_t getTrajectoryWorld() const { return trajectory_world_; }
    int32_t getTrajectoryAgent() const { return trajectory_agent_; }
    
    MER_ManagerHandle getHandle() { return handle_; }
    
private:
    MER_ManagerHandle handle_;
    ActionRecorder recorder_;
    std::vector<std::pair<uint32_t, int32_t>> resets_;  // (step, world_idx)
    bool trajectory_enabled_ = false;
    int32_t trajectory_world_ = -1;
    int32_t trajectory_agent_ = -1;
    bool is_recording_ = false;
    bool has_replay_ = false;
    std::string recording_file_;
    std::string replay_file_;
};

// Test file manager for cleanup
class TestFileManager {
public:
    ~TestFileManager() {
        cleanup();
    }
    
    void addFile(const std::string& filename) {
        files_.push_back(filename);
    }
    
    void cleanup() {
        for (const auto& file : files_) {
            std::remove(file.c_str());
        }
        files_.clear();
    }
    
    bool fileExists(const std::string& filename) {
        std::ifstream file(filename);
        return file.good();
    }
    
    size_t getFileSize(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        return file.is_open() ? static_cast<size_t>(file.tellg()) : 0;
    }
    
private:
    std::vector<std::string> files_;
};