#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>
#include <madrona/heap_array.hpp>

#include <madrona/render/render_mgr.hpp>
#include "replay_metadata.hpp"

namespace madEscape {

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t randSeed; // Seed for random world gen
        bool autoReset; // Immediately generate new world on episode end
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
        madrona::render::APIBackend *extRenderAPI = nullptr;
        madrona::render::GPUDevice *extRenderDev = nullptr;
        bool enableTrajectoryTracking = false; // Print agent trajectories to stdout
    };

    Manager(const Config &cfg);
    ~Manager();

    void step();

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    madrona::py::Tensor resetTensor() const;
    madrona::py::Tensor actionTensor() const;
    madrona::py::Tensor rewardTensor() const;
    madrona::py::Tensor doneTensor() const;
    madrona::py::Tensor selfObservationTensor() const;
    madrona::py::Tensor stepsRemainingTensor() const;
    madrona::py::Tensor progressTensor() const;
    madrona::py::Tensor rgbTensor() const;
    madrona::py::Tensor depthTensor() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    void triggerReset(int32_t world_idx);
    void setAction(int32_t world_idx,
                   int32_t move_amount,
                   int32_t move_angle,
                   int32_t rotate);
    
    // Trajectory logging for debugging
    void enableTrajectoryLogging(int32_t world_idx, int32_t agent_idx, std::optional<const char*> filename = std::nullopt);
    void disableTrajectoryLogging();

    madrona::render::RenderManager & getRenderManager();

    // Replay functionality
    struct ReplayData {
        madrona::escape_room::ReplayMetadata metadata;
        madrona::HeapArray<int32_t> actions;
    };
    
    // Recording functionality
    void startRecording(const std::string& filepath, uint32_t seed);
    void stopRecording();
    bool isRecording() const;
    void recordActions(const std::vector<int32_t>& frame_actions);
    
    // Replay functionality
    bool loadReplay(const std::string& filepath);
    bool hasReplay() const;
    bool replayStep();
    uint32_t getCurrentReplayStep() const;
    uint32_t getTotalReplaySteps() const;
    const ReplayData* getReplayData() const;
    
    // Static method to read replay metadata without creating a Manager
    static std::optional<madrona::escape_room::ReplayMetadata> readReplayMetadata(const std::string& filepath);

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
